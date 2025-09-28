import torch
import torch.nn as nn
import torch.nn.functional as F

from streamvggt.heads.dpt_head import DPTHead


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=2, stride=2)
        
        self.down2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=2, stride=2)
        
        self.down3 = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim*4, hidden_dim*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.Conv2d(hidden_dim*4, hidden_dim*4, kernel_size=2, stride=2)
        
        self.down4 = nn.Sequential(
            nn.Conv2d(hidden_dim*4, hidden_dim*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim*8, hidden_dim*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*8),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        down1 = self.down1(x)
        pool1 = self.pool1(down1)
        
        down2 = self.down2(pool1)
        pool2 = self.pool2(down2)
        
        down3 = self.down3(pool2)
        pool3 = self.pool3(down3)
        
        down4 = self.down4(pool3)

        return {
            'down1': down1,
            'down2': down2,
            'down3': down3,
            'down4': down4,
            'sizes': {
                'down1': down1.shape[2:],  # (H, W)
                'down2': down2.shape[2:],
                'down3': down3.shape[2:],
                'down4': down4.shape[2:]
            }
        }

class UNetDecoder(nn.Module):
    def __init__(self, hidden_dim, out_channels):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(hidden_dim*8, hidden_dim*4, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(hidden_dim*8, hidden_dim*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim*4, hidden_dim*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim*4, hidden_dim*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(hidden_dim*2, hidden_dim, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, encoder_feats):
        down1 = encoder_feats['down1']
        down2 = encoder_feats['down2']
        down3 = encoder_feats['down3']
        down4 = encoder_feats['down4']
        sizes = encoder_feats['sizes']
        
        up1 = self.up1(down4)

        up1_aligned = F.interpolate(up1, size=sizes['down3'], mode='bilinear', align_corners=False)
        merge1 = torch.cat([up1_aligned, down3], dim=1)  
        conv1 = self.conv1(merge1)

        up2 = self.up2(conv1)
        up2_aligned = F.interpolate(up2, size=sizes['down2'], mode='bilinear', align_corners=False)
        merge2 = torch.cat([up2_aligned, down2], dim=1)
        conv2 = self.conv2(merge2)

        up3 = self.up3(conv2)
        up3_aligned = F.interpolate(up3, size=sizes['down1'], mode='bilinear', align_corners=False)
        merge3 = torch.cat([up3_aligned, down1], dim=1)
        conv3 = self.conv3(merge3)
        
        out = self.final_conv(conv3)
        return out


class LanguageHead_Multi(nn.Module):
    def __init__(self, dim_in, patch_size=14, features=128, lang_dim=6, hidden_dim=32): 
        super().__init__()
        self.lang_dim = lang_dim

        self.feature_extractor = DPTHead(
            dim_in=dim_in,
            patch_size=patch_size,
            features=features,
            feature_only=True,  
            down_ratio=1,       
            pos_embed=False
        )

        # self.shared_encoder = UNetEncoder(in_channels=features, hidden_dim=hidden_dim)
        self.lang_encoder = UNetEncoder(in_channels=features, hidden_dim=hidden_dim)
        self.rgb_encoder = UNetEncoder(in_channels=features, hidden_dim=hidden_dim)

        self.lang_decoder_small = UNetDecoder(hidden_dim=hidden_dim, out_channels=lang_dim)
        self.lang_decoder_middle = UNetDecoder(hidden_dim=hidden_dim, out_channels=lang_dim)
        self.lang_decoder_large = UNetDecoder(hidden_dim=hidden_dim, out_channels=lang_dim)

        self.rgb_decoder = nn.Sequential(
            UNetDecoder(hidden_dim=hidden_dim, out_channels=3),
            nn.Sigmoid()  
        )

    def forward(self, aggregated_tokens_list, images, patch_start_idx, img_height=518, img_width=518):
        feature_maps = self.feature_extractor(aggregated_tokens_list, images, patch_start_idx)  # (B, S, C, H, W)
        B, S, C, H, W = feature_maps.shape
        flattened = feature_maps.view(B * S, C, H, W)  # (B*S, C, H, W)
        
        # multi_scale_feats = self.shared_encoder(flattened)  # [down1, down2, down3, down4]
        lang_feat = self.lang_encoder(flattened)

        lang_feat_small = self.lang_decoder_small(lang_feat)
        lang_feat_middle = self.lang_decoder_middle(lang_feat)
        lang_feat_large = self.lang_decoder_large(lang_feat)
        lang_feat = torch.cat([lang_feat_small.unsqueeze(1), lang_feat_middle.unsqueeze(1), lang_feat_large.unsqueeze(1)], dim=1)

        rgb_feat = self.rgb_encoder(flattened)
        rgb_image = self.rgb_decoder(rgb_feat)  # (B*S, 3, H, W)
        
        if H != img_height or W != img_width:
            lang_feat_reshaped = lang_feat.view(B*S, 3*self.lang_dim, H, W)
            lang_feat = F.interpolate(
                lang_feat_reshaped, 
                size=(img_height, img_width), 
                mode='bilinear', 
                align_corners=False
            ).view(B*S, 3, self.lang_dim, img_height, img_width)
            
            rgb_image = F.interpolate(
                rgb_image, 
                size=(img_height, img_width), 
                mode='bilinear', 
                align_corners=False
            )
        
        lang_feat = lang_feat.view(B, S, 3, self.lang_dim, img_height, img_width)
        rgb_image = rgb_image.view(B, S, 3, img_height, img_width)
        
        return lang_feat, rgb_image
