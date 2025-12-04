import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from head.langhead import LanguageHead_Multi
from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.utils.geometry import unproject_depth_map_to_point_map
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
from utils.utils import predictions_to_glb


class Langvggt(nn.Module):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, lang_dim=6):
        super().__init__()
        self.streamvggt = StreamVGGT(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim) # 4D VGGT pretrained model
        self.lang_head = LanguageHead_Multi(dim_in=2 * embed_dim, patch_size=patch_size, lang_dim=lang_dim)
        # self.lang_video_head = LanguageHead_Video(dim_in=2 * embed_dim, patch_size=patch_size, lang_dim=lang_dim, hidden_dim=64)
        self.lang_dim = lang_dim

    def load_from_pretrained(self, streamvggt_model, lang_head_model):
        streamvggt_ckpt = torch.load(streamvggt_model, map_location="cpu")
        self.streamvggt.load_state_dict(streamvggt_ckpt, strict=True)
        self.streamvggt.eval()
        self.streamvggt.aggregator.cuda()
        self.streamvggt.depth_head.cuda().eval()
        self.streamvggt.camera_head.cuda().eval()
        lang_head_ckpt = torch.load(lang_head_model, map_location="cpu")
        if self.lang_dim == 3:
            self.lang_head.load_state_dict(lang_head_ckpt['model_state_dict'], strict=True)
            self.lang_head.eval().cuda()
        # elif self.lang_dim == 6:
        #     self.lang_video_head.load_state_dict(lang_head_ckpt['model_state_dict'], strict=True)
        #     self.lang_video_head.eval().cuda()
        del streamvggt_ckpt
        del lang_head_ckpt

    @torch.no_grad()
    def inference_long_video(self, frames, img_height=518, img_width=518, save_dir=None, max_num=128):
        past_key_values = [None] * self.streamvggt.aggregator.depth
        # lang_feature_list = []
        import os
        import torchvision
        from utils.utils import pca_compress
        lang_dir = os.path.join(save_dir, 'lang')
        levels = ['small', 'middle', 'large']
        image_dir = os.path.join(save_dir, 'rgb')

        if not os.path.exists(lang_dir):
            os.makedirs(lang_dir, exist_ok=True)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir, exist_ok=True)
        for level in levels:
            if not os.path.exists(os.path.join(lang_dir, level)):
                os.makedirs(os.path.join(lang_dir, level), exist_ok=True)
      
        for i, frame in enumerate(tqdm(frames, desc="Processing images")):
            images = frame["img"].unsqueeze(0) 
            aggregator_output = self.streamvggt.aggregator(
                images, 
                past_key_values=past_key_values,
                use_cache=True, 
                past_frame_idx=i
            )
            
            if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                aggregated_tokens, patch_start_idx, past_key_values = aggregator_output
            else:
                aggregated_tokens, patch_start_idx = aggregator_output
            
            with torch.amp.autocast('cuda', enabled=False):
                lang_feature, pred_image = self.lang_head(aggregated_tokens, images, patch_start_idx, img_height=img_height, img_width=img_width)
                # lang_feature_list.append(lang_feature.squeeze(1))
                if len(lang_feature.shape) == 6:
                    dim_check = lang_feature.shape[3]
                elif len(lang_feature.shape) == 5:
                    dim_check = lang_feature.shape[2]

                for level_num in range(lang_feature.shape[2]):
                    if dim_check <= 3:
                        torchvision.utils.save_image(lang_feature[:, :, level_num].squeeze(0).squeeze(0), 
                                                        os.path.join(lang_dir, levels[level_num], '{0:05d}'.format(i+1) + ".png"))
                    else:
                        torchvision.utils.save_image(pca_compress(lang_feature[:, :, level_num]).squeeze(0).squeeze(0), 
                                                        os.path.join(lang_dir, levels[level_num], '{0:05d}'.format(i+1) + ".png"))
                
                if dim_check <= 3:
                    # torchvision.utils.save_image(lang_feature.squeeze(0).squeeze(0), os.path.join(lang_dir, '{0:05d}'.format(i+1) + ".png"))
                    for level_num in range(lang_feature.shape[2]):
                        torchvision.utils.save_image(lang_feature[:, :, level_num].squeeze(0).squeeze(0), 
                                                        os.path.join(lang_dir, levels[level_num], '{0:05d}'.format(i+1) + ".png"))
                else:
                    torchvision.utils.save_image(pca_compress(lang_feature[:, :, level_num]).squeeze(0).squeeze(0), 
                                            os.path.join(lang_dir, levels[level_num], '{0:05d}'.format(i+1) + ".png"))
                torchvision.utils.save_image(pred_image.squeeze(0).squeeze(0), os.path.join(image_dir, '{0:05d}'.format(i+1) + ".png"))
            
            if past_key_values[0][0].shape[2] > max_num:
                for i in range(self.streamvggt.aggregator.depth):
                    temp_list = list(past_key_values[i])
                    temp_list[0] = temp_list[0][:, :, 1:, :, :]  
                    temp_list[1] = temp_list[1][:, :, 1:, :, :]
                    past_key_values[i] = tuple(temp_list)
                    del temp_list

        return None
    
    @torch.no_grad()
    def save_point_cloud(self, frames, img_height=518, img_width=518, save_dir=None, max_num=128):
        img_height = (img_height // 14) * 14
        img_width = (img_width // 14) * 14
        past_key_values = [None] * self.streamvggt.aggregator.depth
        # lang_feature_list = []
        import os
        import torchvision
        from utils.utils import pca_compress
        lang_dir = os.path.join(save_dir, 'lang')
        levels = ['small', 'middle', 'large']
        image_dir = os.path.join(save_dir, 'rgb')

        if not os.path.exists(lang_dir):
            os.makedirs(lang_dir, exist_ok=True)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir, exist_ok=True)
        for level in levels:
            if not os.path.exists(os.path.join(lang_dir, level)):
                os.makedirs(os.path.join(lang_dir, level), exist_ok=True)
      
        for i, frame in enumerate(tqdm(frames, desc="Processing images")):
            images = frame["img"].unsqueeze(0) 
            aggregator_output = self.streamvggt.aggregator(
                images, 
                past_key_values=past_key_values,
                use_cache=True, 
                past_frame_idx=i
            )
            
            if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                aggregated_tokens_list, patch_start_idx, past_key_values = aggregator_output
            else:
                aggregated_tokens_list, patch_start_idx = aggregator_output
            
            # 2. StreamVGGT Depth head & Camera head
            pose_enc_list = self.streamvggt.camera_head(aggregated_tokens_list)
            camera_token = pose_enc_list[-1]  # pose encoding of the last iteration
            del pose_enc_list
            extrinsics, intrinsics = pose_encoding_to_extri_intri(camera_token, images.shape[-2:])

            depth_map, depth_conf = self.streamvggt.depth_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )
            _, _, C = camera_token.shape
            B, S, H, W, _ = depth_map.shape
            camera_token = camera_token.unsqueeze(3).unsqueeze(3).expand(B, S, C, H, W)
            world_points = unproject_depth_map_to_point_map(depth_map.squeeze(0), extrinsics.squeeze(0), intrinsics.squeeze(0))
        
            supp_tokens = torch.cat([camera_token, depth_map.view(B, S, -1, H, W)], dim=2)
            
            with torch.amp.autocast('cuda', enabled=False):
                # 3. Lang head
                lang_feature, pred_image = self.lang_head(aggregated_tokens_list, images, patch_start_idx, img_height=img_height, img_width=img_width)
                # lang_feature_list.append(lang_feature.squeeze(1))
                rgb_prediction = {
                    'world_points_from_depth': world_points,
                    'depth_conf': depth_conf.cpu().numpy().squeeze(0),
                    'images': pred_image.cpu().numpy().squeeze(0),
                    'extrinsic': extrinsics.cpu().numpy().squeeze(0),
                }
                rgb_scene = predictions_to_glb(rgb_prediction, conf_thres=3.0)
                rgb_scene.export(file_obj=os.path.join(image_dir, '{0:05d}'.format(i+1) + ".glb"))
                del rgb_scene
                for level_num in range(lang_feature.shape[2]):
                    lang_prediction = {
                        'world_points_from_depth': world_points,
                        'depth_conf': depth_conf.cpu().numpy().squeeze(0),
                        'images': lang_feature[:, :, level_num].cpu().numpy().squeeze(0),
                        'extrinsic': extrinsics.cpu().numpy().squeeze(0),
                    }
                    lang_scene = predictions_to_glb(lang_prediction, conf_thres=3.0)
                    lang_scene.export(file_obj=os.path.join(lang_dir, levels[level_num], '{0:05d}'.format(i+1) + ".glb"))
                    del lang_scene
            
            if past_key_values[0][0].shape[2] > max_num:
                for i in range(self.streamvggt.aggregator.depth):
                    temp_list = list(past_key_values[i])
                    temp_list[0] = temp_list[0][:, :, 1:, :, :]  
                    temp_list[1] = temp_list[1][:, :, 1:, :, :]
                    past_key_values[i] = tuple(temp_list)
                    del temp_list

        return None


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def cos_loss(network_output, gt):
    return 1 - F.cosine_similarity(network_output, gt, dim=-1).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()