import os, glob, numpy as np
from PIL import Image
import torch
import os
import sys
import numpy as np

CATEGORY_LIST = ["americano", "chickchicken", "espresso", "keyboard", "split-cookie", "torchocolate"]

def collate_fn_img(batch):
    collated = {}
    all_aggregated_tokens = [[] for _ in range(24)] # trick for vggt
    aggregated_tokens_list = []
    for key in batch[0]:
        if key == 'aggregated_tokens_list':
            for pos in range(24):
                for item in batch:
                    all_aggregated_tokens[pos].append(torch.from_numpy(item[key][pos]))
        elif isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([item[key] for item in batch])
        else:
            collated[key] = [item[key] for item in batch]
    for pos in range(24):
        concatenated_tensor = torch.cat((all_aggregated_tokens[pos]), dim=1)
        aggregated_tokens_list.append(concatenated_tensor)
    collated['aggregated_tokens_list'] = aggregated_tokens_list
    del concatenated_tensor
    del aggregated_tokens_list
    return collated


class MultiIMGDataset(torch.utils.data.Dataset):
    def __init__(self, img_root, feat_root='clip_features-language_features_dim3', overfit=False, cat=None):
        """
        mode: 'img' and 'vae'
        """
        # self.feat_root = 'clip_features-language_features_dim3'
        self.feat_root = feat_root
        self.streamvggt_token_root = 'streamvggt_token'

        self.img_root = img_root

        self.samples = []
        if overfit:
            assert cat is not None
            img_dir = os.path.join(img_root, cat, 'rgb', '2x')
            image_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
            keys = [os.path.splitext(os.path.basename(p))[0] for p in image_files]
            print(f"Training with overfit! Category of dataset is {cat}")
            print(f"The path of training dataset is {img_dir}")
            for key in keys:
                self.samples.append((cat, [key]))
        else:
            for cat in CATEGORY_LIST:
                img_dir = os.path.join(img_root, cat, 'rgb', '2x')

                image_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
                keys = [os.path.splitext(os.path.basename(p))[0] for p in image_files]

                for key in keys:
                    self.samples.append((cat, [key]))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cat, clip_keys = self.samples[idx]

        assert len(clip_keys) == 1
        # load frame and features
        for key in clip_keys:
            img_path = os.path.join(self.img_root, cat, 'rgb', '2x', key + '.png')
            
            streamvggt_path = os.path.join(self.img_root, cat, self.streamvggt_token_root, key + '.npy')
            streamvggt_info = np.load(streamvggt_path, allow_pickle=True).item()

            seg_map = torch.from_numpy(np.load(os.path.join(self.img_root, cat, self.feat_root, key + '_s.npy')))
            lang_feat = torch.from_numpy(np.load(os.path.join(self.img_root, cat, self.feat_root, key + '_f.npy')))
            _, self.image_height, self.image_width = seg_map.shape

            y, x = torch.meshgrid(torch.arange(0, self.image_height), torch.arange(0, self.image_width))
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
            seg = seg_map[:, y, x].squeeze(-1).long()
            mask = seg_map != -1
            point_feature_small = lang_feat[seg[1:2]].squeeze(0)
            point_feature_middle = lang_feat[seg[2:3]].squeeze(0)
            point_feature_large = lang_feat[seg[3:4]].squeeze(0)

            mask_small = mask[1:2].reshape(1, self.image_height, self.image_width)
            mask_middle = mask[2:3].reshape(1, self.image_height, self.image_width)
            mask_large = mask[3:4].reshape(1, self.image_height, self.image_width)
            mask = torch.cat([mask_small.unsqueeze(0), mask_middle.unsqueeze(0), mask_large.unsqueeze(0)], dim=0)
            point_feature_small = point_feature_small.reshape(self.image_height, self.image_width, -1).permute(2, 0, 1)
            point_feature_middle = point_feature_middle.reshape(self.image_height, self.image_width, -1).permute(2, 0, 1)
            point_feature_large = point_feature_large.reshape(self.image_height, self.image_width, -1).permute(2, 0, 1)
            point_feature = torch.cat([point_feature_small.unsqueeze(0), point_feature_middle.unsqueeze(0), 
                                        point_feature_large.unsqueeze(0)], dim=0)

        result = {
            'clip_feats': point_feature,
            'seg_maps': mask,
            'aggregated_tokens_list': streamvggt_info['aggregated_tokens_list'],
            'patch_start_idx': streamvggt_info['patch_start_idx'],
            'images_path': img_path,
        }
        
        return result
