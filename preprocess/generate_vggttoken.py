import os
import sys
import glob
import numpy as np

import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from streamvggt.models.streamvggt import StreamVGGT

# CATEGORY_LIST = ['americano', 'chickchicken', 'espresso', 'keyboard', 'split-cookie', 'torchocolate']
CATEGORY_LIST = ['keyboard']

local_ckpt_path = "YOUR_STREAMVGGT_CKPT"
streamvggt = StreamVGGT()
ckpt = torch.load(local_ckpt_path, map_location="cpu")
streamvggt.load_state_dict(ckpt, strict=True)
streamvggt.eval()
streamvggt.aggregator.cuda()  # or .to(device)

for cat in CATEGORY_LIST:
    img_root = 'IMG_DATA_ROOT' 
    # cat = 'chickchicken'
    img_dir = os.path.join(img_root, cat, 'rgb', '2x')
    image_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    keys = [os.path.splitext(os.path.basename(p))[0] for p in image_files]

    # save_dir = os.path.join(img_root, cat, 'streamvggt_token_518')
    save_dir = os.path.join(img_root, cat, 'streamvggt_token')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        # with torch.amp.autocast('cuda', dtype=dtype):
        streamvggt.inference_long_video(image_files, save_dir=save_dir, keys=keys, mode="original", max_num=128)
        torch.cuda.empty_cache()

print("All batches processed successfully!")
    