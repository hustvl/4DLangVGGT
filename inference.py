import os
import argparse
import torch
import torchvision
import numpy as np
import imageio
from sklearn.decomposition import PCA
from pathlib import Path
from streamvggt.utils.load_fn import load_and_preprocess_images

from models.langvggt import Langvggt

# ---------- utils ----------
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

def pca_compress(rendering):
    feature_map = rendering.permute(1,2,0).cpu().numpy()
    pca = PCA(n_components=3)
    h, w, n = feature_map.shape
    feat_reshaped = feature_map.reshape(-1, n)
    feat_pca = pca.fit_transform(feat_reshaped)
    feat_pca_reshaped = feat_pca.reshape(h, w, 3)
    feat_norm = (feat_pca_reshaped - feat_pca_reshaped.min()) / (feat_pca_reshaped.max() - feat_pca_reshaped.min())
    return torch.from_numpy(feat_norm)

def save_png_tensor(tensor_3ch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchvision.utils.save_image(tensor_3ch, path)

def images_to_video_from_folder(image_folder, output_video, fps=16):
    images = []
    files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    for f in files:
        images.append(imageio.imread(os.path.join(image_folder, f)))
    if len(images) == 0:
        return False
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    imageio.mimwrite(output_video, images, fps=fps)
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream_ckpt", type=str, required=True, help="StreamVGGT checkpoint path")
    parser.add_argument("--lang_ckpt", type=str, required=True, help="LangVGGT checkpoint path")
    parser.add_argument("--image_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--pattern", type=str, default=None, help="Pattern for numbered images, e.g. '{:06d}.png'")
    parser.add_argument("--start_id", type=int, default=1)
    parser.add_argument("--clip_length", type=int, default=None)
    parser.add_argument("--save_path", type=str, default="./lang_output", help="Output base path")
    parser.add_argument("--epoch", type=int, default=100, help="Epoch index for naming outputs")
    parser.add_argument("--lang_dim", type=int, default=3, help="Model lang_dim")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument("--make_video", action="store_true", help="Export mp4 videos")
    parser.add_argument("--save_rgb", action="store_true", help="Also save input RGB frames")
    parser.add_argument("--do_render", action="store_true", help="Run 2D semantic rendering")
    parser.add_argument("--do_4d", action="store_true", help="Run 4D point cloud inference")
    args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    print("Using device:", device)

    print("Initializing model...")
    model = Langvggt(lang_dim=args.lang_dim).to(device)
    model.load_from_pretrained(args.stream_ckpt, args.lang_ckpt)
    model.eval()

    if args.pattern:
        if args.clip_length is None:
            raise ValueError("--clip_length required when using --pattern")
        img_paths = [os.path.join(args.image_dir, args.pattern.format(i))
                     for i in range(args.start_id, args.start_id + args.clip_length)]
    else:
        all_files = sorted([os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        img_paths = all_files[args.start_id - 1: args.start_id - 1 + args.clip_length] if args.clip_length else all_files

    if len(img_paths) == 0:
        raise RuntimeError("No images found")

    images, img_width, img_height = load_and_preprocess_images(img_paths, mode="original")
    views = [{"img": images[i].unsqueeze(0).to(device)} for i in range(len(img_paths))]

    save_base = os.path.join(args.save_path, f"epoch_{args.epoch}")
    os.makedirs(save_base, exist_ok=True)

    with torch.no_grad():
        if args.do_render:
            print("Running 2D semantic rendering...")
            if hasattr(model, "inference_long_video"):
                model.inference_long_video(views, img_height, img_width, save_dir=save_base, max_num=120)
            else:
                print("Fallback: model.inference_long_video not available")

            if args.make_video:
                for level in ["small", "middle", "large"]:
                    lang_dir = os.path.join(save_base, "lang", "renders", level)
                    if os.path.exists(lang_dir):
                        images_to_video_from_folder(lang_dir, os.path.join(lang_dir, "video_lang.mp4"))
                rgb_dir = os.path.join(save_base, "rgb", "renders")
                if os.path.exists(rgb_dir):
                    images_to_video_from_folder(rgb_dir, os.path.join(rgb_dir, "video_rgb.mp4"))

        if args.do_4d:
            print("Running 3D point cloud inference...")
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            with torch.amp.autocast("cuda", dtype=dtype):
                model.save_point_cloud(views, img_height, img_width, save_dir=save_base)

    print("Done. Results saved in:", save_base)


if __name__ == "__main__":
    main()
