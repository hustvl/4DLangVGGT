import os
import sys
import glob
import argparse
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from streamvggt.models.streamvggt import StreamVGGT


def parse_args():
    parser = argparse.ArgumentParser(description="Run StreamVGGT inference on image sequences.")
    parser.add_argument("--categories", type=str, nargs="+", required=True,
                        help="List of categories to process (e.g. --categories keyboard chickchicken).")
    parser.add_argument("--img_root", type=str, required=True,
                        help="Root directory containing image data.")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to StreamVGGT checkpoint file.")
    parser.add_argument("--save_suffix", type=str, default="streamvggt_token",
                        help="Subfolder name for saving results. Default: streamvggt_token")
    parser.add_argument("--max_num", type=int, default=128,
                        help="Max number of frames to load per sequence. Default: 128")
    parser.add_argument("--mode", type=str, default="original",
                        choices=["original", "other"],
                        help="Inference mode. Default: original")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on (e.g. 'cuda' or 'cpu'). Default: cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading checkpoint from {args.ckpt} ...")
    streamvggt = StreamVGGT()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    streamvggt.load_state_dict(ckpt, strict=True)
    streamvggt.eval()
    streamvggt.aggregator.to(args.device)

    for cat in args.categories:
        img_dir = os.path.join(args.img_root, cat, "rgb", "2x")
        image_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        if len(image_files) == 0:
            print(f"[Warning] No images found in {img_dir}, skipping {cat}")
            continue

        keys = [os.path.splitext(os.path.basename(p))[0] for p in image_files]
        save_dir = os.path.join(args.img_root, cat, "streamvggt_token")
        os.makedirs(save_dir, exist_ok=True)

        print(f"Processing category: {cat}, images: {len(image_files)} -> saving to {save_dir}")

        with torch.no_grad():
            streamvggt.inference_long_video(
                image_files,
                save_dir=save_dir,
                keys=keys,
                mode=args.mode,
                max_num=args.max_num
            )
            torch.cuda.empty_cache()

    print("All batches processed successfully!")


if __name__ == "__main__":
    main()
