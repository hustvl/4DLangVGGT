# 4DLangVGGT: 4D Language Visual Geometry Grounded Transformer
Official implementation of “4D LangVGGT: 4D Language-Visual Geometry Grounded Transformer”


## Overview
4DLangVGGT is a feed-forward framework for language-aware 4D scene understanding, combining StreamVGGT for dynamic geometry reconstruction with a Semantic Bridging Decoder (SBD) that aligns geometry tokens with language semantics. Unlike Gaussian Splatting methods that require per-scene optimization, our feed-forward design can be trained across multiple scenes and directly applied at inference, achieving scalable, efficient, and open-vocabulary 4D semantic fields with state-of-the-art performance on HyperNeRF and Neu3D benchmarks.

## Installation

4D LangVGGT uses the following software versions:
- Python 3.10
- CUDA 12.4

First, please clone 4DLangVGGT according to the command below.
```bash
git clone https://github.com/hustvl/4DLangVGGT.git
cd 4DLangVGGT
```

Then create a conda environment using the following command:

```bash
# if you lose some pkgs
# apt-get update && apt-get install libgl1 ffmpeg libsm6 libxext6 -y 

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

## Dataset
4DLangVGGT is trained and evaluated on the [HyperNeRF](https://github.com/google/hypernerf) and [Neu3D](https://github.com/facebookresearch/Neural_3D_Video) datasets. Please download the datasets and put them in the folder `./data`. For data processing, please refer to [4DLangSplat](https://github.com/zrporz/4DLangSplat) to generate segmentation map and extract CLIP and Video features.


## QuickStart
### Download Checkpoints
Please download the checkpoint of StreamVGGT from [here](https://github.com/wzzheng/StreamVGGT) and put the checkpoint folder under `./ckpt/streamvggt`

The checkpoint of 4DLangVGGT is availavle at [Hugging Face](https://huggingface.co/YajingB/4DLangVGGT) and put the checkpoint folder under `./ckpt/4dlangvggt`

### Inference
Run the following command to generate the demo:
```bash
bash scripts/infer.sh
```
The results will be saved under `./eval/eval_results`.

## Folder Structure
The overall folder structure should be organized as follows：
```text
4DLangVGGT
|-- ckpt
|   |-- streamvggt
|   |   |-- checkpoints.pth
|   |   |-- model.safetensors
|   |-- 4dlangvggt
|   |   |-- 
|-- data
|   |-- hypernerf
|   |   |-- americano
|   |   |   |-- annotations
|   |   |   |   |-- train
|   |   |   |   |-- README
|   |   |   |   |-- video_annotations.json
|   |   |   |-- camera
|   |   |   |-- rgb
|   |   |   |   |-- 1x
|   |   |   |   |   |-- 000001.png
|   |   |   |   ...
|   |   |   |   |-- 2x
|   |   |   |   |   |-- 000001.png
|   |   |   |-- streamvggt_token
|   |   |   |   |   |-- 000001.npy
|   |   |   ...
|   |   |   |-- dataset.json
|   |   |   |-- metadata.json
|   |   |   |-- points.npy
|   |   |   |-- scene.json
|   |   |   |-- points3D_downsample2.ply
|   |   |-- chickchicken
|   |   ...
|   |-- neu3d
|   |   |-- coffee_martini
|   |   |   |-- annotations
|   |   |   |   |-- train
|   |   ...
```

## Training
### Step1: Generate Geometry Tokens
To reduce the amount of memory required during training, we first preprocess the video using StreamVGGT, extract the geometry tokens, and save them in the folder `./data/<dataset>/<class>/streamvggt_token`. Take the americano class from the HyperNeRF dataset as an example, you need to ensure the extracted geometry tokens are in the folder `./data/hypernerf/americano/streamvggt_token`.
```bash
python preprocess/generate_vggttoken.py \
    --categories americano \
    --img_root data/hypernerf \
    --ckpt ckpt/streamvggt/checkpoints.pth \
    --max_num 128 \
    --device cuda
```

### Step2: Train 4DLangVGGT
We provide the following commands for training.
```bash
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 train.py --batch_size 8 \
                --data_root YOUR_DATA_ROOT --streamvggt_ckpt_path YOUR_STREAMVGGT_CKPT  \
                --num_workers 0 --output_dir unify_hyper_clip --mode gt --cos --wandb --joint_train \
                --feat_root clip_features-all_dim3 \
```

## Acknowledgements
Our code is based on the following brilliant repositories:

- [StreamVGGT](https://github.com/wzzheng/StreamVGGT)

- [VGGT](https://github.com/facebookresearch/vggt)

- [4DLangSplat](https://github.com/zrporz/4DLangSplat)

Many thanks to these authors!

## License

Released under the [MIT](LICENSE) License.