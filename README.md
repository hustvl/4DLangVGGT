# 4DLangVGGT: 4D Language Visual Geometry Grounded Transformer
Official implementation of “4D LangVGGT: 4D Language-Visual Geometry Grounded Transformer”

## Enviroment

4D LangVGGT uses the following software versions:
- Python 3.10
- CUDA 12.4

On default, run the following commands to install the relative packages

```bash
# if you lose some pkgs
# apt-get update && apt-get install libgl1 ffmpeg libsm6 libxext6 -y 

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

## Training
First, download pretrain [StreamVGGT](https://github.com/wzzheng/StreamVGGT) ckpt
```bash
huggingface-cli download --resume-download lch01/StreamVGGT --local-dir lch01/StreamVGGT

```

Then, pretrain your model after follow [4DLangSplat](https://github.com/zrporz/4DLangSplat) to preprocess your dataset

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