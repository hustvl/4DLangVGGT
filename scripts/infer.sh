class_name=americano
dataset_name=hypernerf

python inference.py \
  --stream_ckpt ckpt/streamvggt/checkpoints.pth \
  --lang_ckpt ckpt/4dlangvggt/checkpoint_epoch_100.pth \
  --image_dir data/hypernerf/${class_name}/rgb/2x \
  --save_path eval/eval_results/${dataset_name}_img/${class_name} \
  --name video \
  --make_video \
  --lang_dim 3 \
  --epoch 100 \
  --do_render \
  --do_4d