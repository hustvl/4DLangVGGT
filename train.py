import os
import argparse
from re import T
import time
import wandb
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from head.langhead import LanguageHead_Multi
from models.langvggt import l1_loss, cos_loss, l2_loss
from utils.demo_dataloader import collate_fn_img, MultiIMGDataset
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

from streamvggt.utils.load_fn import load_and_preprocess_images


def compute_loss(lang_feat, pred_image, clip_feats, seg_maps, images, args):
    outputs_masked = lang_feat * seg_maps
    clip_feats_masked = clip_feats * seg_maps
    
    loss = args.lambda1 * l1_loss(outputs_masked, clip_feats_masked)
    loss_value = loss.item()
    
    if args.cos:
        cos_loss_val = args.lambda2 * cos_loss(outputs_masked, clip_feats_masked)
        loss += cos_loss_val
        loss_value += cos_loss_val.item()

    if args.joint_train:
        rec_loss_l1 = args.lambda_img * l1_loss(pred_image, images)
        rec_loss_l2 = (1 - args.lambda_img) * l2_loss(pred_image, images)
        loss += rec_loss_l1
        loss += rec_loss_l2
        loss_value += rec_loss_l1.item()
        loss_value += rec_loss_l2.item()
    
    return loss, loss_value


def parse_args():
    parser = argparse.ArgumentParser(description='4DLangVGGT DDP Training')
    parser.add_argument('--data_root', type=str, default="",
                      help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='batch size per process (default: 8)')
    parser.add_argument('--epochs', type=int, default=100,
                      help='number of total epochs to run')
    parser.add_argument('--lr', type=float, default=4e-5,
                      help='initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                      help='minimum learning rate for cosine schedule')
    parser.add_argument('--warmup_epochs', type=int, default=20,
                      help='warmup_epochs')
    parser.add_argument('--lr_schedule', type=str, default="constant",
                      help='lr_schedule: constant or cosine')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                      help='weight decay (default: 1e-4)')
    parser.add_argument('--print_freq', type=int, default=10,
                      help='print frequency (default: 10)')
    parser.add_argument('--save_freq', type=int, default=5,
                      help='save checkpoint frequency (default: 5)')
    parser.add_argument('--output_dir', type=str, default='./outputs_video',
                      help='path to save outputs')
    parser.add_argument('--resume', type=str, default='',
                      help='path to resume from checkpoint')
    parser.add_argument("--lambda_img", type=float, default=0.5,
                      help='weight for L2 loss')
    parser.add_argument("--lambda1", type=float, default=0.2,
                      help='weight for L1 loss')
    parser.add_argument("--lambda2", type=float, default=0.01,
                      help='weight for cosine loss')
    parser.add_argument('--streamvggt_ckpt_path', type=str, default="",
                      help='path to dataset')
    parser.add_argument('--wandb', action='store_true',
                      help='Pin memory for DataLoader')
    
    # DDP args
    parser.add_argument('--dist_url', default='env://',
                      help='url used to set up distributed training')
    parser.add_argument('--dist_on_itp', action='store_true',
                      help='whether to use ITP for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                      help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                      help='GPU id to use. None means using all available GPUs')
    
    # dataset args
    parser.add_argument('--num_workers', type=int, default=4,
                      help='number of data loading workers (default: 4)')
    parser.add_argument('--stride', type=int, default=1,
                      help='number of stride (default: 1)')
    parser.add_argument('--pin_mem', action='store_true',
                      help='Pin memory for DataLoader')
    
    # training setting
    parser.add_argument('--seed', type=int, default=42,
                      help='random seed (default: 42)')
    parser.add_argument('--cos', action='store_true',
                      help='whether to use cosine loss')
    parser.add_argument('--joint_train', action='store_true',
                      help='whether to joint training')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                      help='gradient clipping value (default: 1.0)')
    parser.add_argument('--mode', type=str, default="crop",
                      help='mode: crop, pad or original (default: crop)')
    parser.add_argument('--overfit', action='store_true',
                      help='training via overfit or all dataset')
    parser.add_argument('--cat', type=str, default="americano",
                      help='if overfit, must give category of dataset')
    parser.add_argument('--feat_root', type=str, default="clip_features-language_features_dim3",
                      help='gt feat root')
    return parser.parse_args()


def main(args):
    misc.init_distributed_mode(args)
    rank = misc.get_rank()
    world_size = misc.get_world_size()

    if rank == 0 and args.wandb:  # Ensure WandB is only initialized by the main process
        wandb.init(
            project="LangVGGT_overfit",
            name=f"overfit_{args.cat}",  
            config=vars(args),
            dir=args.output_dir
        )

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if misc.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir}")
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Rank {rank} using device: {device}")

    model = LanguageHead_Multi(2048, patch_size=14, features=128, lang_dim=3, hidden_dim=32)
    model = model.to(device)

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        # if 'scaler' in checkpoint:
        #     loss_scaler.load_state_dict(checkpoint['scaler'])
        print(f"Rank {rank} loaded checkpoint from epoch {start_epoch}")
    
    model = DDP(model, device_ids=[args.gpu], find_unused_parameters=False)
    model_without_ddp = model.module
    
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    if args.resume and os.path.isfile(args.resume):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
    
    if args.overfit:
        overfit = True
    else:
        overfit = False

    dataset = MultiIMGDataset(
        img_root=args.data_root, 
        overfit=overfit,
        cat=args.cat,
        feat_root=args.feat_root
    )
    
    sampler_train = torch.utils.data.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    print(f"Rank {rank} Sampler_train = {str(sampler_train)}")

    data_loader_train = torch.utils.data.DataLoader(
        dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        # collate_fn=collate_fn
        collate_fn=collate_fn_img,
    )

    total_steps = len(data_loader_train)
    print(f"Rank {rank} starting training for {args.epochs} epochs, {total_steps} steps per epoch")
    
    for epoch in range(start_epoch, args.epochs):
        sampler_train.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        
        current_lr = misc.adjust_learning_rate(optimizer, epoch, args)
        
        for step, batch_data in enumerate(data_loader_train):
            # print(f'current epoch: {epoch + 1}, current step: {step + 1}') # log info
            interation_step = epoch*len(data_loader_train) + step
            step_start_time = time.time()
            optimizer.zero_grad()
            
            batch_size = len(batch_data['images_path'])
            total_batch_loss = 0.0

            images_path = batch_data['images_path']
            clip_feats = batch_data['clip_feats'].to(device)
            seg_maps = batch_data['seg_maps'].to(device)
            ### add 
            aggregated_tokens_list = [aggregated_tokens.to(device) for aggregated_tokens in batch_data['aggregated_tokens_list']]
            patch_start_idx = batch_data['patch_start_idx'][0]
            
            images, width, height = load_and_preprocess_images(images_path, mode=args.mode)
            
            frames = [{"img": images[j].unsqueeze(0)} for j in range(len(images_path))]

            images = torch.stack(
                [view["img"] for view in frames], dim=0
            ).permute(1, 0, 2, 3, 4).to(device)  # B S C H W

            with torch.amp.autocast('cuda'):
                    lang_feat, pred_image = model(
                        aggregated_tokens_list, 
                        images, 
                        patch_start_idx, 
                        img_height=height, 
                        img_width=width
                    )
            
            loss, loss_value = compute_loss(lang_feat, pred_image, clip_feats, seg_maps, images, args)
            total_batch_loss += loss_value
            
            loss_scaler(
                loss, 
                optimizer, 
                clip_grad=args.grad_clip, 
                parameters=model.parameters(), 
                update_grad=True
            )

            if rank == 0 and args.wandb:
                wandb.log({
                    "steps": interation_step, # step
                    "loss": loss_value/batch_size,
                    "learning_rate": optimizer.param_groups[0]["lr"]
                })
            optimizer.zero_grad()
            
            batch_avg_loss = total_batch_loss / batch_size
            running_loss += batch_avg_loss
            
            if step % args.print_freq == 0:
                step_time = time.time() - step_start_time
                if misc.is_main_process():
                    print(
                        f"Epoch [{epoch+1}/{args.epochs}], "
                        f"Step [{step+1}/{total_steps}], "
                        f"LR: {current_lr:.6f}, "
                        f"Loss: {batch_avg_loss:.6f}, "
                        f"Time: {step_time:.2f}s"
                    )
        
        epoch_avg_loss = running_loss / total_steps
        epoch_time = time.time() - epoch_start_time
        
        if misc.is_main_process():
            print(
                f"Epoch [{epoch+1}/{args.epochs}] completed. "
                f"Average Loss: {epoch_avg_loss:.6f}, "
                f"Time: {epoch_time:.2f}s, "
                f"LR: {current_lr:.6f}"
            )
        
        if (epoch + 1) % args.save_freq == 0 and misc.is_main_process():
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            misc.save_on_master({
                'epoch': epoch + 1,
                'model_state_dict': model_without_ddp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': loss_scaler.state_dict(),
                'loss': epoch_avg_loss,
                'args': args,  
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        torch.cuda.empty_cache()
    
    print(f"Rank {rank} training completed!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
