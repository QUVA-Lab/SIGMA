import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path
from timm.models import create_model
from optim_factory import create_optimizer
from datasets import build_pretraining_dataset
# from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import wandb
import modeling_pretrain
import torch.nn.functional as F
from torchvision.utils import draw_segmentation_masks
from PIL import Image


def denormalize_video(video):
    """
    video: [1, nf, c, h, w]
    """
    denormalized_video = video.cpu().detach() * torch.tensor([0.225, 0.225, 0.225]).view(1, 1, 3, 1, 1) + torch.tensor([0.45, 0.45, 0.45]).view(1, 1, 3, 1, 1)
    denormalized_video = (denormalized_video * 255).type(torch.uint8)
    denormalized_video = denormalized_video.squeeze(0)
    return denormalized_video


def convert_list_to_video(frames_list, name, speed, directory="", wdb_log=False):
    frames_list = [Image.fromarray(frame) for frame in frames_list]
    frames_list[0].save(f"{directory}{name}.gif", save_all=True, append_images=frames_list[1:], duration=speed, loop=0)
    if wdb_log:
        wandb.log({name: wandb.Video(f"{directory}{name}.gif", fps=4, format="gif")})

def overlay_video_cmap(cluster_maps, denormalized_video):
    """
    cluster_maps: [nf, h, w]
    denormalized_video: [nf, c, h, w]
    """
        ## convert cluster_maps to [num_maps, h, w]
    masks = []
    cluster_ids = torch.unique(cluster_maps)
    for cluster_map in cluster_maps:
        mask = torch.zeros((cluster_ids.shape[0], cluster_map.shape[0], cluster_map.shape[1])) 
        mask = mask.type(torch.bool)
        for i, cluster_id in enumerate(cluster_ids):
                ## make a boolean mask for each cluster
                ## make a boolean mask for each cluster if cluster_map == cluster_id
            boolean_mask = (cluster_map == cluster_id)
            mask[i, :, :] = boolean_mask
        masks.append(mask)
    cluster_maps = torch.stack(masks)
            
    overlayed = [
                draw_segmentation_masks(img, masks=mask, alpha=0.5)
                for img, mask in zip(denormalized_video, cluster_maps)
            ]
    overlayed = torch.stack(overlayed)
    return cluster_maps,overlayed



def get_args():
    parser = argparse.ArgumentParser('VideoMAE pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)

    # Model parameters
    parser.add_argument('--model', default='pretrain_videomae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--decoder_depth', default=4, type=int,
                        help='depth of decoder')
    
    parser.add_argument('--mask_type', default='tube', choices=['random', 'tube', 'parts'],
                        type=str, help='masked strategy of video tokens/patches')

    parser.add_argument('--target_type', default='pixel', choices=['pixel', 'features'],
                            type=str, help='define target type for loss')

    parser.add_argument('--pos_emd_type_feat', default='average',
                            type=str, help='average or just use embed without xy information')
    
    parser.add_argument('--n_parts', default=20, type=int,
                        help='number of objects parts in features')
    
    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='ratio of the visual tokens/patches need be masked')

    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size for backbone')

    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
                        
    parser.add_argument('--normlize_target', default=True, type=bool,
                        help='normalized the target patch pixels')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.set_defaults(use_checkpoint=False)

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.0, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Dataset parameters
    parser.add_argument('--data_path', default='/ssdstore/mdorkenw/20bn-something-something-v2/something-something-v2-videos_avi/', type=str,
                        help='dataset video path')
    parser.add_argument('--data_path_csv', default='/ssdstore/mdorkenw/20bn-something-something-v2/something-something-v2-annotations/train_mini.csv', type=str,
                        help='dataset csv file path')
    parser.add_argument('--data_path_feat', default='/ssdstore/mdorkenw/TimeT_Features_SSv2_full/something-something-v2-videos_avi/', type=str,
                        help='dataset features path')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_frames', type=int, default= 16)
    parser.add_argument('--sampling_rate', type=int, default= 4)
    parser.add_argument('--output_dir', default='/ssdstore/ssalehi/runs/MAE_pixel_parts_60/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--run_name', default='debug',
                        help='name for wandb to log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    #parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    #compute output dim for decoder
    if args.target_type=='pixel':
        dec_dim = 1536
    elif args.target_type=='features':
        dec_dim = 384
    else:
        raise NotImplementedError
    
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        decoder_depth=args.decoder_depth,
        use_checkpoint=args.use_checkpoint,
        decoder_num_classes=dec_dim,
        n_parts=args.n_parts,
        pos_emd_type_feat=args.pos_emd_type_feat,
        target_type=args.target_type,
        mask_ratio=args.mask_ratio,
        mask_type=args.mask_type
    )
    return model



def main(args):
    utils.init_distributed_mode(args)

    print(args)
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # model = get_model(args)
    # patch_size = model.encoder.patch_embed.patch_size
    # print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // 16, args.input_size // 16)
    # args.patch_size = patch_size

    # get dataset
    dataset_train = build_pretraining_dataset(args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_rank = global_rank

    total_batch_size = args.batch_size * num_tasks
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))
   

    if global_rank == 0:
        time_ = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")    
        project = 'debug' if 'debug' in args.run_name else "Video-MAE"
        log_writer = wandb.init(name=args.run_name + time_, project=project)
    else:
        
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=utils.seed_worker
    )

    # model.to(device)
    # model_without_ddp = model
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    # print('number of params: {} M'.format(n_parameters / 1e6))

    # args.lr = args.lr * total_batch_size / 256
    # args.min_lr = args.min_lr * total_batch_size / 256
    # args.warmup_lr = args.warmup_lr * total_batch_size / 256
    # print("LR = %.8f" % args.lr)
    # print("Batch size = %d" % total_batch_size)
    # print("Number of training steps = %d" % num_training_steps_per_epoch)
    # print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
    #     model_without_ddp = model.module

    # optimizer = create_optimizer(
    #     args, model_without_ddp)
    # loss_scaler = NativeScaler()

    # print("Use step level LR & WD scheduler!")
    # lr_schedule_values = utils.cosine_scheduler(
    #     args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
    #     warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    # )
    # if args.weight_decay_end is None:
    #     args.weight_decay_end = args.weight_decay
    # wd_schedule_values = utils.cosine_scheduler(
    #     args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    # print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    # utils.auto_load_model(
    #     args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    # torch.cuda.empty_cache()
    # print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for step, batch in enumerate(data_loader_train):
        videos, bool_masked_pos, _, _, _  = batch
        bs, c, nf, h, w = videos.shape
        videos = videos.permute(0, 2, 1, 3, 4)
        num_sampled_frames = nf // 2
        bool_masked_pos = bool_masked_pos.reshape(bs, num_sampled_frames, 14, 14)

        ## visualize the clustering
        resized_cluster_map = F.interpolate(bool_masked_pos.float(), size=(h, w), mode="nearest")[0]
        denormalized_video = denormalize_video(videos[0])
        sampled_video_every_other_frame = denormalized_video[::2]
        _, overlayed_images = overlay_video_cmap(resized_cluster_map, sampled_video_every_other_frame)
        convert_list_to_video(overlayed_images.detach().cpu().permute(0, 2, 3, 1).numpy(), f"Temp/overlayed_{step}.mp4", speed=1000/ nf)




if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)