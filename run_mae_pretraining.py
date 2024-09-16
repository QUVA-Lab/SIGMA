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
from engine_for_pretraining import train_one_epoch
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import wandb
from timm.models.vision_transformer import vit_small_patch16_224, vit_base_patch16_224, vit_large_patch16_224
import timm
from modeling_pretrain import FeatureExtractor


def get_args():
    parser = argparse.ArgumentParser('SIGMA pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)

    # Model parameters
    parser.add_argument('--model', default='pretrain_videomae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--decoder_depth', default=4, type=int,
                        help='depth of decoder')
    
    parser.add_argument('--mask_type', default='tube', choices=['random', 'tube', 'parts', 'tube_fgbg'],
                        type=str, help='masked strategy of video tokens/patches')

    parser.add_argument('--target_type', default='mlp', choices=['pixel', 'mlp', 'dino_v1', 'dino_v2'],
                            type=str, help='define target type for loss')
   
    parser.add_argument('--loss_func', default='L2', choices=['L2', 'SWAV'],
                            type=str, help='define target type for loss')
    parser.add_argument('--augmentation', default='resize', choices=['resize', 'multi_scale_crop'],
                            type=str, help='define augmentation type for training')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')

    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size for backbone')

    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
                        
    parser.add_argument('--normlize_target', default=True, type=bool,
                        help='normalized the target patch pixels')
    
    parser.add_argument('--gray_scale_prob', default=0.0, type=float, help='probability of gray scale')

    parser.add_argument('--kwindow', default=1, type=int, help='memory size for DPC')

    parser.add_argument('--memory_size', default=0, type=int, help='memory size for DPC')
    parser.add_argument('--distillation_teacher', default="dino_s", type=str, choices=['dino_s', 'dino_b', 'dino_l'], help='distillation teacher model')
    parser.add_argument('--num_prototypes', default=1024, type=int, help='number of prototypes for swav')
    parser.add_argument('--sinkhorn_eps', default=0.05, type=float, help='number of prototypes for swav')
    parser.add_argument('--sinkhorn_iterations', default=10, type=int, help='number of prototypes for swav')

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
    parser.add_argument('--data_path', default='/ssdstore/mdorkenw/20bn-something-something-v2/', type=str,
                        help='dataset video path')
    parser.add_argument('--mlp_preloading', default='', type=str,
                        help='preloading mlp features for swav')
    parser.add_argument('--data_path_csv', default='/ssdstore/mdorkenw/20bn-something-something-v2/something-something-v2-annotations/train_mini.csv', type=str,
                        help='dataset csv file path')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_frames', type=int, default= 16)
    parser.add_argument('--sampling_rate', type=int, default= 2)
    parser.add_argument('--output_dir', default='',
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
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()


def get_model(args):
    
    if args.loss_func != 'SWAV' and (args.target_type == 'mlp'):
        print('Not implemented as it doesnt make sense')
        exit()
    
    print(f"Creating model: {args.model}")
    #compute output dim for decoder
    if args.target_type=='pixel':
        dec_dim = 1536
    elif 'dino' in args.target_type:
        if args.distillation_teacher == 'dino_s':
            dec_dim = 384
        elif args.distillation_teacher == 'dino_b':
            dec_dim = 768
    else:
        dec_dim = 256
    
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        decoder_depth=args.decoder_depth,
        use_checkpoint=args.use_checkpoint,
        decoder_num_classes=dec_dim,
        target_type=args.target_type,
        mask_ratio=args.mask_ratio,
        mask_type=args.mask_type,
        loss_func=args.loss_func,
        memory_size=args.memory_size,
        num_prototypes=args.num_prototypes,
        world_size=args.world_size,
        sinkhorn_iterations=args.sinkhorn_iterations,
        eps=args.sinkhorn_eps,
        kwindow=args.kwindow,
    )
    if args.target_type == 'dino_v1':
        if args.distillation_teacher == 'dino_s':
            pretraining = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
            feature_extraction_model = vit_small_patch16_224(pretrained=False)
        elif args.distillation_teacher == 'dino_b':
            pretraining = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
            feature_extraction_model = vit_base_patch16_224(pretrained=False)
        elif args.distillation_teacher == 'dino_l':
            pretraining = torch.hub.load('facebookresearch/dino:main', 'dino_vitl16')
            feature_extraction_model = vit_large_patch16_224(pretrained=False)
        msg = feature_extraction_model.load_state_dict(pretraining.state_dict(), strict=False)
        feature_extraction_model = FeatureExtractor(feature_extraction_model, args.input_size, 16)
        print(msg)
        feature_extraction_model.eval()
    elif args.target_type == 'dino_v2':
        if args.distillation_teacher == 'dino_s':
            feature_extraction_model = timm.create_model(
            'vit_small_patch14_reg4_dinov2.lvd142m',
            img_size=(args.input_size, args.input_size),
            pretrained=True,
            num_classes=0)
        elif args.distillation_teacher == 'dino_b':
            feature_extraction_model = timm.create_model(
            'vit_base_patch14_reg4_dinov2.lvd142m',
            img_size=(args.input_size, args.input_size),
            pretrained=True,
            num_classes=0)
        elif args.distillation_teacher == 'dino_l':
            feature_extraction_model = timm.create_model(
            'vit_large_patch14_reg4_dinov2.lvd142m',
            img_size=(args.input_size, args.input_size),
            pretrained=True,
            num_classes=0)
        feature_extraction_model = FeatureExtractor(feature_extraction_model, args.input_size, 14)
        # print(msg)
        feature_extraction_model.eval()
    else:
        feature_extraction_model = None
    return model, feature_extraction_model


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    
    #save args to in the output dir
    file_name = os.path.join(args.output_dir, "args.txt")
    with open(file_name, 'w') as file:
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")

    print(f"File '{file_name}' has been saved with the current configuration.")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    model, feature_extraction_model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size
    
    if len(args.mlp_preloading) > 0:
        print("Loading preloaded MLP features from %s" % args.mlp_preloading)
        utils.load_mlp_weight(args.mlp_preloading, model)

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
        log_writer = wandb.init(name=args.run_name + time_, mode="online", project=project, dir=args.output_dir)
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

    model.to(device)
    if feature_extraction_model is not None:
        feature_extraction_model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params: {} M'.format(n_parameters / 1e6))

    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)        
    torch.cuda.empty_cache()
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.log({"epoch": epoch * num_training_steps_per_epoch})
        train_stats = train_one_epoch(
            model, feature_extraction_model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            patch_size=patch_size[0],
            normlize_target=args.normlize_target,
            target_type=args.target_type,
            mask_type=args.mask_type,
            output_dir=args.output_dir,
            loss_func=args.loss_func)
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
