import numpy as np
from datasets import build_pretraining_dataset
import wandb
import torch
import argparse
import utils


def get_args():
    parser = argparse.ArgumentParser('VideoMAE pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.add_argument('--data_path', default='/ssdstore/mdorkenw/20bn-something-something-v2/', type=str)
    parser.add_argument('--data_path_feat', default='/ssdstore/videomae/TimeT_Features_SSv2_full/something-something-v2-videos_avi/', type=str)
    parser.add_argument('--data_path_csv', default='/ssdstore/mdorkenw/20bn-something-something-v2/something-something-v2-annotations/train_mini.csv', type=str)
    parser.add_argument('--mask_type', default='tube', choices=['random', 'tube', 'parts', 'tube_fgbg'],
                        type=str, help='masked strategy of video tokens/patches')
    parser.add_argument('--target_type', default='pixel', choices=['pixel', 'features', 'pixel_mlp'],
                            type=str, help='define target type for loss')
    parser.add_argument('--feature_extraction', default='all', choices=['all', 'average','k_mean_zero_step',],
                            type=str, help='define feature extraction type ')
    parser.add_argument('--num_frames', default=16, type=int)
    parser.add_argument('--sampling_rate', default=4, type=int)
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')

    parser.add_argument('--n_parts', default=20, type=int,
                        help='number of objects parts in features')

    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size for backbone')

    parser.set_defaults(pin_mem=True)

    return parser.parse_args()


def main(args):
    # get dataset
    # get dataset
    patch_size = (16, 16)
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    dataset_train = build_pretraining_dataset(args)
    logger = wandb.init(project="fgvssl_vis")
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=utils.seed_worker
    )


    for step, batch in enumerate(data_loader_train):
        videos, bool_masked_pos, features, feature_mask, pos_mask, fg_bg_mask  = batch
        # frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
        # wandb.log({"video": wandb.Video(frames, fps=4)})
        clip = videos.permute(0, 2, 1, 3, 4)
        input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        ## convert to 0-255
        clip = clip * torch.tensor(input_std).view(1, 1, 3, 1, 1) + torch.tensor(input_mean).view(1, 1, 3, 1, 1) 
        clip = clip * 255
        ## convert to uint8
        clip = clip.type(torch.uint8)
        clip_1 = clip.numpy()
        clip_1 = clip_1.astype(np.uint8)
        # clip_1 = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
        wandb.log({"video": wandb.Video(clip_1[:5], fps=4)})



if __name__ == '__main__':
    opts = get_args()
    main(opts)