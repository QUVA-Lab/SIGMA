# OUTPUT_DIR='/var/scratch/ssalehid/expirements_finetune/MAE_tube_mask_90_pixel_reconstruction_TimeT_with_fg_bg_loss_new_masks_alpha_5/eval_lr_1e-3_epoch_40_layer_dacay_0.7'
# path to SSV2 annotation file (train.csv/val.csv/test.csv)
# DATA_PATH='/var/scratch/ssalehid/20bn//20bn-something-something-v2/something-something-v2-annotations/'
DATA_PATH='/ssdstore/videomae/20bn-something-something-v2/something-something-v2-annotations/'

# path to pretrain model
#MODEL_PATH='expirements_pretrain/mini_ssv2_videomae_pretrain_small_patch16_224_frame_16x2_tube_mask_ratio_0.9_e2400/checkpoint-599.pth'
#MODEL_PATH='/home/fthoker/runs/MAE_baseline_with_resize/checkpoint-399.pth'
# MODEL_PATH='/var/scratch/ssalehid/runs/Mini-Kinetics-MAE_baseline_with_resize/checkpoint-399.pth'
MODEL_PATH='/ssdstore/videomae/runs/video_mae/MAE_tube_mask_90_infonce_paired_rate2_DINO_clustering_5/checkpoint-399.pth'
OUTPUT_DIR='/ssdstore/videomae/runs/video_mae/MAE_tube_mask_90_infonce_paired_rate2_DINO_clustering_5_finetune/'


# batch_size can be adjusted according to number of GPUs
# this script is for 32 GPUs (4 nodes x 8 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=3 \
    --master_port 12321  run_class_finetuning.py \
    --model vit_small_patch16_224 \
    --data_set SSV2 \
    --nb_classes 174 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 4 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --opt adamw \
    --lr 1e-3 \
    --layer_decay 0.7 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 40 \
    --test_num_segment 2 \
    --test_num_crop 3 \
    --dist_eval \
    --update_freq 8 \
