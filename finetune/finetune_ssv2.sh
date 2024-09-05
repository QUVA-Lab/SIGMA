
# Set the path to save checkpoints
OUTPUT_DIR='expirements_finetune/ssv2_k400_videomae_with_tubelets_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0/eval_lr_5e-4_epoch_50'
# path to SSV2 set (train.csv/val.csv/test.csv)
DATA_PATH='/ssdstore/fmthoker/20bn-something-something-v2/'
# path to pretrain model
MODEL_PATH='/home/fthoker/VideoMAE/expirements_pretrain/k400_videomae_with_tubelets_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e800/checkpoint-799.pth'

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 \
    --master_port 12321  run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set SSV2 \
    --nb_classes 174 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 4 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 50 \
    --dist_eval \
    --test_num_segment 2 \
    --test_num_crop 3 \
    --update_freq 8 \
    #--enable_deepspeed 

