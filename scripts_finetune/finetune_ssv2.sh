# Finetuning SSV2 Pretrained Checkpoint
#DATA_PATH='/ssdstore/fmthoker/20bn-something-something-v2/something-something-v2-annotations/'
## Set the path to save checkpoints
#MODEL_PATH='/ivi/zfs/s0/original_homes/fthoker/sigma_final_models/pretrain/ssv2/ssv2_vit_b_sigma_with_dino.pth'
#OUTPUT_DIR='expirements_finetune/finetune_ssv2_pretrained_with_ssv2_vit_b_sigma_with_dino/eval_lr_1e-3_epoch_40_8gpus_no_update_freq/log.txt'
#
#OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
#    --master_port 12320  run_class_finetuning.py \
#    --model vit_base_patch16_224 \
#    --data_set SSV2 \
#    --nb_classes 174 \
#    --data_path ${DATA_PATH} \
#    --finetune ${MODEL_PATH} \
#    --log_dir ${OUTPUT_DIR} \
#    --output_dir ${OUTPUT_DIR} \
#    --batch_size 4 \
#    --num_sample 1 \
#    --input_size 224 \
#    --short_side_size 224 \
#    --save_ckpt_freq 100 \
#    --num_frames 16 \
#    --opt adamw \
#    --lr 1e-3 \
#    --warmup_lr 1e-6 \
#    --opt_betas 0.9 0.999 \
#    --weight_decay 0.05 \
#    --epochs 40 \
#    --dist_eval \
#    --test_num_segment 2 \
#    --test_num_crop 3 \


# Finetuning Kinetics-400 Pretrained Checkpoint
DATA_PATH='/ssdstore/fmthoker/20bn-something-something-v2/something-something-v2-annotations/'
# Set the path to save checkpoints
MODEL_PATH='/ivi/zfs/s0/original_homes/fthoker/sigma_final_models/pretrain/kinetics-400/kinetics_400_vit_b_sigma_with_dino.pth'
OUTPUT_DIR='expirements_finetune/finetune_ssv2_pretrained_with_kinetics_400_vit_b_sigma_with_dino/eval_lr_1e-3_epoch_40_8gpus_no_update_freq/log.txt'

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port 12320  run_class_finetuning.py \
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
    --save_ckpt_freq 100 \
    --num_frames 16 \
    --opt adamw \
    --lr 1e-3 \
    --warmup_lr 1e-6 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 40 \
    --dist_eval \
    --test_num_segment 2 \
    --test_num_crop 3 \
