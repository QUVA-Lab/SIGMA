DATA_PATH='/nvmestore/mdorkenw/20bn-something-something-v2/something-something-v2-annotations/'
MODEL_PATH='/nvmestore/ssalehi/video_mae/mgmae_ckpts/expirements_pretrain/vit_b_ssv2_mgmae_800e_skeps=0.03/checkpoint-799.pth'
OUTPUT_DIR='/nvmestore/ssalehi/expirements_finetune/vit-base-ckpts/SIGMA_SSV2_DINO_MGMAE_SWAV_3K_E03_799_fida_1/eval_lr_5e-4_epoch_40_4gpus_update_freq_8'

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port 12322  run_class_finetuning.py \
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
    --lr 1e-3 \
    --warmup_lr 1e-6 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 40 \
    --dist_eval \
    --test_num_segment 2 \
    --test_num_crop 3 \