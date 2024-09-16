DATA_PATH="/ssdstore/fmthoker/20bn-something-something-v2/something-something-v2-videos_avi/"
DATA_PATH_CSV='/ssdstore/fmthoker/20bn-something-something-v2/something-something-v2-annotations/train.csv'

OUTPUT_DIR='/ivi/zfs/s0/original_homes/fthoker/runs/VITB_SSV2_Test/'

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
        --master_port 12312 run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --data_path_csv ${DATA_PATH_CSV} \
        --mask_type tube \
        --target_type dino_v1 \
        --distillation_teacher dino_b \
        --loss_func SWAV \
        --run_name SN_FULL_SSV2_DINOv1_SWAV_3K_4nodes \
        --mask_ratio 0.9 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 16 \
        --num_frames 16 \
        --sampling_rate 2 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 5 \
        --epochs 801 \
        --output_dir ${OUTPUT_DIR} \
        --normlize_target True \
        --num_prototypes 3000 \
        --sinkhorn_eps 0.03 \
        --sinkhorn_iterations 10 \
        --augmentation multi_scale_crop \
                                            
