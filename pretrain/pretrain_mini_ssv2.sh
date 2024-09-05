DATA_PATH="/ssdstore/mdorkenw/20bn-something-something-v2/something-something-v2-videos_avi/"
DATA_PATH_CSV='/ssdstore/mdorkenw/20bn-something-something-v2/something-something-v2-annotations/train.csv'

OUTPUT_DIR='/hddstore/ssalehi/videomae/runs/video_mae/VITB_MGMAE_SSV2_swav/'

# DATA_PATH="/ssdstore/videomae/20bn-something-something-v2/something-something-v2-videos_avi/"
# DATA_PATH_CSV='/ssdstore/videomae/20bn-something-something-v2/something-something-v2-annotations/train_mini.csv'

# OUTPUT_DIR='/ssdstore/videomae/runs/video_mae/MAE_tube_mask_90_infonce_rate4_temp005/'


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=3 \
        --master_port 12312 run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --data_path_csv ${DATA_PATH_CSV} \
        --mask_type tube \
        --target_type  dino_v1 \
        --loss_func SWAV \
        --run_name VITS_MGMAE_SSV2_MINI_SWAV \
        --mask_ratio 0.9 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 32 \
        --num_frames 16 \
        --sampling_rate 2 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --distillation_teacher dino_b \
        --save_ckpt_freq 20 \
        --epochs 800 \
        --output_dir ${OUTPUT_DIR} \
        --memory_size 0 \
        --normlize_target True \
        --num_prototypes 3000 \
        --sinkhorn_eps 0.1 \
        --sinkhorn_iterations 10 \
        --augmentation multi_scale_crop \