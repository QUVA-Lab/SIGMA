DATA_PATH="/local/mdorkenw//20bn-something-something-v2/something-something-v2-videos_avi/"
DATA_PATH_CSV='/local/mdorkenw/20bn-something-something-v2/something-something-v2-annotations/train.csv'
DATA_PATH_FEAT='/ssd/fmthoker/TimeT_Features_SSv2_full/something-something-v2-videos_avi/'

OUTPUT_DIR='/var/scratch/mdorkenw/runs/video_mae/MAE_tube_mask_90_infonce_singlevideo_memory4096_sr4_fullssv2/'

# DATA_PATH="/ssdstore/videomae/20bn-something-something-v2/something-something-v2-videos_avi/"
# DATA_PATH_CSV='/ssdstore/videomae/20bn-something-something-v2/something-something-v2-annotations/train_mini.csv'
# DATA_PATH_FEAT='/ssd/fmthoker/TimeT_Features_SSv2_full/something-something-v2-videos_avi/'

# OUTPUT_DIR='/ssdstore/videomae/runs/video_mae/MAE_tube_mask_90_infonce_rate4_temp005/'

#CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 \
CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=3 \
        --master_port 12322 run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --data_path_csv ${DATA_PATH_CSV} \
        --mask_type tube \
        --target_type pixel \
        --loss_func SWAV \
        --run_name MAE_tube_pixel_swav_prot1024_resize \
        --mask_ratio 0.9 \
        --model pretrain_videomae_small_patch16_224 \
        --decoder_depth 4 \
        --batch_size 64 \
        --num_frames 16 \
        --sampling_rate 2 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 20 \
        --epochs 2401 \
        --output_dir ${OUTPUT_DIR} \
        --memory_size 0 \
        --num_prototypes 1024 \
        --augmentation resize
