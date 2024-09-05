
DATA_PATH='/var/scratch/ssalehid/kinetics_fida/VideoData/'

DATA_PATH_CSV='/var/scratch/ssalehid/kinetics_fida/labels/train.csv'

OUTPUT_DIR='/var/scratch/ssalehid/runs/Mini-Kinetics-MAE_baseline_with_resize/'

CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=3 \
        --master_port 12320 run_mae_pretraining_baseline.py \
        --data_path ${DATA_PATH} \
        --data_path_csv ${DATA_PATH_CSV} \
        --mask_type tube \
        --target_type pixel \
        --run_name Mini-Kinetics-MAE_baseline_with_resize \
        --mask_ratio 0.9 \
        --model pretrain_videomae_small_patch16_224 \
        --decoder_depth 4 \
        --batch_size 64 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 20 \
        --epochs 1601 \
        --output_dir ${OUTPUT_DIR}
