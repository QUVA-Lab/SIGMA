DATA_PATH="Your_Path/kinetics/VideoData/"
DATA_PATH_CSV='Your_Path/kinetics/labels/train.csv'
OUTPUT_DIR='Output_Path/VITB_Kinetics_Pretraining_Dino_single_node/'

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
        --master_port 12312 run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --data_path_csv ${DATA_PATH_CSV} \
        --mask_type tube \
        --target_type dino_v1 \
        --distillation_teacher dino_b \
        --loss_func SWAV \
        --run_name FULL_Kinetics_DINO_pretraining \
        --mask_ratio 0.9 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 2 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 5 \
        --epochs 801 \
        --output_dir ${OUTPUT_DIR} \
        --normlize_target True \
        --num_prototypes 3000 \
        --sinkhorn_eps 0.05 \
        --sinkhorn_iterations 10 \
        --augmentation multi_scale_crop \
