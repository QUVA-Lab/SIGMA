#!/bin/bash
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=4
#SBATCH --constraint=v100
#SBATCH --mem=256GB
#SBATCH --cpus-per-gpu=4


# Load modules for MPI and other parallel libraries
module purge

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((RANDOM % (65535 - 1024 + 1) + 1024))
echo $MASTER_ADDR
echo $MASTER_PORT

source activate fgvssl

DATA_PATH="/ssdstore/fmthoker/20bn-something-something-v2/something-something-v2-videos_avi/"
DATA_PATH_CSV='/ssdstore/fmthoker/20bn-something-something-v2/something-something-v2-annotations/train.csv'

         OUTPUT_DIR='Output_Path/VITB_SSV2_Pretraining_Dino_multi_node/'

JOB_NAME=$1
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-16}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}

srun -p --job-name=${JOB_NAME} \
        --gres=gpu:${GPUS_PER_NODE} \
        --ntasks=${GPUS} \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --cpus-per-task=${CPUS_PER_TASK} \
        --kill-on-bad-exit=1 \
        ${SRUN_ARGS} \
        python -u run_mae_pretraining.py \
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
        --batch_size 32 \
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
                                            
