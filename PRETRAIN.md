# Pre-training SIGMA 

The implementation of our SIGMA supports **single and multi-node distributed training**. We provide the **off-the-shelf** scripts in the [scripts_pretrain](./scripts_pretrain).

-  For example, to pre-train SIGMA ViT-Base on **Something-Something V2** on a single node with 8 GPUs (1 nodes x 8 GPUs), you can run

  ```bash
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
  ```


### Note:
- `lr` here is the base learning rate and is set to `1.5e-4` as default. The ` actual lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `` actual lr`` = `lr` * total batch size / 256.

## Slurm

To help the community to reproduce our results on slurm cluster, We provide the **off-the-shelf** scripts in the [scripts_pretrain](./scripts_pretrain).

-  For example, to pre-train SIGMA ViT-Base on **Something-Something V2** with 16 GPUs (4 nodes x 4 GPUs), you can run


  ```bash
          MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
          export MASTER_PORT=$((RANDOM % (65535 - 1024 + 1) + 1024))
          echo $MASTER_ADDR
          echo $MASTER_PORT
          
          source activate fgvssl
          
          DATA_PATH="/ssdstore/fmthoker/20bn-something-something-v2/something-something-v2-videos_avi/"
          DATA_PATH_CSV='/ssdstore/fmthoker/20bn-something-something-v2/something-something-v2-annotations/train.csv'
          
          OUTPUT_DIR='/ivi/zfs/s0/original_homes/fthoker/runs/VITB_SSV2_Test/'
          
          JOB_NAME=$1
          # 4 for 1 node, 16 for 4 node, etc.
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

  ```
