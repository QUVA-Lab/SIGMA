DATA_PATH='/nvmestore/ssalehi/kinetics/labels/'
MODEL_PATH='/nvmestore/ssalehi/runs/FULL_Kinetics_MLP_SWAV_3K_SYMMETRIC_EP_0.05/checkpoint-799.pth'
OUTPUT_DIR='/nvmestore/ssalehi/expirements_finetune/vit-base-ckpts/SIGMA_in_mgm_kinetics_MLP_SWAV_3K_E05_799/eval_lr_1e-3_epoch_40_8gpus_no_update_freq'


#OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 \
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 \
        --master_port 12320 run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set Kinetics-400 \
        --nb_classes 400 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 8 \
        --num_sample 1 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 1 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --lr 5e-4 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --epochs 100 \
        --dist_eval \
        --test_num_segment 7 \
        --test_num_crop 3 \
        --num_workers 10 \
        --update_freq 8