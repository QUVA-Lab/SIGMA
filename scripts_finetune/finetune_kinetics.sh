#Finetuning Kinetics400 
DATA_PATH='/ssdstore/fmthoker/kinetics/labels/'
# Set the path to save checkpoints
MODEL_PATH='/ivi/zfs/s0/original_homes/fthoker/sigma_final_models/pretrain/kinetics-400/kinetics_400_vit_b_sigma_with_dino.pth'
OUTPUT_DIR='expirements_finetune/finetune_kinetics-400_pretrained_with_kinetics_400_vit_b_sigma_with_dino/eval_lr_1e-3_epoch_100/log.txt'

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
        --master_port 12320  run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --nb_classes 400 \
        --data_set Kinetics-400 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 4 \
        --num_sample 1 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 1 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --lr 1e-3 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --epochs 100 \
        --dist_eval \
        --test_num_segment 7 \
        --test_num_crop 3 \
        --num_workers 8 \
        --update_freq 1
