#!/bin/bash
#export WANDB_MODE=offline

# export NCCL_TIMEOUT=3600 

deepspeed    --include localhost:0,1,2,3,5,6,7 \
    llava/train/copy_train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./models/lmsys-vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-siglipdinoconvnextclip-pretrain/mm_projector.bin \
    --pretrain_dino_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-siglipdinoconvnextclip-pretrain/dino_mm_projector.bin \
    --pretrain_siglip_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-siglipdinoconvnextclip-pretrain/siglip_mm_projector.bin \
    --pretrain_convnext_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-siglipdinoconvnextclip-pretrain/convnext_mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-siglipdinoconvnextclip-fintune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4\
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 720\
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2560 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard\
    --logging_dir ./checkpoints/llava-v1.5-7b-siglipdinoconvnextclip-fintune/runs/
