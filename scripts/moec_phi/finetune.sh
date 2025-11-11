#!/bin/bash
#export WANDB_MODE=offline

export NCCL_TIMEOUT=3600 

deepspeed    --include localhost:0,1,2,3,4,5,6,7\
    llava/train/moec_train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./models/microsoft-phi-2 \
    --version phi \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-phi-siglipconvnextclipdino-moec-attention-token-pretrain/mm_projector.bin \
    --pretrain_siglip_mm_mlp_adapter ./checkpoints/llava-phi-siglipconvnextclipdino-moec-attention-token-pretrain/siglip_mm_projector.bin \
    --pretrain_convnext_mm_mlp_adapter ./checkpoints/llava-phi-siglipconvnextclipdino-moec-attention-token-pretrain/convnext_mm_projector.bin \
    --pretrain_convnext_mm_mlp_adapter ./checkpoints/llava-phi-siglipconvnextclipdino-moec-attention-token-pretrain/convnext_mm_projector.bin \
    --scales 1 \
    --mm_projector_type smoe_mlp \
    --mlp_smoe True \
    --num_experts 4 \
    --num_selected 2 \
    --balance_loss_coef 0.1 \
    --router_z_loss_coef 0.01 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-phi-siglipconvnextclipdino-moec-attention-token-fintune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200\
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
    --logging_dir ./checkpoints/llava-phi-siglipconvnextclipdino-moec-attention-token-fintune/runs/
