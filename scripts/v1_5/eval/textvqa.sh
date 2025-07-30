#!/bin/bash
cd /home/pangyq/experiments/LLaVA
python -m llava.eval.model_vqa_loader \
    --model-path  ./checkpoints/llava-phi-convnext-token-fintune \
    --question-file ./playground/data_promblems/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./datasets/TextVQA/train_images/ \
    --answers-file ./results/textvqa/answers/llava-phi-convnext-token-fintun.jsonl \
    --temperature 0\
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./datasets/TextVQA/TextVQA_0.5.1_val.json \
    --result-file ./results/textvqa/answers/llava-phi-convnext-token-fintun.jsonl
