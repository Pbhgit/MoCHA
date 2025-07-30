#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path  ./checkpoints/llava-phi-siglipclipconvnext-moec-attention-token-fintune \
    --question-file ./playground/data_promblems/pope/llava_pope_test.jsonl \
    --image-folder ./datasets/vqav2/val2014 \
    --answers-file ./results/pope/answers/llava-phi-siglipclipconvnext-moec-attention-token-fintune.jsonl \
    --temperature 0 \
    --conv-mode phi

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data_promblems/pope/coco \
    --question-file ./playground/data_promblems/pope/llava_pope_test.jsonl \
    --result-file ./results/pope/answers/llava-phi-siglipclipconvnext-moec-attention-token-fintune.jsonl
