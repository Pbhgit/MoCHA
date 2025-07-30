#!/bin/bash

SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path ./checkpoints/llava-phi-siglipclipconvnext-moec-attention-token-fintune \
    --question-file ./playground/data_promblems/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data_promblems/mmbench/answers/$SPLIT/llava-phi-siglipclipconvnext-moec-attention-token-fintune.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode phi

mkdir -p playground/data_promblems/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data_promblems/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data_promblems/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data_promblems/mmbench/answers_upload/$SPLIT \
    --experiment llava-phi-siglipclipconvnext-moec-attention-token-fintune