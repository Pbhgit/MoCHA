#!/bin/bash
cd ./experiments/LLaVA

python -m llava.eval.model_vqa_science \
    --model-path ./checkpoints/llava-phi-siglipclipconvnext-moec-attention-token-fintune\
    --question-file ./playground/data_promblems/scienceqa/llava_test_CQM-A.json \
    --image-folder ./datasets/scienceQA/test \
    --answers-file ./results/scienceqa/answers/llava-phi-siglipclipconvnext-moec-attention-token-fintune.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode phi


python llava/eval/eval_science_qa.py \
    --base-dir ./datasets/scienceQA \
    --result-file ./results/scienceqa/answers/llava-phi-siglipclipconvnext-moec-attention-token-fintune.jsonl \
    --output-file ./results/scienceqa/answers/llava-phi-siglipclipconvnext-moec-attention-token-fintune_output.jsonl \
    --output-result ./results/scienceqa/answers/llava-phi-siglipclipconvnext-moec-attention-token-fintune_result.json
