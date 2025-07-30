#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path ./checkpoints/llava-phi-clip-token-fintune\
    --question-file ./playground/data_promblems/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./datasets/mm-vet/images \
    --answers-file ./results/mm-vet/answers/llava-phi-clip-token-fintune.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1


python scripts/convert_mmvet_for_eval.py \
    --src ./results/mm-vet/answers/llava-phi-clip-token-fintune.jsonl \
    --dst ./results/mm-vet/answers/llava-phi-clip-token-fintune.json

