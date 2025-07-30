#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path   ./checkpoints/llava-phi-siglipconvnext-moec-token-fintune  \
    --question-file ./playground/data_promblems/MME/llava_mme.jsonl \
    --image-folder ./playground/data_promblems/MME/MME_Benchmark_release_version/MME_Benchmark \
    --answers-file ./playground/data_promblems/MME/answers/llava-phi-siglipconvnext-moec-token-fintune.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data_promblems/MME/

python convert_answer_to_mme.py --experiment llava-phi-siglipconvnext-moec-token-fintune

cd eval_tool

python calculation.py --results_dir answers/llava-phi-siglipconvnext-moec-token-fintune