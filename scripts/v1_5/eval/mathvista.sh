#!/bin/bash

python -m llava.eval.model_vqa_mathvista \
    --model-path ./checkpoints/llava-v1.5-7b-siglipdinoconvnextclip-fintune \
    --answers-file ./results/mathvista/answers/llava-v1.5-7b-siglipdinoconvnextclip-fintune.json \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

# # extract answer
python llava/eval/extract_answer.py \
    --output_file llava-v1.5-7b-siglipdinoconvnextclip-fintune.json \
    --response_label response \
    --output_label extraction 

# calculate score
python llava/eval/calculate_score.py \
    --gt_file ./results/mathvista/data/testmini.json \
    --output_dir ./results/mathvista/answers \
    --output_file llava-v1.5-7b-siglipdinoconvnextclip-fintune_extraction.json \
    --score_file scores_llava-v1.5-7b-siglipdinoconvnextclip-fintune_extraction.json