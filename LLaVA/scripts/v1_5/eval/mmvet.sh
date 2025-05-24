#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path /home/directory/LLaVA/checkpoints/llava-merged-clip-full \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/llava-v1.5-13b-clip-full.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/llava-v1.5-13b-clip-full.jsonl \
    --dst ./playground/data/eval/mm-vet/results/llava-v1.5-13b-clip-full.json

