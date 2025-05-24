#!/bin/bash
SPLIT="mmbench_dev_clip_full_new_20250519"

python -m llava.eval.model_vqa_mmbench \
    --model-path /home/directory/LLaVA/checkpoints/llava-merged-clip-full \
    --question-file ./playground/data/eval/mmbench/mmbench_dev_20250519.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/llava-merged-clip-full.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/mmbench_dev_20250519.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment llava-merged-clip-full
