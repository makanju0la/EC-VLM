#!/bin/bash

SPLIT="mmbench_dev_cn_clip_20250519"

python -m llava.eval.model_vqa_mmbench \
    --model-path /home/directory/LLaVA/checkpoints/llava-merged-clip \
    --question-file ./playground/data/eval/mmbench/mmbench_dev_cn_20250519.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/llava-merged-cn-clip.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/mmbench_dev_cn_20250519.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment llava-merged-cn-clip
