#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path /home/grads/mogunleye/Research/LLaVA/checkpoints/llava-merged-clip \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /home/grads/mogunleye/Research/EC_pretraining/EC_data/LLaVa/coco2014_val \
    --answers-file ./playground/data/eval/pope/answers/llava-v1.5-13b-clip.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-v1.5-13b-clip.jsonl
