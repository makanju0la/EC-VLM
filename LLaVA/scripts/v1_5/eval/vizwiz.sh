#!/bin/bash

#run evaluations on images in /test, taking questions from /vizwiz/llava_test.jsonl 
#and putting results in /answers/llava-v1.5-13b.jsonl
python -m llava.eval.model_vqa_loader \
    --model-path /home/grads/mogunleye/Research/LLaVA/checkpoints/llava-v1.5-13b \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

#put final result in /answers_upload/llava-v1.5-13b.json
python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/llava-v1.5-13b.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/llava-v1.5-13b.json
