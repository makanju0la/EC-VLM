#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path /home/grads/mogunleye/Research/LLaVA/checkpoints/llava-merged-clip \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version/MME_Benchmark \
    --answers-file ./playground/data/eval/MME/answers/llava-v1.5-13b-clip.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.5-13b-clip

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-13b-clip
