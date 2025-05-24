#!/bin/bash
#SBATCH -J ofa
#SBATCH -A virtual_presenter
#SBATCH -p v100_normal_q
#SBATCH --nodes=1
#SBATCH -t 23:30:00
#SBATCH --gres=gpu:2
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 3 > $SLURM_JOBID.gpu.log &


#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6091
export CUDA_VISIBLE_DEVICES=0,1
export GPUS_PER_NODE=2


########################## Evaluate Refcoco ##########################
user_dir=../../ofa_module
bpe_dir=../../utils/BPE
selected_cols=0,4,2,3

# data=../../dataset/refcoco_data/refcoco_val.tsv
# path=../../checkpoints/refcoco_base_best.pt
# result_path=../../results/refcoco
# split='refcoco_val'
# python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
#     ${data} \
#     --path=${path} \
#     --user-dir=${user_dir} \
#     --task=refcoco \
#     --batch-size=16 \
#     --log-format=simple --log-interval=10 \
#     --seed=7 \
#     --gen-subset=${split} \
#     --results-path=${result_path} \
#     --beam=5 \
#     --min-len=4 \
#     --max-len-a=0 \
#     --max-len-b=4 \
#     --no-repeat-ngram-size=3 \
#     --fp16 \
#     --num-workers=0 \
#     --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

# data=../../dataset/refcoco_data/refcoco_testA.tsv
# path=../../checkpoints/refcoco_base_best.pt
# result_path=../../results/refcoco
# split='refcoco_testA'
# python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
#     ${data} \
#     --path=${path} \
#     --user-dir=${user_dir} \
#     --task=refcoco \
#     --batch-size=16 \
#     --log-format=simple --log-interval=10 \
#     --seed=7 \
#     --gen-subset=${split} \
#     --results-path=${result_path} \
#     --beam=5 \
#     --min-len=4 \
#     --max-len-a=0 \
#     --max-len-b=4 \
#     --no-repeat-ngram-size=3 \
#     --fp16 \
#     --num-workers=0 \
#     --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

# data=../../dataset/refcoco_data/refcoco_testB.tsv
# path=../../checkpoints/refcoco_base_best.pt
# result_path=../../results/refcoco
# split='refcoco_testB'
# python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
#     ${data} \
#     --path=${path} \
#     --user-dir=${user_dir} \
#     --task=refcoco \
#     --batch-size=16 \
#     --log-format=simple --log-interval=10 \
#     --seed=7 \
#     --gen-subset=${split} \
#     --results-path=${result_path} \
#     --beam=5 \
#     --min-len=4 \
#     --max-len-a=0 \
#     --max-len-b=4 \
#     --no-repeat-ngram-size=3 \
#     --fp16 \
#     --num-workers=0 \
#     --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"



######################### Evaluate Refcocoplus ##########################
data=../../dataset/refcocoplus_data/refcocoplus_val.tsv
# path=../../checkpoints/refcocoplus_base_best.pt
path=../../run_scripts/refcoco/refcocoplus_scratch_checkpoints/10_5e-5_512/checkpoint_best.pt
result_path=../../results/refcocoplus_scratch
split='refcocoplus_val'
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=refcoco \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --min-len=4 \
    --max-len-a=0 \
    --max-len-b=4 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

data=../../dataset/refcocoplus_data/refcocoplus_testA.tsv
# path=../../checkpoints/refcocoplus_base_best.pt
path=../../run_scripts/refcoco/refcocoplus_scratch_checkpoints/10_5e-5_512/checkpoint_best.pt
result_path=../../results/refcocoplus_scratch
split='refcocoplus_testA'
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=refcoco \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --min-len=4 \
    --max-len-a=0 \
    --max-len-b=4 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

data=../../dataset/refcocoplus_data/refcocoplus_testB.tsv
# path=../../checkpoints/refcocoplus_base_best.pt
path=../../run_scripts/refcoco/refcocoplus_scratch_checkpoints/10_5e-5_512/checkpoint_best.pt
result_path=../../results/refcocoplus_scratch
split='refcocoplus_testB'
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=refcoco \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --min-len=4 \
    --max-len-a=0 \
    --max-len-b=4 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"



########################## Evaluate Refcocog ##########################
# data=../../dataset/refcocog_data/refcocog_val.tsv
# path=../../checkpoints/refcocog_base_best.pt
# result_path=../../results/refcocog
# split='refcocog_val'
# python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
#     ${data} \
#     --path=${path} \
#     --user-dir=${user_dir} \
#     --task=refcoco \
#     --batch-size=16 \
#     --log-format=simple --log-interval=10 \
#     --seed=7 \
#     --gen-subset=${split} \
#     --results-path=${result_path} \
#     --beam=5 \
#     --min-len=4 \
#     --max-len-a=0 \
#     --max-len-b=4 \
#     --no-repeat-ngram-size=3 \
#     --fp16 \
#     --num-workers=0 \
#     --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

# data=../../dataset/refcocog_data/refcocog_test.tsv
# path=../../checkpoints/refcocog_base_best.pt
# result_path=../../results/refcocog
# split='refcocog_test'
# python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
#     ${data} \
#     --path=${path} \
#     --user-dir=${user_dir} \
#     --task=refcoco \
#     --batch-size=16 \
#     --log-format=simple --log-interval=10 \
#     --seed=7 \
#     --gen-subset=${split} \
#     --results-path=${result_path} \
#     --beam=5 \
#     --min-len=4 \
#     --max-len-a=0 \
#     --max-len-b=4 \
#     --no-repeat-ngram-size=3 \
#     --fp16 \
#     --num-workers=0 \
#     --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"
