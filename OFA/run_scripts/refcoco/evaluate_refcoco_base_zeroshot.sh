#!/bin/bash
#SBATCH -J ofa
#SBATCH -A virtual_presenter
#SBATCH -p v100_normal_q
#SBATCH --nodes=1
#SBATCH -t 4:30:00
#SBATCH --gres=gpu:2
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 3 > $SLURM_JOBID.gpu.log &

#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6081
export CUDA_VISIBLE_DEVICES=0,1
export GPUS_PER_NODE=2

user_dir=../../ofa_module
bpe_dir=../../utils/BPE
selected_cols=0,4,2,3

# data=../../dataset/refcocoplus_data/refcocoplus_val.tsv
# data=../../dataset/refcocoplus_data/refcocoplus_testA.tsv
data=../../dataset/refcocoplus_data/refcocoplus_testB.tsv
path=../../model_checkpoints/ofa_base.pt
result_path=../../results/refcocoplus_zeroshot_baseline
split='refcocoplus_testB'
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --bpe-dir=${bpe_dir} \
    --selected-cols=${selected_cols} \
    --task=refcoco \
    --patch-image-size=480 \
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
    --zero-shot \
    --fp16 \
    --num-workers=0

# data=../../dataset/refcocoplus_data/refcocoplus_testA.tsv
# # path=../../checkpoints/refcocoplus_base_best.pt
# path=../../model_checkpoints/ofa_base.pt
# result_path=../../results/refcocoplus_zeroshot_baseline
# split='refcocoplus_testA'
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

# data=../../dataset/refcocoplus_data/refcocoplus_testB.tsv
# # path=../../checkpoints/refcocoplus_base_best.pt
# path=../../model_checkpoints/ofa_base.pt
# result_path=../../results/refcocoplus_zeroshot_baseline
# split='refcocoplus_testB'
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
