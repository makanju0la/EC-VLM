#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=1088
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=4

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

data=/data/datasets/EC_pretraining/OFA/dataset/caption/caption_data/caption_val.tsv
path=/home/grads/mogunleye/Research/EC_pretraining/OFA/run_scripts/caption/stage2_checkpoints/nl/1e-5_3/checkpoint_2_5000.pt     #change
result_path=../../results/caption/nl/val
selected_cols=1,4,2
split='test'

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=caption \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --max-len-b=16 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"eval_cider\":False,\"selected_cols\":\"${selected_cols}\"}"

python coco_eval.py ../../results/caption/nl/val/test_predict.json /data/datasets/EC_pretraining/OFA/dataset/caption/caption_data/test_caption_coco_format.json
