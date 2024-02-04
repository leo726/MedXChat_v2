#!/bin/bash

version=VisualGrounding
savepath="./save/$version"

rg_dataset='data_qwen/report_qwenvl.json'
vqa_dataset='data_qwen/vqa_qwenvl.json'
sd_dataset='data_qwen/sd_qwenvl.json'
vg_dataset='data_qwen/vg_NIH.json'
delta_file=None

python -u train_qwenvl.py \
    --llm_use_lora True \
    --dataset $dataset \
    --delta_file $delta_file \
    --lora_inference False \
    --batch_size 4 \
    --val_batch_size 16 \
    --max_length 240 \
    --num_workers 8 \
    --learning_rate 0.00002 \
    --devices 4 \
    --accelerator gpu \
    --precision bf16-mixed \
    --num_nodes 1 \
    --strategy ddp \
    --max_epochs 30 \
    --accumulate_grad_batches 2 \
    --num_sanity_val_steps 2 \
    --limit_val_batches 0.5 \
    --val_check_interval 0.3 \
    --savedmodel_path ${savepath} \
    2>&1 |tee -a ${savepath}/log.txt

