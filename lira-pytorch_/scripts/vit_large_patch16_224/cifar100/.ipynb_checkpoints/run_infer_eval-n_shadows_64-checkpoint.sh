#!/bin/bash

n_shadows=64
# gpus=(0 1 2 3 4 5 6 7)
gpus=(1 2 3 4 5 6 7)
exp_per_gpu=$((n_shadows / ${#gpus[@]}))

model="vit_large_patch16_224"
dataset="cifar100"
mode="eval"

for ((i=0; i<$n_shadows; i++)); do
  gpu_id=${gpus[$((i % ${#gpus[@]}))]}  
  exp_id=$i  
  log_dir="logs/${model}_${dataset}/inference"
  savedir="exp/${model}_${dataset}/${exp_id}"

  mkdir -p "$log_dir"

  log_file="logs/${model}_${dataset}/inference/log_${exp_id}"

  echo "Starting inference experiment $exp_id on GPU $gpu_id"

  CUDA_VISIBLE_DEVICES=$gpu_id python3 -u inference.py \
    --model $model --dataset $dataset \
    --mode $mode --savedir=$savedir &> $log_file &

  if (( (i + 1) % ${#gpus[@]} == 0 )); then
    wait  
  fi
done

