#!/bin/bash

n_shadows=64
gpus=(0 1 2 3 4 5 6 7)
exp_per_gpu=$((n_shadows / ${#gpus[@]}))

model="vgg19"
dataset="cifar10"
mode="eval"

for ((i=0; i<$n_shadows; i++)); do
  gpu_id=${gpus[$((i % ${#gpus[@]}))]}  
  exp_id=$i  
  log_file="logs/${model}_${dataset}/inference/log_${exp_id}"
  savedir="exp/${model}_${dataset}/${exp_id}"

  echo "Starting inference experiment $exp_id on GPU $gpu_id"

  CUDA_VISIBLE_DEVICES=$gpu_id python3 -u inference.py \
    --savedir=$savedir --model $model --mode $mode &> $log_file &

  if (( (i + 1) % ${#gpus[@]} == 0 )); then
    wait  
  fi
done

