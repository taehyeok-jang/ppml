#!/bin/bash

# n_shadows=64/
n_shadows=8

gpus=(0 1 2 3 4 5 6 7)
shadows_per_gpu=$((n_shadows / ${#gpus[@]}))

for ((i=0; i<$n_shadows; i++)); do
  gpu_id=${gpus[$((i % ${#gpus[@]}))]}  
  shadow_id=$i  
  log_file="logs/log_${shadow_id}" 

  echo "Starting shadow experiment $shadow_id on GPU $gpu_id"

  CUDA_VISIBLE_DEVICES=$gpu_id python3 -u train.py \
    --model vgg19 --epochs=20 --lr=0.001 --weight_decay=0.0001 \
    --n_shadows=$n_shadows --shadow_id=$shadow_id --debug \
    &> $log_file &  

  # 8개의 GPU가 동시에 처리 중이면 대기
  if (( (i + 1) % ${#gpus[@]} == 0 )); then
    wait
  fi
done

wait

