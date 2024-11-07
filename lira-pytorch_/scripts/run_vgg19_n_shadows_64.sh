#!/bin/bash

n_shadows=64
gpus=(0 1 2 3 4 5 6 7)
shadows_per_gpu=$((n_shadows / ${#gpus[@]}))  # 각 GPU당 할당되는 shadow 수

for ((i=0; i<$n_shadows; i++)); do
  gpu_id=${gpus[$((i % ${#gpus[@]}))]}  # GPU ID를 순환하면서 할당
  shadow_id=$i  # shadow ID는 0부터 시작
  log_file="logs/log_${shadow_id}"  # 로그 파일 경로 설정

  echo "Starting shadow experiment $shadow_id on GPU $gpu_id"

  CUDA_VISIBLE_DEVICES=$gpu_id python3 -u train.py \
    --model vgg19 --epochs=25 --lr=0.02 \
    --n_shadows=$n_shadows --shadow_id=$shadow_id --debug \
    &> $log_file &  # 로그 남기고 백그라운드에서 실행

  # 8개의 GPU가 동시에 처리 중이면 대기
  if (( (i + 1) % ${#gpus[@]} == 0 )); then
    wait
  fi
done

wait

