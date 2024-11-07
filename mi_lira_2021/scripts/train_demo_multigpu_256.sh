#!/bin/bash

num_experiments=256
gpus=(0 1 2 3 4 5 6 7)
exp_per_gpu=$((num_experiments / ${#gpus[@]}))  # 각 GPU당 할당되는 실험 수

for ((i=0; i<$num_experiments; i++)); do
  gpu_id=${gpus[$((i % ${#gpus[@]}))]}  # GPU ID를 순환하면서 할당
  expid=$i  # 실험 ID는 0부터 시작
  log_file="logs/log_${expid}"  # 로그 파일 경로 설정
  
  echo "Starting experiment $expid on GPU $gpu_id"
  
  CUDA_VISIBLE_DEVICES=$gpu_id python3 -u train.py \
    --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2 \
    --num_experiments $num_experiments --expid $expid --logdir exp/cifar10 \
    &> $log_file &  # 로그 남기고 백그라운드에서 실행
  
  # 8개의 GPU가 동시에 처리 중이면 대기
  if (( (i + 1) % ${#gpus[@]} == 0 )); then
    wait
  fi
done

wait
