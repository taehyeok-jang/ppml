#!/bin/bash

# Parameters
gpus=(0 1 2 3 4 5 6 7)  # Available GPUs
n_labeled=(10000 40000)
lrs=(0.02 0.05 0.1)

# Create logs directory if it doesn't exist
mkdir -p logs/efficientnet_b7_only

# GPU index tracker
gpu_index=0
num_gpus=${#gpus[@]}  # Total number of GPUs

# Generate training commands
for labeled in "${n_labeled[@]}"; do
  for lr in "${lrs[@]}"; do
    # Get the current GPU
    gpu=${gpus[$gpu_index]}
    
    echo "Starting training on GPU $gpu with n_labeled=$labeled and lr=$lr"
    python -u train_cifar100-sup.py \
    --gpu "$gpu" --model efficientnet_b7 --dataset cifar100 \
    --n-labeled "$labeled" --lr "$lr" \
    > logs/efficientnet_b7_only/cifar100@"$labeled"_lr_"$lr".log 2>&1 &
    
    # Update GPU index (circular rotation)
    gpu_index=$(( (gpu_index + 1) % num_gpus ))
  done
done

echo "All training scripts are running in the background."