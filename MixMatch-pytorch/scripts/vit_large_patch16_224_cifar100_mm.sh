#!/bin/bash

# Define array for eps values
eps_values=(1 5 10 100)

# Initialize GPU index
gpu=0

# Iterate over eps values
for eps in "${eps_values[@]}"; do
    # Create log directory if it doesn't exist
    log_dir="logs/vit_large_patch16_224_mm"
    mkdir -p "$log_dir"
    
    # Run the Python script in the background
    python -u train_large_cifar-semi.py \
        --gpu "$gpu" --model vit_large_patch16_224 --dataset large_cifar100 \
        --n-labeled 4000 --lr 0.0005 --eps "$eps" \
        > "$log_dir/cifar100@4000_lr_0.0005_eps_${eps}.log" 2>&1 &
    
    echo "Started process with eps=$eps and gpu=$gpu. Log: $log_dir/cifar100@4000_lr_0.0005_eps_${eps}.log"
    
    # Increment GPU index
    gpu=$((gpu + 2))
done

echo "All processes are runnning now."