#!/bin/bash

# Interval in seconds
INTERVAL=0.5

# Infinite loop to monitor GPU usage
while true; do
    clear  # Clear the screen for better readability
    echo "Monitoring GPU usage (updates every $INTERVAL seconds)..."
    nvidia-smi
    sleep $INTERVAL
done