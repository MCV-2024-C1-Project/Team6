#!/bin/bash

# Define the list of values you want to loop over
block_fraction=(4 8 16 24 32)  # Add other values if needed
thresholds=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)  # Add other values if needed

if [ "$1" == "init" ]; then
    # Loop through each combination of block size and threshold
    for block_size in "${block_fraction[@]}"; do
        for threshold in "${thresholds[@]}"; do
            # Run the command with the current values
            python main.py init --db_path ../W3/BBDD/ --descriptor-type Texture-DCTPiecewise_${block_size}_${threshold}_0-0
        done
    done
elif [ "$1" == "predict" ]; then
    for block_size in "${block_fraction[@]}"; do
        echo "block fraction: ${block_size}"
        for threshold in "${thresholds[@]}"; do
            # Run the command with the current values
            python main.py predict --input ../W3/qsd1_w3 --result-number $2 --descriptor-type Texture-DCTPiecewise_${block_size}_${threshold}_0-0 --measure Cosine-Median --evaluate
        done
    done
else
    for block_size in "${block_fraction[@]}"; do
        for threshold in "${thresholds[@]}"; do
            # Run the command with the current values
            python main.py init --db_path ../W3/BBDD/ --descriptor-type Texture-DCTPiecewise_${block_size}_${threshold}_0-0
        done
    done

    for block_size in "${block_fraction[@]}"; do
        for threshold in "${thresholds[@]}"; do
            # Run the command with the current values
            python main.py predict --input ../W3/qsd1_w3 --result-number $2 --descriptor-type Texture-DCTPiecewise_${block_size}_${threshold}_0-0 --measure Cosine-Median --evaluate
        done
    done
fi