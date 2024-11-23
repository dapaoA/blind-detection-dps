#!/bin/bash

# Define an array of parameters
params=("data_config_leather")


# for param in "${params[@]}"; do
#     echo "Running script with parameter: $param"
#     python3 create_mean.py --data_config "configs/$param.yaml"
# done
# Loop over each parameter and run the Python script
for param in "${params[@]}"; do
    echo "Running script with parameter: $param"
    python3 create_mean.py --data_config "configs/$param.yaml"
done
for param in "${params[@]}"; do
    echo "Running script with parameter: $param"
    python3 train_diffusion.py --data_config "configs/$param.yaml"
done
