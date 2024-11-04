#!/bin/bash

# Run the deblur demo with specified configurations
python3 blind_anomaly_demo.py \
    --img_model_config=configs/model_config.yaml \
    --kernel_model_config=configs/kernel_model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/anomaly_detection_config.yaml \
    --reg_ord=1 \
    --reg_scale=1.0