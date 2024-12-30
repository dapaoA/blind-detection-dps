#!/bin/bash

# Run the deblur demo with specified configurations
python3 blind_anomaly_demo.py \
    --img_model_config=configs/model_config.yaml \
    --kernel_model_config=configs/kernel_model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/anomaly_detection_config.yaml \
    --data_config=configs/data_config_leather.yaml \
    --reg_ord=1 \
    --reg_scale=1.0


# Run the deblur demo with specified configurations
python3 blind_anomaly_demo.py \
    --img_model_config=configs/model_config.yaml \
    --kernel_model_config=configs/kernel_model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/anomaly_detection_sdps_config.yaml \
    --data_config=configs/data_config_bottle.yaml \
    --reg_ord=1 \
    --reg_scale=1.0


# Run the deblur demo with specified configurations
python3 blind_anomaly_demo.py \
    --img_model_config=configs/model_config.yaml \
    --kernel_model_config=configs/kernel_model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/anomaly_detection_sdps_config.yaml \
    --data_config=configs/data_config_capsule.yaml \
    --reg_ord=1 \
    --reg_scale=1.0


python3 blind_anomaly_demo_ddpm.py \
    --img_model_config=configs/model_config.yaml \
    --kernel_model_config=configs/kernel_model_config.yaml \
    --diffusion_config=configs/diffusion_ddpm_config.yaml \
    --task_config=configs/anomaly_detection_sdps_config.yaml \
    --data_config=configs/data_config_bottle.yaml \
    --reg_ord=1 \
    --reg_scale=1.0

python3 blind_anomaly_demo_ddpm.py \
    --img_model_config=configs/model_config.yaml \
    --kernel_model_config=configs/kernel_model_config.yaml \
    --diffusion_config=configs/diffusion_ddpm_config.yaml \
    --task_config=configs/anomaly_detection_sdps_config.yaml \
    --data_config=configs/data_config_bottle.yaml \
    --reg_ord=1 \
    --reg_scale=1.0 \
    --if_inference=0 \
    --if_evaluate=1    