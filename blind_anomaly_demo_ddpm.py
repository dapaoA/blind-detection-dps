import argparse
import os
from functools import partial

import torch
import numpy as np
from guided_diffusion.blind_condition_methods import get_conditioning_method
from guided_diffusion.gaussian_diffusion import create_sampler
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from util.img_utils import clear_color
from util.loader import create_mask, denormalize, load_yaml, prepare_dataloader
from util.logger import get_logger
from evaluate import compute_auroc
import matplotlib.pyplot as plt


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_parameters():
    """加载和处理所有配置参数"""
    parser = argparse.ArgumentParser()
    # Configurations
    parser.add_argument('--img_model_config', type=str, default='configs/model_config.yaml')
    parser.add_argument('--kernel_model_config', type=str, default='configs/kernel_model_config.yaml')
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config.yaml')
    parser.add_argument('--task_config', type=str, default='configs/anomaly_detection_sdps_config.yaml')
    parser.add_argument('--data_config', type=str, default='configs/data_config.yaml')
    # Training
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    # Regularization
    parser.add_argument('--reg_scale', type=float, default=0.1)
    parser.add_argument('--reg_ord', type=int, default=0, choices=[0, 1])
    parser.add_argument('--if_evaluate', type=str2bool, default=False)
    parser.add_argument('--if_inference', type=str2bool, default=False)
    args = parser.parse_args()
    logger = get_logger()

    # Load YAML configs
    model_config = load_yaml(args.img_model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
    data_config = load_yaml(args.data_config)['data']
    
    # Add kernel configs to args namespace
    args.kernel = task_config["kernel"]
    args.kernel_size = task_config["kernel_size"]
    args.intensity = task_config["intensity"]

    return args, logger, model_config, diffusion_config, task_config, data_config

def setup_model_and_task(args, logger, model_config, diffusion_config, task_config, device):

    # Create model
    img_model = create_model(**model_config)
    img_model = img_model.to(device)
    img_model.eval()
    model = {'img': img_model, 'kernel': None}

    # Setup measurement operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Setup conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
    measurement_cond_fn = cond_method.conditioning

    # Add regularization
    # Not to use regularization, set reg_scale = 0 or remove this part.
    regularization = {'kernel': (args.reg_ord, args.reg_scale)}
    measurement_cond_fn = partial(measurement_cond_fn, regularization=regularization)
    if args.reg_scale == 0.0:
        logger.info("Got kernel regularization scale 0.0, skip calculating regularization term.")
    else:
        logger.info(f"Kernel regularization : L{args.reg_ord}")

    # Create sampler
    sampler = create_sampler(**diffusion_config)
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=None)

    return model, sampler, sample_fn, operator, noiser

def run_inference(loader, sample_fn, sampler, diffusion_config, out_path, device, mean_image, std_image):

    sdps_iteration = 1
    os.makedirs(os.path.join(out_path, 'recon'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'mask_norm'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'progress'), exist_ok=True)

    for i, ref_img_dict in enumerate(loader):
        batch_size = ref_img_dict['image'].shape[0]
        
        for _ in range(sdps_iteration):
            # Get batch data
            ref_img = ref_img_dict['image'].to(device)
            y_n = ref_img
            
            # Set initial sample
            x_start = sampler.q_sample(ref_img, t=torch.tensor([0], device=device)).to(device)
            
            # Sample
            sample = sample_fn(x_start=x_start, measurement=y_n, record=False, 
                             save_root=out_path, start_t=diffusion_config['start_t'])
            
            # Process each sample in batch
            for batch_idx in range(batch_size):
                fname = os.path.basename(ref_img_dict['category'][batch_idx]) + '_' + ref_img_dict['name'][batch_idx]
                
                curr_y_n = y_n[batch_idx:batch_idx+1]
                curr_sample = sample[batch_idx:batch_idx+1]
                
                # Calculate mask before denormalization
                curr_mask_norm = create_mask(curr_sample, curr_y_n, threshold=0.1, device=device)
                
                # Denormalize images
                y_n_denorm = denormalize(curr_y_n, mean_image, std_image, device)
                sample_img_denorm = denormalize(curr_sample, mean_image, std_image, device)

                # Calculate mask after denormalization
                curr_mask_denorm = create_mask(sample_img_denorm, y_n_denorm, threshold=0.1, device=device)

                # Save masks and reconstruction
                plt.imsave(os.path.join(out_path, 'mask', fname), clear_color(curr_mask_denorm), cmap='gray')
                plt.imsave(os.path.join(out_path, 'mask_norm', fname), clear_color(curr_mask_norm), cmap='gray')
                plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample_img_denorm), cmap='gray')


def evaluate_all(loader, out_path, save_dir=None):
    return compute_auroc(loader, out_path)


def main():
    # Setup logger
    logger = get_logger()

    # Load parameters
    args, logger, model_config, diffusion_config, task_config, data_config = load_parameters()
    # Setup device
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    # Setup model and task
    model, sampler, sample_fn, operator, noiser = setup_model_and_task(
        args, logger, model_config, diffusion_config, task_config, device
    )

    # Create output directory
    out_path = os.path.join(args.save_dir, task_config['measurement']['operator']['name']) + '_' + data_config['name']
    os.makedirs(out_path, exist_ok=True)

    # Prepare data
    loader, mean_image, std_image = prepare_dataloader(data_config, model_config, if_train=False)
    
    # Set random seed
    np.random.seed(123)

    # Run inference
    if args.if_inference:
        run_inference(loader, sample_fn, sampler, diffusion_config, out_path, device, mean_image, std_image)

    # Run evaluation
    if args.if_evaluate:
        print("out_path: ", out_path)
        auroc = evaluate_all(loader, out_path, save_dir=out_path)
        print(f"Overall AUROC: {auroc:.4f}")

        # 保存AUROC
        with open(os.path.join(out_path, 'auroc.txt'), 'w') as f:
            f.write(f"AUROC: {auroc:.4f}")

if __name__ == '__main__':
    main()