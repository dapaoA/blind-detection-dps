import argparse
import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch

from guided_diffusion.blind_condition_methods import get_conditioning_method
from guided_diffusion.gaussian_diffusion import create_sampler
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from util.img_utils import clear_color
from util.loader import create_mask, denormalize, load_yaml, prepare_dataloader, apply_gaussian_blur
from util.logger import get_logger


def load_config():
    # Configurations
    parser = argparse.ArgumentParser()
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

    args = parser.parse_args()

    return args

def main():
    # Configurations
    args = load_config()
    # logger
    logger = get_logger()

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    # Load configurations
    model_config = load_yaml(args.img_model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
    data_config = load_yaml(args.data_config)['data']
    # Kernel configs to namespace save space
    args.kernel = task_config["kernel"]
    args.kernel_size = task_config["kernel_size"]
    args.intensity = task_config["intensity"]

    # Load model
    img_model = create_model(**model_config)
    img_model = img_model.to(device)
    img_model.eval()

    model = {'img': img_model, 'kernel': None}

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
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

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config)
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)

    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    logger.info(f"work directory is created as {out_path}")
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)


    loader, mean_image, std_image = prepare_dataloader(data_config, model_config, if_train=False, task_config=task_config)
    # set seed for reproduce
    np.random.seed(123)

    # Do Inference

    sdps_iteration = 3
    for i, ref_img in enumerate(loader):
        if i == 1:
            mask = torch.ones(ref_img.shape, device=device)
            for _ in range(sdps_iteration):
                logger.info(f"Inference for image {i}")
                fname = str(i).zfill(5) + '.png'
                ref_img = ref_img.to(device)
                os.makedirs(os.path.join(out_path, 'label' + str(_)), exist_ok=True)

                # Save original images before denormalization
                plt.imsave(os.path.join(out_path, 'label' + str(_), 'img_orig_'+fname), clear_color(ref_img), cmap='gray')
                # Forward measurement model (Ax + n)
                y_n = ref_img
                # Set initial sample
                # !All values will be given to operator.forward(). Please be aware it.
                x_start = {'img': torch.randn(ref_img.shape, device=device).requires_grad_(),
                        'kernel': mask}

                # !prior check: keys of model (line 74) must be the same as those of x_start to use diffusion prior.
                for k in x_start:
                    if k in model.keys():
                        logger.info(f"{k} will use diffusion prior")
                    else:
                        logger.info(f"{k} will use uniform prior.")

                # sample
                print(x_start['img'].shape)
                sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path)

                os.makedirs(os.path.join(out_path, 'label' + str(_)), exist_ok=True)

                # Save original images before denormalization
                plt.imsave(os.path.join(out_path, 'label' + str(_), 'img_orig_'+fname), clear_color(ref_img), cmap='gray')
                plt.imsave(os.path.join(out_path, 'label' + str(_), 'recon_orig_'+fname), clear_color(sample['img']), cmap='gray')

                # Denormalize images
                y_n_denorm = denormalize(y_n, mean_image, std_image, device)
                ref_img_denorm = denormalize(ref_img, mean_image, std_image, device)
                sample_img_denorm = denormalize(sample['img'], mean_image, std_image, device)
                print(sample_img_denorm)
                print(ref_img_denorm)

                # Calculate mask using denormalized images
                mask = create_mask(sample['img'], y_n, threshold=0.4, device=device)
                plt.imsave(os.path.join(out_path, 'recon' + str(_), 'mask_'+fname), clear_color(mask), cmap='gray')
                mask = apply_gaussian_blur(mask, kernel_size=5, sigma=2.0, threshold_of_blur=0.5)
                plt.imsave(os.path.join(out_path, 'recon' + str(_), 'blurred_mask_'+fname), clear_color(mask), cmap='gray') 
                # Create directories
                os.makedirs(os.path.join(out_path, 'input' + str(_)), exist_ok=True)
                os.makedirs(os.path.join(out_path, 'label' + str(_)), exist_ok=True)
                os.makedirs(os.path.join(out_path, 'recon' + str(_)), exist_ok=True)

                # Save denormalized images
                plt.imsave(os.path.join(out_path, 'input' + str(_), fname), clear_color(y_n), cmap='gray')
                plt.imsave(os.path.join(out_path, 'label' + str(_), 'img_'+fname), clear_color(ref_img_denorm), cmap='gray')
                plt.imsave(os.path.join(out_path, 'recon' + str(_), 'img_'+fname), clear_color(sample_img_denorm), cmap='gray')

if __name__ == '__main__':
    main()
