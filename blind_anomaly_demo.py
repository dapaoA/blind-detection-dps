from functools import partial
import os
import argparse
from PIL import Image

import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.blind_condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_operator, get_noise
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import Blurkernel, clear_color
from util.logger import get_logger
from util.loader import load_yaml, data_transformer_list


def main():
    # Configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_model_config', type=str, default='configs/model_config.yaml')
    parser.add_argument('--kernel_model_config', type=str, default='configs/kernel_model_config.yaml')
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config.yaml')
    parser.add_argument('--task_config', type=str, default='configs/anomaly_detection_config.yaml')
    # Training
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    # Regularization
    parser.add_argument('--reg_scale', type=float, default=0.1)
    parser.add_argument('--reg_ord', type=int, default=0, choices=[0, 1])
    
    args = parser.parse_args()
   
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
        logger.info(f"Got kernel regularization scale 0.0, skip calculating regularization term.")
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

    # Prepare dataloader
    data_config = task_config['data']
    data_config = data_config['data']
    mean_image_path = os.path.join(data_config['root'], 'mean.png')
    variance_path = os.path.join(data_config['root'], 'variance.npy')
    if_grayscale = model_config['grayscale']
    mean_image = Image.open(mean_image_path)
    mean_image = transforms.Compose([
        transforms.Resize((model_config['image_size'], model_config['image_size'])),
        transforms.ToTensor(),
    ])(mean_image)
    
    variance = np.load(variance_path)
    std_image = torch.from_numpy(np.sqrt(variance)).float()
    std_image = transforms.Compose([
        transforms.Resize((model_config['image_size'], model_config['image_size'])),
    ])(std_image)
    if if_grayscale:
        mean_image = mean_image.mean(dim=0, keepdim=True)
        std_image = std_image.mean(dim=0, keepdim=True)

    transform = data_transformer_list(mean_image, std_image,
                                      data_config['image_size'],
                                      data_config['image_size'])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # set seed for reproduce
    np.random.seed(123)
    
    # Do Inference
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)
        
        # Initialize random kernel mask with same spatial dims as ref_img but 1 channel
        kernel = torch.randn(1, 1, ref_img.shape[2], ref_img.shape[3], device=device)
        
        # Forward measurement model (Ax + n)
        y = operator.forward(ref_img, kernel) 
        y_n = noiser(y)
        
        # Set initial sample 
        # !All values will be given to operator.forward(). Please be aware it.
        x_start = {'img': torch.randn(ref_img.shape, device=device).requires_grad_(),
                   'kernel': torch.randn(kernel.shape, device=device).requires_grad_()}
        
        # !prior check: keys of model (line 74) must be the same as those of x_start to use diffusion prior.
        for k in x_start:
            if k in model.keys():
                logger.info(f"{k} will use diffusion prior")
            else:
                logger.info(f"{k} will use uniform prior.")
       
        # sample 
        sample = sample_fn(x_start=x_start, measurement=y_n, record=False, save_root=out_path)

        plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'label', 'ker_'+fname), clear_color(kernel))
        plt.imsave(os.path.join(out_path, 'label', 'img_'+fname), clear_color(ref_img))
        plt.imsave(os.path.join(out_path, 'recon', 'img_'+fname), clear_color(sample['img']))
        plt.imsave(os.path.join(out_path, 'recon', 'ker_'+fname), clear_color(sample['kernel']))

if __name__ == '__main__':
    main()
