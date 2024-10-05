from functools import partial
import os
import argparse
import yaml

import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.blind_condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_operator, get_noise
from guided_diffusion.unet import create_model_for_train
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from motionblur.motionblur import Kernel
from util.img_utils import Blurkernel, clear_color
from util.logger import get_logger


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config



def train(segment_model, loader, sample_fn, num_epochs=100, batch_size=32, lr=1e-4):
    device = next(segment_model.parameters()).device
    optimizer = torch.optim.Adam(segment_model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            
            # Move batch to device
            batch = batch.to(device)
            
            # Sample t uniformly for each example in the batch
            t = torch.randint(0, sample_fn.num_timesteps, (batch_size,), device=device).long()
            
            # Forward process: add noise to the input
            noise = torch.randn_like(batch)
            x_t = sample_fn.q_sample(x_start=batch, t=t, noise=noise)
            
            # Predict the noise
            predicted_noise = segment_model(x_t, t)
            
            # Calculate loss
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            
            # Backpropagate and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    return segment_model



def main():
    # Configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_model_config', type=str, default='configs/model_config.yaml')
    parser.add_argument('--segment_model_config', type=str, default='configs/segment_model_config.yaml')
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config.yaml')
    parser.add_argument('--task_config', type=str, default='configs/motion_deblur_config.yaml')
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
    segment_model_config = load_yaml(args.segment_model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    # Segment configs to namespace save space
    args.segment = task_config["segment"]
    args.segment_size = task_config["segment_size"]
    args.intensity = task_config["intensity"]

    # Create model
    segment_model = create_model_for_train(**segment_model_config)
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=segment_model, measurement_cond_fn=None)
   
    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=True)

    # set seed for reproduce
    np.random.seed(123)
    
    # Do training
    segment_model = train(segment_model, loader, sample_fn)



if __name__ == '__main__':
    main()
