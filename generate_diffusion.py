from functools import partial
import os
import argparse
import yaml
import numpy as np
import torch
import tqdm
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
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

def generate_and_save_images(model, model_name, sampler, device, out_path, dataset_name, folder_name, sample_fn, mean_image, std_image):
    model.eval()
    images = []
    with torch.no_grad():
        for _ in range(10):
            x_start = torch.randn(1, 3, model.image_size, model.image_size).to(device)
            sample = sample_fn(x_start=x_start, measurement=None, record=False, save_root=out_path)
            
            # Denormalize the generated sample
            sample = sample * std_image.to(device) + mean_image.to(device)
            
            # Clip values to [0, 1] range and convert to PIL image
            sample = torch.clamp(sample, 0, 1)
            sample = (sample * 255).byte().cpu().numpy()
            
            img = Image.fromarray(sample[0].transpose(1, 2, 0))
            images.append(img)

    # Create a 2x5 grid of images
    fig, axs = plt.subplots(2, 5, figsize=(25, 10))
    for i, img in enumerate(images):
        axs[i//5, i%5].imshow(img)
        axs[i//5, i%5].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.join(out_path, folder_name), exist_ok=True)
    os.makedirs(os.path.join(out_path, folder_name, dataset_name), exist_ok=True)
    plt.savefig(os.path.join(out_path, folder_name, dataset_name, f'model_{model_name}.png'))
    
    plt.close()
    model.train()


def main():
    # Configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default='configs/model_training_config.yaml')
    parser.add_argument('--diffusion_config', type=str, default='configs/train_diffusion_config.yaml')
    parser.add_argument('--data_config', type=str, default='configs/data_config.yaml')
    parser.add_argument('--model_path', type=str, default='results/checkpoint_epoch_images_with_mean_1500.pth')
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
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    data_config = load_yaml(args.data_config)
   
    # Load model
    model_path = args.model_path
    model = create_model_for_train(**model_config)

    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=None)
    
    # Load mean and variance
    mean_image_path = os.path.join(data_config['data']['root'], 'mean.png')
    variance_path = os.path.join(data_config['data']['root'], 'variance.npy')
    
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
    
    # Custom normalization transform
    class NormalizeWithMeanStd(object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std
        
        def __call__(self, tensor):
            return (tensor - self.mean) / self.std
    
    # Prepare dataloader
    data_config = data_config['data']
    transform = transforms.Compose([
        transforms.Resize((model_config['image_size'], model_config['image_size'])),
        transforms.ToTensor(),
        NormalizeWithMeanStd(mean_image, std_image)
    ])
    batch_size = 64
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=batch_size, num_workers=4, train=True)

    # Move mean and std to device
    mean_image = mean_image.to(device)
    std_image = std_image.to(device)

    # Generate images
    model_name = model_path.split('/')[-1].split('.')[0]
    folder_name = 'images_with_mean_and_std'
    dataset_name = data_config['name'] 
    dataset_name = ''
    generate_and_save_images(model, model_name, sampler, device, args.save_dir, dataset_name, folder_name, sample_fn, mean_image, std_image)

    logger.info("Image generation completed.")

if __name__ == '__main__':
    main()
