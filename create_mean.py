import os
import argparse
import yaml
import numpy as np
import torch
import tqdm
import torchvision.transforms as transforms
from PIL import Image
from data.dataloader import get_dataset

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def compute_and_save_mean_variance(dataset, image_size, save_path):
    mean_image = np.zeros((3, image_size, image_size), dtype=np.float32)
    variance_image = np.zeros((3, image_size, image_size), dtype=np.float32)
    
    # First pass: compute mean
    for i in tqdm.tqdm(range(len(dataset)), desc="Computing mean image"):
        img = dataset[i].numpy()
        mean_image += img
    
    mean_image /= len(dataset)
    
    # Second pass: compute variance
    for i in tqdm.tqdm(range(len(dataset)), desc="Computing variance image"):
        img = dataset[i].numpy()
        variance_image += (img - mean_image) ** 2
    
    variance_image /= len(dataset)
    
    # Replace zeros with ones in variance_image
    variance_image[variance_image == 0] = 1
    
    # Convert mean to PIL Image and save
    mean_image_uint8 = (mean_image * 255).astype(np.uint8)
    mean_image_uint8 = np.transpose(mean_image_uint8, (1, 2, 0))
    mean_image_pil = Image.fromarray(mean_image_uint8)
    mean_image_pil.save(os.path.join(save_path, "mean.png"))
    
    # Save variance as numpy array
    np.save(os.path.join(save_path, "variance.npy"), variance_image)
    
    print(f"Mean image saved as 'mean.png' in {save_path}")
    print(f"Variance image saved as 'variance.npy' in {save_path}")
    
    return mean_image, variance_image

def main():
    # Configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str, default='configs/data_config.yaml')
    
    args = parser.parse_args()
    
    # Load configurations
    data_config = load_yaml(args.data_config)
    
    # Prepare dataset
    data_config = data_config['data']
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = get_dataset(**data_config, transforms=transform)
    
    # Get image size from the first image in the dataset
    print(len(dataset))
    sample_image = dataset[0]
    image_size = sample_image.shape[-1]  # Assuming square images
    
    # Compute and save mean image
    compute_and_save_mean_variance(dataset, image_size, data_config['root'])

if __name__ == '__main__':
    main()
