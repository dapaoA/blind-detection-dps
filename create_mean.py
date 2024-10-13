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

def compute_and_save_mean_image(dataset, image_size, save_path):
    mean_image = np.zeros((3, image_size, image_size), dtype=np.float32)
    
    for i in tqdm.tqdm(range(len(dataset)), desc="Computing mean image"):
        img = dataset[i]
        mean_image += img.numpy()
    
    mean_image /= len(dataset)
    
    # Convert to PIL Image and save
    mean_image = (mean_image * 255).astype(np.uint8)
    mean_image = np.transpose(mean_image, (1, 2, 0))
    mean_image = Image.fromarray(mean_image)
    mean_image.save(os.path.join(save_path, "mean.png"))
    
    print(f"Mean image saved as 'mean.png' in {save_path}")

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
    compute_and_save_mean_image(dataset, image_size, data_config['root'])

if __name__ == '__main__':
    main()
