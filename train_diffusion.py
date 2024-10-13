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

def generate_and_save_images(epoch, model, sampler, device, out_path, dataset_name, folder_name, sample_fn, mean_image):
    model.eval()
    images = []
    with torch.no_grad():
        x_start = torch.randn(10, 3, model.image_size, model.image_size).to(device)
        samples = sample_fn(x_start=x_start, measurement=None, record=False, save_root=out_path)
        
        # Add mean image to the generated samples
        samples = samples + mean_image.to(device)
        
        # Denormalize and convert to PIL images
        samples = (samples.clamp(-1, 1) + 1) / 2
        samples = (samples * 255).byte().cpu().numpy()
        
        for i in range(10):
            img = Image.fromarray(samples[i].transpose(1, 2, 0))
            images.append(img)

    # Create a 2x5 grid of images
    fig, axs = plt.subplots(2, 5, figsize=(25, 10))
    for i, img in enumerate(images):
        axs[i//5, i%5].imshow(img)
        axs[i//5, i%5].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.join(out_path, folder_name), exist_ok=True)
    os.makedirs(os.path.join(out_path, folder_name, dataset_name), exist_ok=True)
    plt.savefig(os.path.join(out_path, folder_name, dataset_name, f'epoch_{epoch+1}.png'))
    plt.close()
    model.train()

def train(model, loader, sampler, optimizer, epochs, device, batch_size, logger, out_path, sample_fn=None, sample_interval=10, save_interval=100, mean_image=None, dataset_name=None, folder_name='generated_images'):
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm.tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            # Move data to device
            x_start = batch.to(device)
            # Subtract mean image
            x_start = x_start - mean_image.to(device)
            print(x_start.shape)
            # Generate random timesteps
            t = torch.randint(0, sampler.num_timesteps, (x_start.shape[0],), device=device).long()
            
            # Compute loss using the sampler's training_losses method
            loss_dict = sampler.training_losses(model, x_start, t)
            loss = loss_dict['loss'].mean()
            
            # Backpropagate and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Log epoch results
        avg_loss = epoch_loss / len(loader)
        logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        # Generate and save images
        if sample_fn is not None and (epoch + 1) % sample_interval == 0:
            generate_and_save_images(epoch, model, sampler, device, out_path, dataset_name, folder_name, sample_fn, mean_image)
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(out_path, f'checkpoint_epoch_{dataset_name}_{epoch+1}.pth'))

    # Move model back to CPU to free up GPU memory
    model.to('cpu')
    torch.cuda.empty_cache()

def main():
    # Configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default='configs/model_training_config.yaml')
    parser.add_argument('--diffusion_config', type=str, default='configs/train_diffusion_config.yaml')
    parser.add_argument('--data_config', type=str, default='configs/data_config.yaml')
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
    model = create_model_for_train(**model_config)
    model = model.to(device)
    
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=None)
    
    # Prepare dataloader
    data_config = data_config['data']
    transform = transforms.Compose([
        transforms.Resize((model_config['image_size'], model_config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    batch_size = 60
    sample_interval = 100
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=batch_size, num_workers=4, train=True)

    # Load mean image
    mean_image_path = os.path.join(data_config['root'], 'mean.png')
    mean_image = Image.open(mean_image_path)
    mean_image = transforms.Compose([
        transforms.Resize((model_config['image_size'], model_config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])(mean_image)
    mean_image = mean_image.to(device)

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    num_epochs = 10000  # Adjust as needed
    save_interval = 500
    folder_name = 'images_with_mean'
    dataset_name = data_config['name']
    train(model, loader, sampler, optimizer, num_epochs, device, batch_size, 
          logger, args.save_dir, sample_fn, sample_interval, save_interval, mean_image, dataset_name, folder_name)

    logger.info("Training completed.")

if __name__ == '__main__':
    main()
