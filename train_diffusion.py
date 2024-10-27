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
from guided_diffusion.unet import create_model_for_train
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.logger import get_logger
from torch.utils.tensorboard import SummaryWriter
from util.loader import data_transformer_list, load_yaml



def generate_and_save_images(epoch, model, sampler, device, out_path,
                             dataset_name, folder_name, sample_fn,
                             mean_image, std_image, if_grayscale, writer):
    model.eval()
    images = []
    with torch.no_grad():
        x_start = torch.randn(10, 1 if if_grayscale else 3, model.image_size, model.image_size).to(device)
        samples = sample_fn(x_start=x_start, measurement=None, record=False, save_root=out_path)
        
        # Denormalize the generated samples
        samples = samples * std_image.to(device) + mean_image.to(device)
        
        # Clip values to [0, 1] range
        samples = torch.clamp(samples, 0, 1)
        samples = (samples * 255).byte().cpu().numpy()
        
        for i in range(10):
            if if_grayscale:  # Check if the channel is 1 for grayscale
                img = Image.fromarray(samples[i][0], mode='L')  # Use mode 'L' for grayscale
            else:
                img = Image.fromarray(samples[i].transpose(1, 2, 0))
            images.append(img)

    # Create a 2x5 grid of images
    fig, axs = plt.subplots(2, 5, figsize=(25, 10))
    for i, img in enumerate(images):
        if if_grayscale:  # Check if the channel is 1 for grayscale
            axs[i//5, i%5].imshow(img, cmap='gray')  # Use cmap 'gray' for grayscale
        else:
            axs[i//5, i%5].imshow(img)
        axs[i//5, i%5].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, folder_name, dataset_name, "images", f'epoch_{epoch+1}.png'))
    
    # Add the figure to TensorBoard
    writer.add_figure('Generated Images', fig, epoch)
    
    plt.close()
    model.train()

def train(model, loader, sampler, optimizer, epochs,
          device, batch_size, logger, out_path,
          sample_fn=None, sample_interval=10, save_interval=100,
          mean_image=None, std_image=None, dataset_name=None,
          folder_name='generated_images', if_grayscale=False, writer=None):
    # Create directories if they don't exist
    os.makedirs(os.path.join(out_path, folder_name, dataset_name, "models"), exist_ok=True)
    os.makedirs(os.path.join(out_path, folder_name, dataset_name, "images"), exist_ok=True)
    model.train()
    best_loss = float('inf')
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm.tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            # Move data to device
            x_start = batch.to(device)

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
            
            # Add loss to TensorBoard
            global_step = epoch * len(loader) + i
            writer.add_scalar('Loss/train', loss.item(), global_step)
        
        # Log epoch results
        avg_loss = epoch_loss / len(loader)
        logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        # Add average loss to TensorBoard
        writer.add_scalar('Loss/epoch', avg_loss, epoch)
        
        # Generate and save images
        if sample_fn is not None and (epoch + 1) % sample_interval == 0:
            generate_and_save_images(epoch, model, sampler, device,
                                     out_path, dataset_name, folder_name,
                                     sample_fn, mean_image, std_image, if_grayscale, writer)
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(out_path, folder_name, dataset_name, "models", f'checkpoint_epoch_{dataset_name}_{epoch+1}.pth'))

        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(out_path, folder_name, dataset_name, "models", f'best_model_{dataset_name}.pth'))
            logger.info(f"New best model saved with loss: {best_loss:.4f}")

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
    # Load mean and variance
    data_config = data_config['data']
    mean_image_path = os.path.join(data_config['root'], 'mean_and_std', 'mean.png')
    variance_path = os.path.join(data_config['root'], 'mean_and_std', 'variance.npy')
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

    # Prepare dataloader
    transform = data_transformer_list(mean_image, std_image, 
                                      model_config['image_size'],
                                      model_config['image_size'],
                                      if_grayscale=if_grayscale)
    
    batch_size = 8
    sample_interval = 50
    num_epochs = 10000  # Adjust as needed
    save_interval = 500

    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=batch_size, num_workers=8, train=True)

    # Move mean and std to device
    mean_image = mean_image.to(device)
    std_image = std_image.to(device)
    # print(std_image)
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Set up TensorBoard

    # Train the model
    folder_name = 'generated_images'
    dataset_name = data_config['name']
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, folder_name), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, folder_name, dataset_name), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, folder_name, dataset_name, 'tensorboard_logs'))
    # tensorboard --logdir=D:\experiments\su\blind-detection-dps\results\generated_images\tensorboard_logs
    train(model, loader, sampler, optimizer, num_epochs, device, batch_size, 
          logger, args.save_dir, sample_fn, sample_interval,
          save_interval, mean_image, std_image, dataset_name,
          folder_name, if_grayscale, writer)

    writer.close()
    logger.info("Training completed.")

if __name__ == '__main__':
    main()
