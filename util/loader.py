import os
import random

import torch
import torchvision.transforms as transforms
import yaml
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur

from data.dataloader import get_dataloader, get_dataset


def data_transformer_list(mean=None, variance=None, size_l=None, size_w=None, if_grayscale=False):
    transform_list = [
        transforms.Resize((size_l, size_w)),
        transforms.ToTensor(),
    ]
    if if_grayscale:
        transform_list.append(transforms.Grayscale())  # Convert to grayscale first
        
    if mean is not None and variance is not None:
        transform_list.append(transforms.Normalize(mean, variance))  # Normalize if mean/var provided
    else:
        transform_list.append(transforms.Lambda(lambda x: 2.0 * x - 1.0))  # Scale to [-1,1]
        
    return transforms.Compose(transform_list)






def data_transformer_list_augmentation(mean, variance, size_l, size_w, if_grayscale=False):
    base_transforms = [
        transforms.Resize((size_l, size_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean, variance),
    ]

    if if_grayscale:
        base_transforms.insert(0, transforms.Grayscale())  # Convert to grayscale

    augmentations = [
        transforms.Compose(base_transforms),  # Original
        transforms.Compose(base_transforms + [transforms.RandomRotation((90, 90))]),  # 90 degrees
        transforms.Compose(base_transforms + [transforms.RandomRotation((180, 180))]),  # 180 degrees
        transforms.Compose(base_transforms + [transforms.RandomRotation((270, 270))]),  # 270 degrees
        transforms.Compose(base_transforms + [transforms.RandomHorizontalFlip(p=1.0)]),  # Horizontal flip
        transforms.Compose(base_transforms + [transforms.RandomRotation((90, 90)), transforms.RandomHorizontalFlip(p=1.0)]),  # 90 degrees + flip
        transforms.Compose(base_transforms + [transforms.RandomRotation((180, 180)), transforms.RandomHorizontalFlip(p=1.0)]),  # 180 degrees + flip
        transforms.Compose(base_transforms + [transforms.RandomRotation((270, 270)), transforms.RandomHorizontalFlip(p=1.0)]),  # 270 degrees + flip
    ]

    # Return a callable that randomly selects one of the augmentations
    def random_augmentation(img):
        transform = random.choice(augmentations)
        return transform(img)

    return random_augmentation


    # Prepare dataloader
def prepare_dataloader(data_config, model_config, if_train=False, train_config=None, if_normalize=False):
    if if_normalize:
        mean_image_path = os.path.join(data_config['train_root'], "mean_and_std", 'mean.pth')
        variance_path = os.path.join(data_config['train_root'], "mean_and_std", 'variance.pth')
            # Load mean image as grayscale
        mean_tensor = torch.load(mean_image_path)

        # Load variance and convert to std
        variance_tensor = torch.load(variance_path)
        std_tensor = torch.sqrt(variance_tensor)

        # Check if dimensions match
        if mean_tensor.shape[-1] != model_config['image_size'] or mean_tensor.shape[-2] != model_config['image_size']:
            raise ValueError(f"Mean image dimensions {mean_tensor.shape[-2:]} do not match model image size {model_config['image_size']}")

        if std_tensor.shape[-1] != model_config['image_size'] or std_tensor.shape[-2] != model_config['image_size']:
            raise ValueError(f"Std image dimensions {std_tensor.shape[-2:]} do not match model image size {model_config['image_size']}")
    else:
        mean_image_path = None
        variance_path = None

    if_grayscale = model_config['grayscale']


    if if_normalize:
        transform = data_transformer_list(mean=mean_tensor, variance=std_tensor,
                                        size_l=model_config['image_size'],
                                        size_w=model_config['image_size'],
                                        if_grayscale=if_grayscale)
    else:
        transform = data_transformer_list(size_l=model_config['image_size'],
                                        size_w=model_config['image_size'],
                                        if_grayscale=if_grayscale)
    if if_train:
        dataset = get_dataset(name=data_config['name'], root=data_config['train_root'], transforms=transform)
    else:
        dataset = get_dataset(name=data_config['name'], root=data_config['test_root'], transforms=transform)
    if train_config is None:
        loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=if_train)
    else:
        loader = get_dataloader(dataset, batch_size=train_config['batch_size'], num_workers=train_config['num_workers'], train=if_train)
    if if_normalize:
        return loader, mean_tensor, std_tensor
    else:
        return loader, None, None


def denormalize(img, mean_image, std_image, device):
    if mean_image is not None and std_image is not None:
        outcome = img * std_image.to(device) + mean_image.to(device)
    else:
        outcome = (img + 1) / 2
    outcome = torch.clamp(outcome, 0, 1)
    return outcome


def denormalize_steps(img, mean_image, std_image, device):
    if mean_image is not None and std_image is not None:
        std_outcome = img * std_image.to(device)
        mean_outcome = std_outcome + mean_image.to(device)
        outcome = torch.clamp(mean_outcome, 0, 1)
    else:
        outcome = (img + 1) / 2
        outcome = torch.clamp(outcome, 0, 1)
    return std_outcome, mean_outcome, outcome


def apply_gaussian_blur(mask, kernel_size=5, sigma=2.0, threshold_of_blur=0.5):
    """
    对mask应用高斯模糊，扩散黑色（0）区域
    Args:
        mask: 输入mask, 形状为 [1,C,H,W]，其中 0 表示mask区域
        kernel_size: 高斯核大小（奇数）
        sigma: 高斯核标准差
        threshold_of_blur: 模糊后的二值化阈值
    Returns:
        模糊后的mask, 形状为 [1,C,H,W]
    """
    # 确保 kernel_size 是奇数
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    # 创建高斯模糊层
    blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    
    # 先反转mask（把0变成1，1变成0）
    mask = 1 - mask
    
    # 应用高斯模糊
    mask = blur(mask)
    
    # 重新二值化（小于阈值的变成0，大于阈值的变成1）
    mask = (mask < threshold_of_blur).float()
    
    return mask

# Create circle mask from final kernel parameters
def create_circle_mask(sample, H, W, device):
    circle_mask = torch.ones((1, 1, H, W), device=device)
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
    x_coord = sample['kernel'][:, 1].view(1, 1)
    y_coord = sample['kernel'][:, 2].view(1, 1)
    radius = sample['kernel'][:, 0].view(1, 1)
    dist = torch.sqrt((x[None, :, :] - x_coord)**2 + (y[None, :, :] - y_coord)**2)
    circle_mask[:, 0] = (dist >= radius).float()
    return circle_mask

def create_mask(sample, ref_img, threshold=0.5, device='cuda'):
    sample = sample.to(device)
    ref_img = ref_img.to(device)
    
    # Determine number of channels
    if sample.dim() == 3:  # shape: [C, H, W]
        channels = sample.size(0)
    elif sample.dim() >= 4:  # shape: [N, C, H, W]
        channels = sample.size(1)
    else:
        raise ValueError("Unsupported tensor dimensions: {}".format(sample.dim()))
    
    # Compute the difference between sample and reference image
    diff = sample - ref_img
    
    # For grayscale images (channel==1), use absolute difference.
    # For colored images (channel==3), compute the L2 norm along the channel dimension.
    if channels == 1:
        mask = torch.abs(diff)
    elif channels == 3:
        if sample.dim() == 3:  # [3, H, W]
            mask = torch.norm(diff, p=2, dim=0, keepdim=True)  # resulting shape: [1, H, W]
        else:  # [N, 3, H, W] or similar
            mask = torch.norm(diff, p=2, dim=1, keepdim=True)  # resulting shape: [N, 1, H, W]
    else:
        raise ValueError("Unexpected number of channels: {}. Expected 1 or 3.".format(channels))
    
    # Optionally, you can threshold the mask to obtain a binary mask:
    # mask = (mask > threshold).float()
    
    return mask

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
