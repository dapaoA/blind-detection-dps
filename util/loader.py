import os
import random

import torch
import torchvision.transforms as transforms
import yaml

from data.dataloader import get_dataloader, get_dataset


def data_transformer_list(mean, variance, size_l, size_w, if_grayscale=False):
    transform_list = [
        transforms.Resize((size_l, size_w)),
        transforms.ToTensor(),
    ]
    if if_grayscale:
        transform_list.append(transforms.Grayscale())  # 先转灰度
    transform_list.append(transforms.Normalize(mean, variance))  # 后归一化
    return transforms.Compose(transform_list)


def data_transformer_list_no_normalize(size_l, size_w, if_grayscale=False):
    transform_list = [
        transforms.Resize((size_l, size_w)),
        transforms.ToTensor(),
        ]
    if if_grayscale:
        transform_list.append(transforms.Grayscale())  # Convert to grayscale
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
def prepare_dataloader(data_config, model_config, if_train=False, task_config=None, train_config=None):
    mean_image_path = os.path.join(data_config['root'], "mean_and_std", 'mean.pth')
    variance_path = os.path.join(data_config['root'], "mean_and_std", 'variance.pth')
    if_grayscale = model_config['grayscale']

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

    transform = data_transformer_list(mean_tensor, std_tensor,
                                    model_config['image_size'],
                                    model_config['image_size'],
                                    if_grayscale=if_grayscale)
    if task_config is None:
        dataset = get_dataset(**data_config, transforms=transform)
    else:
        dataset = get_dataset(**task_config['data'], transforms=transform)
    if train_config is None:
        loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=if_train)
    else:
        loader = get_dataloader(dataset, batch_size=train_config['batch_size'], num_workers=train_config['num_workers'], train=if_train)
    return loader, mean_tensor, std_tensor


def denormalize(img, mean_image, std_image, device):
    outcome = img * std_image.to(device) + mean_image.to(device)
    outcome = torch.clamp(outcome, 0, 1)
    return outcome


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
    mask = torch.abs(sample.to(device) - ref_img.to(device))
    mask = (mask < threshold).float()  # Changed > to < to match instructions
    return mask

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
