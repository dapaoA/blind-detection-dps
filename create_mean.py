import argparse
import os

import numpy as np
import torch
import torchvision.transforms as transforms
import tqdm
import yaml
from PIL import Image

from data.dataloader import get_dataset


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def compute_and_save_mean_variance(dataset, target_size, save_path, if_grayscale=False):
    """计算并保存数据集的均值和方差

    Args:
        dataset: 数据集
        target_size: 目标图像大小
        save_path: 保存路径
        if_grayscale: 是否转换为灰度图
    """
    os.makedirs(os.path.join(save_path, "mean_and_std"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建调整大小的转换
    resize_transform = transforms.Resize((target_size, target_size))

    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=0
    )

    # 初始化均值和方差张量
    channels = 1 if if_grayscale else 3
    mean_tensor = torch.zeros((channels, target_size, target_size), device=device)
    var_tensor = torch.zeros((channels, target_size, target_size), device=device)

    # 计算均值
    for batch_dict in tqdm.tqdm(dataloader, desc="Computing mean"):
        # 直接获取batch中的image
        batch = batch_dict['image'].to(device)  # 已经是batch形式
        batch = resize_transform(batch)  # 调整大小
        if if_grayscale:
            batch = 0.2989 * batch[:, 0:1] + 0.5870 * batch[:, 1:2] + 0.1140 * batch[:, 2:3]
        mean_tensor += batch.sum(dim=0)

    mean_tensor /= len(dataset)

    # 计算方差
    for batch_dict in tqdm.tqdm(dataloader, desc="Computing variance"):
        batch = batch_dict['image'].to(device)  # 已经是batch形式
        batch = resize_transform(batch)  # 调整大小
        if if_grayscale:
            batch = 0.2989 * batch[:, 0:1] + 0.5870 * batch[:, 1:2] + 0.1140 * batch[:, 2:3]
        var_tensor += (batch - mean_tensor).pow(2).sum(dim=0)

    var_tensor /= len(dataset)
    var_tensor[var_tensor <= 1e-6] = 1

    # 保存为 .pth 文件
    torch.save(mean_tensor.cpu(), os.path.join(save_path, "mean_and_std", "mean.pth"))
    torch.save(var_tensor.cpu(), os.path.join(save_path, "mean_and_std", "variance.pth"))

    # 保存预览图片
    save_preview_images(mean_tensor, var_tensor, save_path, if_grayscale)

    print(f"Files saved in {os.path.join(save_path, 'mean_and_std')}:")
    print("- mean.pth (tensor data in [0,1] range)")
    print("- variance.pth (tensor data)")
    print("- mean_preview.jpg (preview image)")
    print("- variance_preview.jpg (preview image)")

    return mean_tensor.cpu(), var_tensor.cpu()

def save_preview_images(mean_tensor, var_tensor, save_path, if_grayscale):
    """保存预览图片"""
    # 保存均值预览
    if if_grayscale:
        preview = (mean_tensor[0].cpu().numpy() * 255).astype(np.uint8)
        preview_img = Image.fromarray(preview, mode='L')
    else:
        preview = (mean_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        preview_img = Image.fromarray(preview, mode='RGB')
    preview_img.save(os.path.join(save_path, "mean_and_std", "mean_preview.jpg"), quality=95)

    # 保存方差预览 - 先归一化再转uint8
    if if_grayscale:
        var_data = var_tensor[0].cpu().numpy()
        var_normalized = (var_data - var_data.min()) / (var_data.max() - var_data.min())
        var_preview = (var_normalized * 255).astype(np.uint8)
        var_preview_img = Image.fromarray(var_preview, mode='L')
    else:
        var_data = var_tensor.cpu().numpy().transpose(1, 2, 0)
        var_normalized = (var_data - var_data.min()) / (var_data.max() - var_data.min())
        var_preview = (var_normalized * 255).astype(np.uint8)
        var_preview_img = Image.fromarray(var_preview, mode='RGB')
    var_preview_img.save(os.path.join(save_path, "mean_and_std", "variance_preview.jpg"), quality=95)

def main():
    # Configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str, default='configs/data_config.yaml')
    parser.add_argument('--img_model_config', type=str, default='configs/model_config.yaml')
    args = parser.parse_args()
    model_config = load_yaml(args.img_model_config)

    # Load configurations
    data_config = load_yaml(args.data_config)

    # Prepare dataset
    data_config = data_config['data']
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = get_dataset(root=data_config['train_root'], **data_config, transforms=transform)

    # Get image size from the first image in the dataset
    print(len(dataset))
    sample_dict = dataset[0]
    sample_image = sample_dict['image']  # 获取字典中的图像张量
    image_size = sample_image.shape[-1]  # 现在可以获取图像大小了

    # Compute and save mean image
    compute_and_save_mean_variance(dataset, model_config['image_size'], data_config['train_root'], model_config['grayscale'])

if __name__ == '__main__':
    main()
