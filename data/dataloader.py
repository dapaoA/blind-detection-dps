from glob import glob
from typing import Callable, Optional

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import os
from torchvision import transforms
__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: VisionDataset,
                   batch_size: int,
                   num_workers: int,
                   train: bool):
    dataloader = DataLoader(dataset,
                            batch_size,
                            shuffle=train,
                            num_workers=num_workers,
                            drop_last=train)
    return dataloader


@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        return img

@register_dataset(name='imagenet1k')
class ImageNet1kDataset(FFHQDataset):
    def __init__(self, root: str, transforms: Optional[Callable] = None):
        super().__init__(root, transforms)

        self.fpaths = self.fpaths[:1000]  # only takes the first 1k images

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index: int):
        return super().__getitem__(index)

@register_dataset(name='afhq')
class AFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable] = None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.jpg', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)
        return img

@register_dataset(name='BraTS')
class BraTSDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.h5', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        with h5py.File(fpath, 'r') as hf:
            print("hf image: ", hf['image'] )
            img = np.array(hf['image'][:])

        # Save each channel as a separate black and white PNG
        for i in range(4):
            print(img[:, :, i])
            channel = img[:, :, i].astype(np.float64)
            print("channel: ", channel)
            print("channel max: ", channel.max())
            channel_img = Image.fromarray(channel, mode='L')
            save_path = fpath.replace('.h5', f'_channel_{i+1}.png')
            channel_img.save(save_path, format='PNG')
            print(f"Saved {save_path}")

        # For the purpose of returning an image, we'll use the first channel
        img = Image.fromarray(img[:, :, 0].astype(np.uint8), mode='L')

        if self.transforms is not None:
            img = self.transforms(img)

        return img

@register_dataset(name='mvtec')
class MVTecDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None, mode='train'):
        super().__init__(root, transforms)
        self.mode = mode
        
        # 检查root路径是否为train目录
        root_dir = os.path.basename(os.path.normpath(root))
        if root_dir == 'train':
            # train模式：读取good目录
            data_path = os.path.join(root, 'good')
            self.data_info = []
            for img_name in sorted(os.listdir(data_path)):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.data_info.append({
                        'image_path': os.path.join(data_path, img_name),
                        'gt_path': None,
                        'category': 'good',
                        'name': img_name
                    })
        else:
            # test模式：读取所有测试数据
            self.test_path = root
            parent_dir = os.path.dirname(root)
            self.gt_path = os.path.join(parent_dir, 'ground_truth')
            
            self.data_info = []
            for category in os.listdir(self.test_path):
                category_path = os.path.join(self.test_path, category)
                if os.path.isdir(category_path):
                    for img_name in sorted(os.listdir(category_path)):
                        if img_name.endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(category_path, img_name)
                            gt_path = os.path.join(self.gt_path, category, img_name)
                            
                            self.data_info.append({
                                'image_path': img_path,
                                'gt_path': gt_path if os.path.exists(gt_path) else None,
                                'category': category,
                                'name': img_name
                            })
        
        assert len(self.data_info) > 0, f"File list is empty. Check the directory: {root}"

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index: int):
        info = self.data_info[index]
        
        # 读取图像
        img = Image.open(info['image_path']).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
            
        # 准备基础返回字典
        result_dict = {
            'image': img,
            'category': info['category'],
            'name': info['name'],
            'path': info['image_path']
        }
        
        # 如果存在ground truth，添加mask和gt_path
        if info['gt_path'] is not None:
            mask = Image.open(info['gt_path']).convert('L')
            mask = transforms.Resize(img.shape[-2:])(transforms.ToTensor()(mask))
            result_dict.update({
                'mask': mask,
                'gt_path': info['gt_path']
            })
        
        return result_dict

@register_dataset(name='bottle')
class BottleDataset(MVTecDataset):
    def __init__(self, root, transforms=None):
        super().__init__(root, transforms)

@register_dataset(name='wood')
class WoodDataset(MVTecDataset):
    def __init__(self, root, transforms=None):
        super().__init__(root, transforms)

@register_dataset(name='cable')
class CableDataset(MVTecDataset):
    def __init__(self, root, transforms=None):
        super().__init__(root, transforms)

@register_dataset(name='capsule')
class CapsuleDataset(MVTecDataset):
    def __init__(self, root, transforms=None):
        super().__init__(root, transforms)

@register_dataset(name='carpet')
class CarpetDataset(MVTecDataset):
    def __init__(self, root, transforms=None):
        super().__init__(root, transforms)

@register_dataset(name='grid')
class GridDataset(MVTecDataset):
    def __init__(self, root, transforms=None):
        super().__init__(root, transforms)

@register_dataset(name='hazelnut')
class HazelnutDataset(MVTecDataset):
    def __init__(self, root, transforms=None):
        super().__init__(root, transforms)

@register_dataset(name='leather')
class LeatherDataset(MVTecDataset):
    def __init__(self, root, transforms=None):
        super().__init__(root, transforms)

@register_dataset(name='metalnut')
class MetalNutDataset(MVTecDataset):
    def __init__(self, root, transforms=None):
        super().__init__(root, transforms)

@register_dataset(name='pill')
class PillDataset(MVTecDataset):
    def __init__(self, root, transforms=None):
        super().__init__(root, transforms)

@register_dataset(name='screw')
class ScrewDataset(MVTecDataset):
    def __init__(self, root, transforms=None):
        super().__init__(root, transforms)

@register_dataset(name='tile')
class TileDataset(MVTecDataset):
    def __init__(self, root, transforms=None):
        super().__init__(root, transforms)

@register_dataset(name='toothbrush')
class ToothbrushDataset(MVTecDataset):
    def __init__(self, root, transforms=None):
        super().__init__(root, transforms)

@register_dataset(name='transistor')
class TransistorDataset(MVTecDataset):
    def __init__(self, root, transforms=None):
        super().__init__(root, transforms)

@register_dataset(name='zipper')
class ZipperDataset(MVTecDataset):
    def __init__(self, root, transforms=None):
        super().__init__(root, transforms)
