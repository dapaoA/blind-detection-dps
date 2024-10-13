from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import h5py
import numpy as np

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
    
@register_dataset(name='leather')
class LeatherDataset(VisionDataset):
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
    
@register_dataset(name='bottle')
class BottleDataset(VisionDataset):
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


@register_dataset(name='toothbrush')
class ToothbrushDataset(VisionDataset):
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

        exit()
        
        # For the purpose of returning an image, we'll use the first channel
        img = Image.fromarray(img[:, :, 0].astype(np.uint8), mode='L')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img
