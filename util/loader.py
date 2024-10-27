import torchvision.transforms as transforms
import yaml
import random


def data_transformer_list(mean, variance, size_l, size_w, if_grayscale=False):
    transform_list = [
        transforms.Resize((size_l, size_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean, variance)
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


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config