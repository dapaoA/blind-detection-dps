import torchvision.transforms as transforms
import yaml

def data_transformer_list(mean, variance, size_l, size_w, if_grayscale=False):
    transform_list = [
        transforms.Resize((size_l, size_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean, variance)
    ]
    if if_grayscale:
        transform_list.append(transforms.Grayscale())  # Convert to grayscale
    return transforms.Compose(transform_list)

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config