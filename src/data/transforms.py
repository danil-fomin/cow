import torch
from torchvision.transforms import v2

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


def build_cpu_transform(img_size: int) -> v2.Compose:
    return v2.Compose(
        [
            v2.Resize((img_size, img_size), antialias=True),
            v2.PILToTensor(),
        ]
    )


def build_gpu_transform(img_size: int, train: bool) -> v2.Compose:
    if train:
        return v2.Compose(
            [
                v2.RandomResizedCrop(img_size, scale=(0.7, 1.0), antialias=True),
                v2.RandomHorizontalFlip(),
                v2.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD),
                v2.RandomErasing(p=0.25),
            ]
        )

    return v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD),
        ]
    )
