from torchvision import transforms 

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

def build_train_transforms(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD),
            transforms.RandomErasing(p=0.25),
        ]
    )


def build_eval_transforms(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD),
        ]
    )
