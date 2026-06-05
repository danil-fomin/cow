from pathlib import Path

import torch
from torch import nn
from PIL import Image


@torch.no_grad()
def predict_image(
    model: nn.Module,
    image_path: Path,
    transforms,
    class_values: list,
    device: torch.device,
    predict_fn,
    gpu_transform,
):
    model.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = transforms(image).unsqueeze(0).to(device)
    tensor = gpu_transform(tensor)

    logits = model(tensor)
    index = predict_fn(logits).item()

    return index, class_values[index]


def predict_folder(
    model: nn.Module,
    folder: Path,
    transforms,
    class_values: list,
    device: torch.device,
    predict_fn,
    gpu_transform,
):
    results = []
    for image_path in sorted(Path(folder).glob("*.jpg")):
        index, value = predict_image(
            model, image_path, transforms, class_values, device, predict_fn, gpu_transform
        )
        results.append((image_path, index, value))

    return results
