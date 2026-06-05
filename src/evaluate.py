import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.metrics import compute_metrics
from src.logging_setup import get_logger

log = get_logger(__name__)


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device, predict_fn, gpu_transform):
    model.eval()
    predictions, targets = [], []

    for images, batch_targets in tqdm(loader, desc="evaluate"):
        logits = model(gpu_transform(images.to(device)))
        predictions.append(predict_fn(logits).cpu())
        targets.append(batch_targets)

    return torch.cat(predictions).numpy(), torch.cat(targets).numpy()


def evaluate_predictions(
    predictions,
    targets,
    class_values: list,
    output_dir: Path = Path("outputs"),
) -> dict:
    num_classes = len(class_values)
    labels = list(range(num_classes))

    metrics = compute_metrics(predictions, targets, class_values)

    for name, value in metrics.items():
        log.info("%-22s: %.4f", name, value)

    report = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=[str(value) for value in class_values],
        zero_division=0,
    )
    
    log.info(report)

    matrix = confusion_matrix(targets, predictions, labels=labels)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "confusion_matrix.png"

    side = max(6, num_classes * 0.4)
    figure, axes = plt.subplots(figsize=(side, side))
    display = ConfusionMatrixDisplay(matrix, display_labels=class_values)
    display.plot(ax=axes, colorbar=False, xticks_rotation="vertical")
    figure.tight_layout()
    figure.savefig(matrix_path, dpi=150)
    plt.close(figure)
    log.info("confusion matrix saved: %s", matrix_path)

    return metrics


def plot_training_curves(history: dict, output_dir: Path = Path("outputs")) -> Path:
    train_history = history["train"]
    val_history = history["val"]
    epochs = range(1, len(train_history) + 1)

    metric_names = list(train_history[0].keys())
    columns = min(3, len(metric_names))
    rows = math.ceil(len(metric_names) / columns)

    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows), squeeze=False)
    flat_axes = axes.flatten()

    for axis, name in zip(flat_axes, metric_names):
        axis.plot(epochs, [metrics[name] for metrics in train_history], marker="o", label="train")
        axis.plot(epochs, [metrics[name] for metrics in val_history], marker="o", label="val")
        axis.set_title(name)
        axis.set_xlabel("epoch")
        axis.legend()

    for axis in flat_axes[len(metric_names):]:
        axis.set_visible(False)

    figure.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_curves.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)

    return path
