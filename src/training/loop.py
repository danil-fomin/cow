import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import compute_metrics
from src.utils import all_gather_concat, all_reduce_sum, is_main_process


def _summarize(
    total_loss: float,
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    class_values: list,
) -> dict:
    predictions = all_gather_concat(torch.cat(predictions))
    targets = all_gather_concat(torch.cat(targets))
    total_loss = all_reduce_sum(total_loss)
    num_samples = len(targets)

    return {"loss": total_loss / num_samples, **compute_metrics(predictions, targets, class_values)}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_values: list,
    epoch: int,
    predict_fn,
) -> dict:
    model.train()
    total_loss = 0.0
    predictions, targets = [], []

    progress = tqdm(loader, desc=f"train [{epoch}]", disable=not is_main_process())
    for images, batch_targets in progress:
        images = images.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, batch_targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions.append(predict_fn(logits).cpu())
        targets.append(batch_targets.cpu())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return _summarize(total_loss, predictions, targets, class_values)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    class_values: list,
    epoch: int,
    predict_fn,
) -> dict:
    model.eval()
    total_loss = 0.0
    predictions, targets = [], []

    progress = tqdm(loader, desc=f"val   [{epoch}]", disable=not is_main_process())
    for images, batch_targets in progress:
        images = images.to(device)
        batch_targets = batch_targets.to(device)

        logits = model(images)
        loss = loss_fn(logits, batch_targets)

        total_loss += loss.item() * images.size(0)
        predictions.append(predict_fn(logits).cpu())
        targets.append(batch_targets.cpu())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return _summarize(total_loss, predictions, targets, class_values)
