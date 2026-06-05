import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import compute_metrics
from src.utils import all_gather_concat, all_reduce_sum, is_main_process


def _summarize(
    total_loss: torch.Tensor,
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    class_values: list,
) -> dict:
    predictions = all_gather_concat(torch.cat(predictions).cpu())
    targets = all_gather_concat(torch.cat(targets).cpu())
    total_loss = all_reduce_sum(float(total_loss))
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
    total_loss = torch.zeros((), device=device)
    predictions, targets = [], []

    progress = tqdm(loader, desc=f"train [{epoch}]", disable=not is_main_process())
    for batch_index, (images, batch_targets) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, batch_targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach() * images.size(0)
        predictions.append(predict_fn(logits).detach())
        targets.append(batch_targets)

        if is_main_process() and batch_index % 50 == 0:
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
    total_loss = torch.zeros((), device=device)
    predictions, targets = [], []

    progress = tqdm(loader, desc=f"val   [{epoch}]", disable=not is_main_process())
    for batch_index, (images, batch_targets) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)

        logits = model(images)
        loss = loss_fn(logits, batch_targets)

        total_loss += loss.detach() * images.size(0)
        predictions.append(predict_fn(logits).detach())
        targets.append(batch_targets)

        if is_main_process() and batch_index % 50 == 0:
            progress.set_postfix(loss=f"{loss.item():.4f}")

    return _summarize(total_loss, predictions, targets, class_values)
