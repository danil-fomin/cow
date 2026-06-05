import torch
import torch.nn.functional as F
from torch import nn


def corn_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    total_loss = 0.0
    num_examples = 0

    for task_index in range(num_classes - 1):
        conditional_mask = targets > task_index - 1
        binary_targets = (targets[conditional_mask] > task_index).float()
        if binary_targets.numel() == 0:
            continue

        predictions = logits[conditional_mask, task_index]
        total_loss += -torch.sum(
            F.logsigmoid(predictions) * binary_targets
            + (F.logsigmoid(predictions) - predictions) * (1.0 - binary_targets)
        )
        num_examples += binary_targets.numel()

    return total_loss / num_examples


class CornLoss(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        return corn_loss(logits, targets, self.num_classes)


def build_loss(config: dict) -> nn.Module:
    if config["head"] == "corn":
        return CornLoss(len(config["class_values"]))

    return nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
