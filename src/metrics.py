import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)

ADJACENT_THRESHOLDS = (1, 2)
_FP_SLACK = 1e-9


def _to_numpy(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()

    return np.asarray(values)


def compute_metrics(predictions, targets, class_values) -> dict:
    predictions = _to_numpy(predictions)
    targets = _to_numpy(targets)
    labels = list(range(len(class_values)))
    values = np.asarray(class_values, dtype=float)

    absolute_error = np.abs(values[predictions] - values[targets])

    metrics = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, labels=labels, average="macro", zero_division=0)),
        "recall": float(recall_score(targets, predictions, labels=labels, average="macro", zero_division=0)),
        "f1": float(f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)),
        "mae": float(absolute_error.mean()),
        "qwk": float(cohen_kappa_score(targets, predictions, weights="quadratic", labels=labels)),
    }
    for threshold in ADJACENT_THRESHOLDS:
        metrics[f"adjacent_accuracy@{threshold}"] = float((absolute_error <= threshold + _FP_SLACK).mean())

    return metrics
