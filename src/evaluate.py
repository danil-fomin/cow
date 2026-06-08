from pathlib import Path

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
