from pathlib import Path

from ultralytics import YOLO

from src.data.labels import class_index_to_value, resolve_value
from src.evaluate import evaluate_predictions
from src.predict import gather_images, predict_paths
from src.logging_setup import get_logger

log = get_logger(__name__)


def _weights(config: dict) -> str:
    if config.get("weights"):
        return config["weights"]
    project = config.get("project", "runs/classify")
    name = config.get("name", "train")
    return str(Path(project) / name / "weights" / "best.pt")


def run_evaluate(config: dict):
    model = YOLO(_weights(config))
    data_dir = Path(config["data_dir"])
    class_values = sorted(config["class_values"])

    index_to_value = class_index_to_value(model.names, class_values)
    value_to_rank = {value: rank for rank, value in enumerate(class_values)}

    test_dir = data_dir / "test"
    image_paths, target_ranks = [], []
    for class_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        value = resolve_value(class_dir.name, class_values)
        for image_path in sorted(class_dir.iterdir()):
            image_paths.append(image_path)
            target_ranks.append(value_to_rank[value])

    predictions = model.predict([str(p) for p in image_paths], verbose=False)
    pred_ranks = [value_to_rank[index_to_value[int(r.probs.top1)]] for r in predictions]

    evaluate_predictions(pred_ranks, target_ranks, class_values)


def run_predict(config: dict, input_path: str):
    model = YOLO(_weights(config))
    class_values = sorted(config["class_values"])
    index_to_value = class_index_to_value(model.names, class_values)

    paths = gather_images(input_path)
    for path, index, value in predict_paths(model, paths, index_to_value):
        log.info("%s: class=%s value=%s", path, index, value)
