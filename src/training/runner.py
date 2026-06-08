from pathlib import Path

from ultralytics import YOLO

from src.data.labels import class_index_to_value, resolve_value
from src.evaluate import evaluate_predictions
from src.predict import gather_images, predict_paths
from src.training.train import default_project
from src.logging_setup import get_logger

log = get_logger(__name__)


def _weights(config: dict) -> str:
    if config.get("weights"):
        return config["weights"]
    name = config.get("name", "bcs")
    return str(Path(default_project(config)) / name / "weights" / "best.pt")


def _best_box_class(result):
    """Class index of the highest-confidence box, or None if there is no box."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    best = int(boxes.conf.argmax())
    return int(boxes.cls[best])


# ---------------------------------------------------------------- classification


def _evaluate_classify(config: dict):
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


def _predict_classify(config: dict, input_path: str):
    model = YOLO(_weights(config))
    class_values = sorted(config["class_values"])
    index_to_value = class_index_to_value(model.names, class_values)

    paths = gather_images(input_path)
    for path, index, value in predict_paths(model, paths, index_to_value):
        log.info("%s: class=%s value=%s", path, index, value)


# --------------------------------------------------------------------- detection


def _evaluate_detect(config: dict):
    model = YOLO(_weights(config))
    data_dir = Path(config["det_data_dir"])
    class_values = sorted(config["class_values"])  # index order matches data.yaml names

    test_image_dir = data_dir / "images" / "test"
    test_label_dir = data_dir / "labels" / "test"

    pred_ranks, target_ranks, missing = [], [], 0
    for image_path in sorted(test_image_dir.iterdir()):
        label_path = test_label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        tokens = label_path.read_text().split()
        if not tokens:
            continue
        target_index = int(tokens[0])

        result = model.predict(str(image_path), conf=0.001, verbose=False)[0]
        predicted_index = _best_box_class(result)
        if predicted_index is None:
            missing += 1
            log.warning("no detection for %s; excluded from metrics", image_path.name)
            continue

        target_ranks.append(target_index)
        pred_ranks.append(predicted_index)

    if missing:
        log.warning("%d test image(s) had no detection and were excluded", missing)

    evaluate_predictions(pred_ranks, target_ranks, class_values)


def _predict_detect(config: dict, input_path: str):
    model = YOLO(_weights(config))
    class_values = sorted(config["class_values"])

    for image_path in gather_images(input_path):
        result = model.predict(str(image_path), verbose=False)[0]
        index = _best_box_class(result)
        if index is None:
            log.info("%s: no cow detected", image_path)
            continue
        boxes = result.boxes
        best = int(boxes.conf.argmax())
        box = [round(c, 1) for c in boxes.xyxy[best].tolist()]
        log.info("%s: class=%s value=%s box=%s", image_path, index, class_values[index], box)


# ------------------------------------------------------------------- dispatchers


def run_evaluate(config: dict):
    if config.get("task") == "detect":
        return _evaluate_detect(config)
    return _evaluate_classify(config)


def run_predict(config: dict, input_path: str):
    if config.get("task") == "detect":
        return _predict_detect(config, input_path)
    return _predict_classify(config, input_path)
