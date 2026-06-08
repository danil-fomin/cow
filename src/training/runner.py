from pathlib import Path

from ultralytics import YOLO

from src.evaluate import evaluate_predictions
from src.predict import gather_images
from src.logging_setup import get_logger

log = get_logger(__name__)


def _weights(config: dict) -> str:
    if config.get("weights"):
        return config["weights"]
    project = config.get("project", "runs/detect")
    name = config.get("name", "bcs")
    return str(Path(project) / name / "weights" / "best.pt")


def _best_box_class(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    best = int(boxes.conf.argmax())
    return int(boxes.cls[best])


def run_evaluate(config: dict):
    model = YOLO(_weights(config))
    data_dir = Path(config["data_dir"])
    class_values = sorted(config["class_values"])  

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


def run_predict(config: dict, input_path: str):
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
