from pathlib import Path

from src.data.transforms import build_eval_transforms
from src.models.model import build_model, build_predictor
from src.evaluate import collect_predictions, evaluate_predictions
from src.predict import predict_image, predict_folder
from src.training.train import _build_loader, BEST_CHECKPOINT_PATH
from src.logging_setup import get_logger
from src.utils import get_device, load_checkpoint

log = get_logger(__name__)


def run_evaluate(config: dict):
    device = get_device()
    image_size = config["img_size"]

    test_loader = _build_loader(
        "test", build_eval_transforms(image_size), config, distributed=False, train=False
    )

    model = build_model(config).to(device)
    load_checkpoint(model, BEST_CHECKPOINT_PATH, device)

    predict_fn = build_predictor(config)
    predictions, targets = collect_predictions(model, test_loader, device, predict_fn)
    evaluate_predictions(predictions, targets, config["class_values"])


def run_predict(config: dict, input_path: str):
    device = get_device()
    transforms = build_eval_transforms(config["img_size"])
    class_values = config["class_values"]

    model = build_model(config).to(device)
    load_checkpoint(model, BEST_CHECKPOINT_PATH, device)
    predict_fn = build_predictor(config)

    input_path = Path(input_path)
    if input_path.is_dir():
        results = predict_folder(model, input_path, transforms, class_values, device, predict_fn)
    else:
        index, value = predict_image(model, input_path, transforms, class_values, device, predict_fn)
        results = [(input_path, index, value)]

    for path, index, value in results:
        log.info("%s: class=%s value=%s", path, index, value)
