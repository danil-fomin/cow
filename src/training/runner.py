from pathlib import Path

from src.data.transforms import build_cpu_transform, build_gpu_transform
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

    cpu_transform = build_cpu_transform(image_size)
    eval_gpu_transform = build_gpu_transform(image_size, train=False)
    test_loader = _build_loader(
        "test", cpu_transform, config, distributed=False, train=False
    )

    model = build_model(config).to(device)
    load_checkpoint(model, BEST_CHECKPOINT_PATH, device)

    predict_fn = build_predictor(config)
    predictions, targets = collect_predictions(model, test_loader, device, predict_fn, eval_gpu_transform)
    evaluate_predictions(predictions, targets, config["class_values"])


def run_predict(config: dict, input_path: str):
    device = get_device()
    cpu_transform = build_cpu_transform(config["img_size"])
    eval_gpu_transform = build_gpu_transform(config["img_size"], train=False)
    class_values = config["class_values"]

    model = build_model(config).to(device)
    load_checkpoint(model, BEST_CHECKPOINT_PATH, device)
    predict_fn = build_predictor(config)

    input_path = Path(input_path)
    if input_path.is_dir():
        results = predict_folder(
            model, input_path, cpu_transform, class_values, device, predict_fn, eval_gpu_transform
        )
    else:
        index, value = predict_image(
            model, input_path, cpu_transform, class_values, device, predict_fn, eval_gpu_transform
        )
        results = [(input_path, index, value)]

    for path, index, value in results:
        log.info("%s: class=%s value=%s", path, index, value)
