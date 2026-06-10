from pathlib import Path

import torch
from ultralytics import YOLO

from src.logging_setup import get_logger

log = get_logger(__name__)


def device_arg(num_gpus: int):
    if not torch.cuda.is_available() or num_gpus < 1:
        return "cpu"
    if num_gpus == 1:
        return 0
    return ",".join(str(i) for i in range(num_gpus))


def build_train_kwargs(config: dict) -> dict:
    project = Path(config.get("project", "runs/detect")).resolve()
    return {
        "data": str((Path(config["data_dir"]) / "data.yaml").resolve()),
        "epochs": config["epochs"],
        "imgsz": config["img_size"],
        "batch": config["batch_size"],
        "lr0": float(config["learn_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "patience": config["early_stopping_patience"],
        "device": device_arg(config["num_gpus"]),
        "seed": config.get("seed", 42),
        "project": str(project),
        "name": config.get("name", "bcs"),
        "exist_ok": True,
    }


def run_train(config: dict, on_model_save=None):
    kwargs = build_train_kwargs(config)
    last = Path(kwargs["project"]) / kwargs["name"] / "weights" / "last.pt"

    if last.exists():
        log.info("resuming from %s", last)
        model = YOLO(str(last))
        train_kwargs = {"resume": True}
    else:
        model = YOLO(config["model"])
        train_kwargs = kwargs

    if on_model_save is not None:
        model.add_callback("on_model_save", on_model_save)

    results = model.train(**train_kwargs)
    log.info("training complete; best weights: %s", model.trainer.best)
    return results
