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
    return {
        "data": str(Path(config["data_dir"]) / "data.yaml"),
        "epochs": config["epochs"],
        "imgsz": config["img_size"],
        "batch": config["batch_size"],
        "lr0": float(config["learn_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "patience": config["early_stopping_patience"],
        "device": device_arg(config["num_gpus"]),
        "seed": config.get("seed", 42),
        "project": config.get("project", "runs/detect"),
        "name": config.get("name", "bcs"),
    }


def run_train(config: dict):
    model = YOLO(config["model"])
    results = model.train(**build_train_kwargs(config))
    log.info("training complete; best weights: %s", model.trainer.best)
    return results
