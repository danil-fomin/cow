from pathlib import Path

import torch
import torch.multiprocessing as multiprocessing
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data.dataset import BCSDataset
from src.data.transforms import build_cpu_transform, build_gpu_transform
from src.models.model import build_model, build_predictor
from src.losses import build_loss
from src.evaluate import plot_training_curves
from src.training.loop import train_one_epoch, validate
from src.logging_setup import setup_logging, get_logger
from src.utils import (
    set_seed,
    save_checkpoint,
    save_training_state,
    load_training_state,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
)

log = get_logger(__name__)

BEST_CHECKPOINT_PATH = Path("outputs/checkpoints/best.pt")
LAST_CHECKPOINT_PATH = Path("outputs/checkpoints/last.pt")


def _build_loader(split: str, transforms, config: dict, distributed: bool, train: bool):
    dataset = BCSDataset(split, transforms=transforms)
    sampler = DistributedSampler(dataset, shuffle=train) if distributed else None

    num_workers = config["num_workers"]
    loader_kwargs = {
        "batch_size": config["batch_size"],
        "shuffle": train and sampler is None,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    return DataLoader(dataset, **loader_kwargs)


def _train_worker(rank: int, world_size: int, config: dict):
    distributed = world_size > 1

    if distributed:
        setup_distributed(rank, world_size)
        
    setup_logging(is_main=is_main_process())

    set_seed()
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
    class_values = config["class_values"]
    image_size = config["img_size"]

    cpu_transform = build_cpu_transform(image_size)
    train_gpu_transform = build_gpu_transform(image_size, train=True)
    eval_gpu_transform = build_gpu_transform(image_size, train=False)

    train_loader = _build_loader("train", cpu_transform, config, distributed, train=True)
    val_loader = _build_loader("val", cpu_transform, config, distributed, train=False)

    model = build_model(config).to(device)
    if distributed:
        device_ids = [rank] if torch.cuda.is_available() else None
        model = DistributedDataParallel(model, device_ids=device_ids)

    loss_fn = build_loss(config)
    predict_fn = build_predictor(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learn_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    history = {"train": [], "val": []}
    start_epoch = 1
    best_qwk = -1.0
    epochs_without_improvement = 0
    patience = config["early_stopping_patience"]

    underlying_model = model.module if distributed else model
    if LAST_CHECKPOINT_PATH.exists():
        state = load_training_state(LAST_CHECKPOINT_PATH, underlying_model, optimizer, scheduler, device)
        start_epoch = state["epoch"] + 1
        best_qwk = state["best_qwk"]
        epochs_without_improvement = state["epochs_without_improvement"]
        history = state["history"]
        log.info("resumed from epoch %d (best qwk=%.4f)", state["epoch"], best_qwk)

    for epoch in range(start_epoch, config["epochs"] + 1):
        if distributed:
            train_loader.sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, class_values, epoch,
            predict_fn, train_gpu_transform
        )
        val_metrics = validate(
            model, val_loader, loss_fn, device, class_values, epoch,
            predict_fn, eval_gpu_transform
        )
        scheduler.step()

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        log.info("epoch %d: train %s", epoch, train_metrics)
        log.info("epoch %d: val   %s", epoch, val_metrics)

        if val_metrics["qwk"] > best_qwk:
            best_qwk = val_metrics["qwk"]
            epochs_without_improvement = 0
            if is_main_process():
                save_checkpoint(underlying_model, BEST_CHECKPOINT_PATH, epoch=epoch, qwk=best_qwk)
                log.info("saved best -> %s (qwk=%.4f)", BEST_CHECKPOINT_PATH, best_qwk)
        else:
            epochs_without_improvement += 1

        if is_main_process():
            save_training_state(
                LAST_CHECKPOINT_PATH,
                underlying_model,
                optimizer,
                scheduler,
                epoch=epoch,
                best_qwk=best_qwk,
                epochs_without_improvement=epochs_without_improvement,
                history=history,
            )

        if epochs_without_improvement >= patience:
            log.info("early stopping: val QWK has not improved for %d consecutive epochs", patience)
            break

    if is_main_process():
        curves_path = plot_training_curves(history)
        log.info("training curves saved: %s", curves_path)

    if distributed:
        cleanup_distributed()


def run_train(config: dict):
    num_gpus = config["num_gpus"]

    if num_gpus > 1 and torch.cuda.is_available():
        multiprocessing.spawn(_train_worker, args=(num_gpus, config), nprocs=num_gpus)
    else:
        _train_worker(rank=0, world_size=1, config=config)
