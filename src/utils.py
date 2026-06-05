import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.distributed as distributed


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model: nn.Module, path: Path, **metadata):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), **metadata}, path)


def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    return checkpoint


def save_training_state(path: Path, model: nn.Module, optimizer, scheduler, **progress):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            **progress,
        },
        path,
    )


def load_training_state(path: Path, model: nn.Module, optimizer, scheduler, device) -> dict:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])

    return checkpoint


def setup_distributed(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    distributed.init_process_group(backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup_distributed():
    if distributed.is_initialized():
        distributed.destroy_process_group()


def is_distributed() -> bool:
    return distributed.is_available() and distributed.is_initialized()


def is_main_process() -> bool:
    return not is_distributed() or distributed.get_rank() == 0


def all_gather_concat(tensor: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return tensor

    gathered = [None] * distributed.get_world_size()
    distributed.all_gather_object(gathered, tensor)

    return torch.cat(gathered)


def all_reduce_sum(value: float) -> float:
    if not is_distributed():
        return value

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")

    tensor = torch.tensor(value, device=device)
    distributed.all_reduce(tensor, op=distributed.ReduceOp.SUM)

    return tensor.item()
