import torch
import os
from typing import BinaryIO, IO


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    obj = {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "iteration": iteration,
    }
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    obj = torch.load(src)
    model_state, optimizer_state, iteration = (
        obj["model_state"],
        obj["optimizer_state"],
        obj["iteration"]
    )
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    return iteration 
