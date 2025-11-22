from collections.abc import Iterable
import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    l2_norm = torch.sqrt(sum((p.grad**2).sum() for p in parameters if p.grad is not None))
    if l2_norm >= max_l2_norm:
        factor = max_l2_norm / (l2_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad *= factor
