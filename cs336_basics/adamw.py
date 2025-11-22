import torch
from typing import Optional
from collections.abc import Callable
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.1, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, (beta1, beta2), weight_decay, eps = (
                group["lr"],
                group["betas"],
                group["weight_decay"],
                group["eps"],
            )
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                    state["t"] = 0
                m, v, t = state["m"], state["v"], state["t"] + 1

                grad = p.grad.data

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad**2)
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= alpha_t * m / (v.sqrt() + eps)
                if weight_decay != 0:
                    p.data -= lr * weight_decay * p.data

                state["m"] = m
                state["v"] = v
                state["t"] = t
        return loss
