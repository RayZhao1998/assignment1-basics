import numpy as np
import torch


def data_loading(
    dataset: np.array, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(dataset)
    max_start = n - context_length - 1
    if max_start < 0:
        raise ValueError("dataset is too short for given context_length")
    starts = np.random.randint(0, max_start + 1, size=batch_size)
    inputs_np = np.stack([dataset[s : s + context_length] for s in starts])
    targets_np = np.stack([dataset[s + 1 : s + 1 + context_length] for s in starts])
    inputs = torch.tensor(inputs_np, device=device, dtype=torch.long)
    targets = torch.tensor(targets_np, device=device, dtype=torch.long)
    return inputs, targets