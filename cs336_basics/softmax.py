import torch
from jaxtyping import Float

def softmax(tensor: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
    max = tensor.max(dim=dim, keepdim=True).values
    tensor_shifted = tensor - max
    exp_tensor = torch.exp(tensor_shifted)
    exp_sum = exp_tensor.sum(dim=dim, keepdim=True)
    result = exp_tensor / exp_sum
    return result