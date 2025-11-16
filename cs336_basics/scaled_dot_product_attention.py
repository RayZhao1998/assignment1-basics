import torch
from jaxtyping import Float
from cs336_basics.softmax import softmax

def scaled_dot_product_attention(
    Q: Float[torch.Tensor, "... queries d_k"],
    K: Float[torch.Tensor, "... keys d_k"],
    V: Float[torch.Tensor, "... values d_v"],
    mask: Float[torch.Tensor, "... queries keys"] | None = None
) -> Float[torch.Tensor, "... queries d_v"]:
    d_k = Q.shape[-1]
    sqrt_d_k: int = d_k ** 0.5
    scores = Q @ K.transpose(-2, -1) / sqrt_d_k # (..., queries, keys)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attention = softmax(scores, dim=-1) # (..., queries, keys)
    # (..., queries, keys) @ (..., values, d_v) -> (..., queries, d_v)
    result = attention @ V 
    return result