from jaxtyping import Float
from torch import Tensor, exp


def SiLU(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    return in_features * (1 / (1 + exp(-in_features)))
