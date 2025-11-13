import torch

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.weights = torch.nn.init.trunc_normal_(self.weights)
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights.T