import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, sequence_length, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x_squared = x ** 2
        x_mean_squared = x_squared.mean(dim=-1, keepdim=True)
        rms = torch.sqrt(x_mean_squared + self.eps)
        rmsnorm = x / rms * self.g
        result = rmsnorm.to(in_dtype)
        return result