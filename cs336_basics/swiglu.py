import torch

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff= None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            d_ff_raw = int((8/3) * d_model)
            d_ff = ((d_ff_raw + 32) // 64) * 64
        self.d_ff = d_ff

        self.w1 = torch.nn.Parameter(torch.empty((self.d_ff, self.d_model), device=device, dtype=dtype))
        self.w3 = torch.nn.Parameter(torch.empty((self.d_ff, self.d_model), device=device, dtype=dtype))
        self.w2 = torch.nn.Parameter(torch.empty((self.d_model, self.d_ff), device=device, dtype=dtype))
        std_w1_w3 = (2.0 / (self.d_model + self.d_ff)) ** 0.5
        std_w2 = (2.0 / (self.d_ff + self.d_model)) ** 0.5
        torch.nn.init.trunc_normal_(self.w1, mean=0.0, std=std_w1_w3, a=-3*std_w1_w3, b=3*std_w1_w3)
        torch.nn.init.trunc_normal_(self.w3, mean=0.0, std=std_w1_w3, a=-3*std_w1_w3, b=3*std_w1_w3)
        torch.nn.init.trunc_normal_(self.w2, mean=0.0, std=std_w2, a=-3*std_w2, b=3*std_w2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = x @ self.w1.T
        w3x = x @ self.w3.T
        silu_w1x = w1x * torch.sigmoid(w1x)
        gated = silu_w1x * w3x
        output = gated @ self.w2.T
        return output
