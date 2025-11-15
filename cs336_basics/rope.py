import torch

class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        d_k_half= d_k // 2
        inv_freq = 1 / (theta ** (torch.arange(0, d_k_half, device=device, dtype=torch.float32) / d_k_half))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        # token_position: (..., seq_len)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        d_k = x.size(-1)
        half = d_k // 2
        x = x.view(*x.shape[:-1], half, 2)
        x1 = x[..., 0]
        x2 = x[..., 1]

        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos

        out = torch.stack([x1_rot, x2_rot], dim=-1).reshape(*x.shape[:-2], d_k)
        return out