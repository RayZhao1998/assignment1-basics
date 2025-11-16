import torch
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from jaxtyping import Float, Int
from einops import rearrange
from cs336_basics.rope import RoPE

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = torch.nn.Parameter(torch.zeros(d_model, d_model, device=device, dtype=dtype))
        self.W_k = torch.nn.Parameter(torch.zeros(d_model, d_model, device=device, dtype=dtype))
        self.W_v = torch.nn.Parameter(torch.zeros(d_model, d_model, device=device, dtype=dtype))
        self.W_o = torch.nn.Parameter(torch.zeros(d_model, d_model, device=device, dtype=dtype))

    def forward(self, x: Float[torch.Tensor, "... sequence_length d_in"]) -> Float[torch.Tensor, "... sequence_length d_out"]:
        # x: (..., seq_len, d_model)
        *batch_dims, seq_len, _ = x.shape

        # QKV projection
        q = x @ self.W_q.T
        k = x @ self.W_k.T
        v = x @ self.W_v.T

        # reshape into heads
        q = q.view(*batch_dims, seq_len, self.num_heads, self.head_dim).transpose(-3, -2) # (..., num_heads, seq_len, head_dim)
        k = k.view(*batch_dims, seq_len, self.num_heads, self.head_dim).transpose(-3, -2) # (..., num_heads, seq_len, head_dim)
        v = v.view(*batch_dims, seq_len, self.num_heads, self.head_dim).transpose(-3, -2) # (..., num_heads, seq_len, head_dim)

        mask = torch.ones(seq_len, seq_len, device=x.device).tril()
        mask = rearrange(mask, 'T1 T2 -> 1 1 T1 T2')

        # scaled dot-product attention
        out = scaled_dot_product_attention(q, k, v, mask=mask) # (..., num_heads, seq_len, head_dim)

        # concat heads
        out = out.transpose(-3, -2).reshape(*batch_dims, seq_len, self.d_model)

        # output projection
        out = out @ self.W_o.T
        return out
    
class MultiHeadSelfAttentionWithRoPE(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = torch.nn.Parameter(torch.zeros(d_model, d_model, device=device, dtype=dtype))
        self.W_k = torch.nn.Parameter(torch.zeros(d_model, d_model, device=device, dtype=dtype))
        self.W_v = torch.nn.Parameter(torch.zeros(d_model, d_model, device=device, dtype=dtype))
        self.W_o = torch.nn.Parameter(torch.zeros(d_model, d_model, device=device, dtype=dtype))

        self.rope = RoPE(theta, self.head_dim, max_seq_len, device=device)

    def forward(self, x: Float[torch.Tensor, "... sequence_length d_in"], token_positions: Int[torch.Tensor, "... sequence_length"]) -> Float[torch.Tensor, "... sequence_length d_out"]:
        # x: (..., seq_len, d_model)
        *batch_dims, seq_len, _ = x.shape

        # QKV projection
        q = x @ self.W_q.T
        k = x @ self.W_k.T
        v = x @ self.W_v.T

        # reshape into heads
        q = q.view(*batch_dims, seq_len, self.num_heads, self.head_dim).transpose(-3, -2) # (..., num_heads, seq_len, head_dim)
        k = k.view(*batch_dims, seq_len, self.num_heads, self.head_dim).transpose(-3, -2) # (..., num_heads, seq_len, head_dim)
        v = v.view(*batch_dims, seq_len, self.num_heads, self.head_dim).transpose(-3, -2) # (..., num_heads, seq_len, head_dim)

        q = self.rope.forward(q, token_positions=token_positions)
        k = self.rope.forward(k, token_positions=token_positions)

        mask = torch.ones(seq_len, seq_len, device=x.device).tril()
        mask = rearrange(mask, 'T1 T2 -> 1 1 T1 T2')

        # scaled dot-product attention
        out = scaled_dot_product_attention(q, k, v, mask=mask) # (..., num_heads, seq_len, head_dim)

        # concat heads
        out = out.transpose(-3, -2).reshape(*batch_dims, seq_len, self.d_model)

        # output projection
        out = out @ self.W_o.T
        return out 