import torch
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.multihead_self_attention import MultiHeadSelfAttentionWithRoPE
from cs336_basics.swiglu import SwiGLU
from jaxtyping import Float

class TranformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: int,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.mhsa = MultiHeadSelfAttentionWithRoPE(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[torch.Tensor, "... sequence_length d_model"]):
        batch, seq_len, _ = x.shape

        token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        token_positions = token_positions.unsqueeze(0).expand(batch, seq_len)

        norm1 = self.ln1(x)
        attention = self.mhsa(norm1, token_positions)
        sublayer1_result = x + attention
        norm2 = self.ln2(sublayer1_result)
        pwff = self.ffn(norm2)
        result = sublayer1_result + pwff
        return result
