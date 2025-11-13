import torch

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weights = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.weights = torch.nn.init.trunc_normal_(self.weights, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.device = device
        self.dtype = dtype

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]