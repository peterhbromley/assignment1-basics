import torch
from torch import nn, Tensor


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # Weights have shape (vocab, d_model), initialized with
        # N(mean 0, var 1) truncated at -3 to 3
        weights = torch.empty(
            num_embeddings,
            embedding_dim,
            device=device,
            dtype=dtype,
        )
        nn.init.trunc_normal_(weights, 0, 1, -3, 3)

        self.weight = nn.Parameter(weights)

    def forward(self, token_ids: Tensor) -> Tensor:
        # W:          (vocab_size, d_model)
        # token_ids:  (batch_size, sequence_length)
        # output:     (batch_size, sequence_length, d_model)
        return self.weight[token_ids]
