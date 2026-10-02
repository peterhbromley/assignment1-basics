from math import sqrt

from einops import einsum
import torch
from torch import nn, Tensor


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # Weights have shape (d_out, d_in), initialized with
        # N(mean 0, var 2/(d_in + d_out)), truncated at -3*std to 3*std
        weights = torch.empty(
            out_features,
            in_features,
            device=device,
            dtype=dtype,
        )
        std = sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(weights, 0, std, -3 * std, 3 * std)

        self.weight = nn.Parameter(weights)

    def forward(self, x: Tensor) -> Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
