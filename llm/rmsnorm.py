from einops import reduce
import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        # Learnable "gain" parameter
        weights = torch.ones(d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 to prevent overflow when squaring input.
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(reduce(torch.square(x), "... d_model -> ... 1", "mean") + self.eps)
        rms_norm = (x / rms) * self.weight

        return rms_norm.to(in_dtype)
