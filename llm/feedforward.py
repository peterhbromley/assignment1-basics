import torch
from torch import nn

from llm.linear import Linear


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # Feed forward dimension defaults to 8/3 * model dim, rounded to nearest
        # multiple of 64 for hardware efficiency
        if d_ff is None:
            self.d_ff = 64 * round((8 / 3 * d_model) / 64)
        else:
            self.d_ff = d_ff

        self.d_model = d_model

        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply SiLU to one branch
        w1_out = self.w1(x)
        silu = w1_out * torch.sigmoid(w1_out)

        # No SiLU applied to second branch
        w3_out = self.w3(x)

        # SwiGLU is eltwise output from the two branches, passed through another
        # linear transform.
        return self.w2(silu * w3_out)
