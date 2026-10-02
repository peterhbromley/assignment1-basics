import torch
from torch import nn


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()

        assert d_k % 2 == 0, f"d_k must be even, got {d_k}"

        # Initialize position indices and per-pair rotation rates, shape (max_seq_len, 1)
        # and (1, d_k//2) so they can be broadcast into a matrix.
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)[:, None]
        pair_indices = torch.arange(d_k // 2, device=device, dtype=torch.float32)[None, :]

        # Calculate angles. pair_indices start at 0 (instead of 1 in the book),
        # so rates calc has been updated accordingly
        rates = theta ** (-2 * pair_indices / d_k)
        angles = positions * rates

        # Apply cosine / sine elementwise to create lookup tables to store in buffer.
        sin = torch.sin(angles)
        cos = torch.cos(angles)

        # Store in buffer (not Parameter since these are fixed)
        # Buffer is a tensor owned by nn.Module that moves with module when
        # .to(device) called. persistent=False means it won't be saved in
        # state dict
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x = x.to(torch.float32)

        # Get cos and sin values for the token positions
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # Get the first (a) and second (b) coordinates of each adjacent feature pair
        a = x[..., 0::2]
        b = x[..., 1::2]

        # Calculate rotations
        rotated_a = a * cos - b * sin
        rotated_b = a * sin + b * cos

        # Pair the coordinates back up, then restore original shape
        paired = torch.stack((rotated_a, rotated_b), dim=-1)
        return paired.flatten(start_dim=-2).to(original_dtype)
