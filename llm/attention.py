from math import sqrt

from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
import torch
from torch import Tensor, nn

from llm.functional import softmax
from llm.linear import Linear
from llm.positional import RotaryPositionalEmbeddings


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    # Get d_k from last dim of Q matrix
    d_k = Q.shape[-1]

    # Scaled attention tensor scores matrix is QK^T / sqrt(d_k)
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / sqrt(d_k)

    # Put -inf in matrix positions that can't be attended to (e.g. token comes after
    # current token). -inf will become 0 after softmax.
    if mask is not None:
        scores = torch.masked_fill(scores, ~mask, -torch.inf)

    return einsum(softmax(scores, dim=-1), V, "... queries keys, ... keys d_v -> ... queries d_v")


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbeddings | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        assert num_heads > 0 and d_model % num_heads == 0, (
            f"num_heads must divide d_model, got num_heads: {num_heads} and d_model: {d_model}"
        )

        d_k = d_v = d_model // num_heads

        self.q_proj = Linear(d_model, num_heads * d_k, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, num_heads * d_k, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, num_heads * d_v, device=device, dtype=dtype)
        self.output_proj = Linear(num_heads * d_v, d_model, device=device, dtype=dtype)

        self.rope = rope
        self.num_heads = num_heads

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_model"],
        rope_token_positions: (Int[Tensor, " seq_len"] | Int[Tensor, " ... seq_len"] | None) = None,
    ) -> Tensor:
        # Input x has shape (..., seq_len, d_model)
        seq_len = x.shape[-2]
        device = x.device

        # Apply linear transforms for each of W_Q, W_K, W_V layers. Output will have
        # shape (..., seq_len h*d_k) for Q and K, and (..., seq_len h*d_v) for V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split the d_model dimension (size h * d_k or d_v) into (h, d_k or d_v)
        # and rearrange so that h is a leading "batch-like" dimension
        Q = rearrange(Q, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        K = rearrange(K, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        V = rearrange(V, "... seq_len (h d_v) -> ... h seq_len d_v", h=self.num_heads)

        # Optionally apply RoPE if the module was supplied. Additionally, use supplied
        # token positions or a default (sequence from 0 to seq_len-1) if not supplied.
        if self.rope is not None:
            if rope_token_positions is None:
                rope_token_positions = torch.arange(0, seq_len, device=device, dtype=torch.int)
            else:
                # In case caller-supplied token positions differ per batch, add
                # a singleton head dim.
                if rope_token_positions.ndim > 1:
                    rope_token_positions = rope_token_positions.unsqueeze(-2)
            Q = self.rope(Q, rope_token_positions)
            K = self.rope(K, rope_token_positions)

        # Create *allow* mask for dot product attention. Attention matrix is shape
        # (queries, keys), so we allow on and *below* the diagonal.
        mask = torch.tril(
            torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
            diagonal=0,
        )
        mha = scaled_dot_product_attention(Q, K, V, mask)

        # After attention, need to reshape for expected W_O input dim of h*d_v
        mha = rearrange(mha, "... h seq_len d_v -> ... seq_len (h d_v)")

        # Complete the Multi Head Self Attention operation by applying final output
        # linear transformation.
        return self.output_proj(mha)
