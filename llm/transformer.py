from jaxtyping import Float, Int
import torch
from torch import Tensor, nn

from llm.attention import CausalMultiHeadSelfAttention
from llm.embedding import Embedding
from llm.feedforward import FeedForward
from llm.linear import Linear
from llm.positional import RotaryPositionalEmbeddings
from llm.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbeddings | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.ffn = FeedForward(
            d_model,
            d_ff,
            device=device,
            dtype=dtype,
        )
        self.attn = CausalMultiHeadSelfAttention(
            d_model,
            num_heads,
            rope=rope,
            device=device,
            dtype=dtype,
        )
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... seq_len d_model"]):
        y = x + self.attn(self.ln1(x))
        return y + self.ffn(self.ln2(y))


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        rope = RotaryPositionalEmbeddings(theta, d_model // num_heads, context_length, device=device)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, rope=rope, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len vocab_size"]:
        x = self.token_embeddings(x)
        for block in self.layers:
            x = block(x)
        return self.lm_head(self.ln_final(x))
