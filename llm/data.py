from jaxtyping import Int
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor


def get_batch(
    token_ids: Int[NDArray, " num_tokens"], 
    batch_size: int, 
    context_length: int, 
    device: torch.device,
) -> tuple[
    Int[Tensor, " batch_size context_length"], 
    Int[Tensor, " batch_size context_length"],
]:
    batch_start_idxs = np.random.randint(0, len(token_ids) - context_length, batch_size)
    batch_idxs = batch_start_idxs[:, None] + np.arange(context_length + 1)[None, :]
    seqs = token_ids[batch_idxs]
    return torch.from_numpy(seqs[:, :-1]).to(device=device), torch.from_numpy(seqs[:, 1:]).to(device=device)
