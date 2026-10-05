from jaxtyping import Int, Float
from numpy.typing import NDArray
import torch
from torch import Tensor

from llm.functional import softmax


@torch.no_grad
def decode(
    model: torch.nn.Module, 
    prompt_ids: Int[Tensor, " seq_len"],
    context_length: int,
    eos_token_id: int,
    max_tokens: int,
    temperature: float,
    top_p_target: float,
) -> list[int]:
    if len(prompt_ids) == 0:
        raise ValueError("prompt_ids must not be empty.")

    if context_length <= 0:
        raise ValueError(f"context_length must be > 0, got {context_length}")

    if max_tokens < 0:
        raise ValueError(f"max_tokens must be >= 0, got {max_tokens}")

    was_training = model.training
    model.eval()
    try:
        generated = []
        while len(generated) < max_tokens:
            # Pass in the most recent context_length tokens.
            prompt_start_idx = max(0, len(prompt_ids) - context_length)
            logits = model(prompt_ids[prompt_start_idx:])

            # Model output is (..., seq_len, vocab_size), but we only need to sample
            # from the most recent token logits in the sequence.
            next_token_logits = logits[..., -1, :]

            probabs = softmax(next_token_logits, dim=-1, temperature=temperature) 
            sampled = top_p_sampling(probabs, top_p_target)

            # NOTE: This only works with batch size of 1. When we extend this to larger
            # batch size, we will need to update this.
            sampled_id = sampled.item()

            if sampled_id == eos_token_id:
                break
            
            prompt_ids = torch.cat((prompt_ids, sampled), dim=-1)
            generated.append(sampled_id)
    finally: 
        model.train(was_training)
    
    return generated


def top_p_sampling(
    probabs: Float[torch.Tensor, " ... vocab_size"],
    target: float,
) -> Int[torch.Tensor, " ... 1"]:
    if target <= 0 or target > 1:
        raise ValueError(f"target must be > 0 and <= 1, got {target}")

    # Sort probabs over vocab in descending order, retain indices,
    # calculate cumulative sum of sorted probabs.
    sorted_probabs, sorted_probabs_idxs = torch.sort(
        probabs, 
        dim=-1, 
        descending=True, 
        stable=True,
    )
    cumulative = torch.cumsum(sorted_probabs, dim=-1)

    # Subtracting sorted_probabs from cumulative ensures that we keep all
    # values that sum to the probab mass >= target, _including_ the final
    # token that reaches or crosses the threshold.
    # 
    # Ex: (p=0.7)
    #  Sorted probabilities:   [0.50, 0.30, 0.15, 0.05]
    #  Cumulative mass:        [0.50, 0.80, 0.95, 1.00]
    #  Cumulative - Sorted:    [0.00, 0.50, 0.80, 0.95]
    #  Keep:                   [True, True, False, False]
    keep_mask = (cumulative - sorted_probabs) < target

    # Zero out excluded probabs.
    sorted_probabs[~keep_mask] = 0

    # Flatten leading batch-like dimensions for torch.multinomial.
    flat_probabs = sorted_probabs.reshape(-1, sorted_probabs.shape[-1])

    # Sample from new probab distribution. torch.multinomial handles normalizaion.
    sampled = torch.multinomial(flat_probabs, 1)

    # Restore leading batch dimensions.
    sampled = sampled.reshape(*sorted_probabs.shape[:-1], 1)

    return torch.gather(sorted_probabs_idxs, dim=-1, index=sampled) 
