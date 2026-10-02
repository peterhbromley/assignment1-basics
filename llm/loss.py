from jaxtyping import Float, Int
import torch
from torch import Tensor


def cross_entropy_loss(
    logits: Float[Tensor, "... vocab_size"],
    targets: Int[Tensor, "..."],
) -> Float[Tensor, ""]:
    """
    Where o_i = logit at position i, t_i+1 = target at position i+1:
    loss = -log(softmax(o_i)[t_i+1])
         = -log(exp(o_i[t_i+1]) / sum_j exp(o_i[j]))
         , log(a/b) = log(a) - log(b), so:
         = -log(exp(o_i[t_i+1])) + log(sum_j exp(o_i[j]))
         , log(exp(z)) = z, so:

         = -o_i[t_i+1] + log(sum_j exp(o_i[j]))
    """
    # Subtract largest logit (over vocabulary dim) for numerical stability
    stable = logits - torch.max(logits, dim=-1, keepdim=True).values

    # Index the vocab dimension for each position in the sequence with the target
    # token index, to get the logit (score) for the target token.
    target_scores = torch.gather(stable, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # Calculate cross entropy using the formula derived from -log(softmax(o_i)[t_i+1])
    cross_entropy = -1 * target_scores + torch.log(torch.sum(torch.exp(stable), dim=-1))

    # Average over entire batch
    return cross_entropy.mean()


def perplexity(
    logits: Float[Tensor, "... vocab_size"],
    targets: Int[Tensor, "..."],
) -> Float[Tensor, ""]:
    return torch.exp(cross_entropy_loss(logits, targets))
