import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Subtract each group's maximum along dim for numerical stability (avoid inf/inf = NaN)
    # Then take the elementwise exp of all values, and divide by the sum of the exp'd
    # values along dim.
    exp_out = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    return exp_out / torch.sum(exp_out, dim=dim, keepdim=True)
