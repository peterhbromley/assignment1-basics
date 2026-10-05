import torch


def softmax(
    x: torch.Tensor, 
    dim: int, 
    temperature: float | None=None,
) -> torch.Tensor:
    # Subtract each group's maximum along dim for numerical stability (avoid inf/inf = NaN).
    # If temperature param is supplied, divide by temperature. 
    stable = x - torch.max(x, dim=dim, keepdim=True).values
    if temperature is not None:
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        stable = stable / temperature

    # Then take the elementwise exp of all values, and divide by the sum of the exp'd
    # values along dim. 
    exp_out = torch.exp(stable) 
    return exp_out / torch.sum(exp_out, dim=dim, keepdim=True)
