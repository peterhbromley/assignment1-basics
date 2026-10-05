import math


def cosine_annealing_lr_schedule(
        iteration: int, 
        max_lr: float, 
        min_lr: float, 
        warmups: int, 
        final_iteration: int,
    ):
    if warmups <= 0:
        raise ValueError(f"Invalid value for warmups: {warmups}")

    if final_iteration == warmups:
        raise ValueError("final_iteration can't equal warmups")

    if iteration < warmups:
        return (iteration / warmups) * max_lr

    if warmups <= iteration <= final_iteration:
        cos_term = math.pi * ((iteration - warmups) / (final_iteration - warmups))
        return min_lr + 0.5 * (1 + math.cos(cos_term)) * (max_lr - min_lr)

    
    return min_lr
