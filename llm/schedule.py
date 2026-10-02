import math


def cosine_annealing_lr_schedule(
        iteration: int, 
        max_lr: float, 
        min_lr: float, 
        warm_ups: int, 
        final_iteration: int,
    ):
    if warm_ups <= 0:
        raise ValueError(f"Invalid value for warm_ups: {warm_ups}")

    if final_iteration == warm_ups:
        raise ValueError("final_iteration can't equal warm_ups")

    if iteration < warm_ups:
        return (iteration / warm_ups) * max_lr

    if warm_ups <= iteration <= final_iteration:
        cos_term = math.pi * ((iteration - warm_ups) / (final_iteration - warm_ups))
        return min_lr + 0.5 * (1 + math.cos(cos_term)) * (max_lr - min_lr)

    
    return min_lr
