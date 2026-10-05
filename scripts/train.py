import argparse
from collections.abc import Callable, Iterator
from functools import partial
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import Parameter
import wandb

from llm.checkpoint import load_checkpoint, save_checkpoint
from llm.data import get_batch
from llm.loss import cross_entropy_loss
from llm.optimizer import AdamW, gradient_clipping
from llm.schedule import cosine_annealing_lr_schedule
from llm.transformer import Transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer language model.")

    # Data: paths to tokenized datasets.
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--val-path", required=True)

    # Model: TinyStories defaults.
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    # Optimizer and learning-rate schedule: starting values for tuning.
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=40000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--cosine-end-step", type=int, default=None,
                        help="Defaults to --num-steps.")

    # Runtime.
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu", help="For example: cpu, mps, or cuda:0.")
    parser.add_argument("--seed", type=int, default=42)

    # Logging and checkpoints.
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--val-interval", type=int, default=1000)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--output-dir", default="runs/default")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume from.")

    # Optional Weights & Biases logging.
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="cs336")
    parser.add_argument("--wandb-entity", default=None, help="W&B team or username.")
    parser.add_argument("--wandb-run-name", default=None, help="Display name for the W&B run.")

    args = parser.parse_args()

    if args.cosine_end_step is None:
        args.cosine_end_step = args.num_steps

    # Check that a bunch of int args are >= 0
    for name in (
        "vocab_size", "context_length", "num_layers", "d_model", "num_heads",
        "d_ff", "batch_size", "num_steps", "log_interval", "val_interval",
        "val_batches", "checkpoint_interval",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")

    # Make sure d_model divides evenly across heads
    if args.d_model % args.num_heads:
        parser.error("d_model must divide evenly across heads")

    # Make sure d_model // num_heads (d_k and d_v) is even (for RoPE)
    if (args.d_model // args.num_heads) % 2:
        parser.error("d_model // num_heads must be even (for RoPE implementation).")

    # Validate lr scheduling args
    if not 0 < args.warmup_steps < args.cosine_end_step <= args.num_steps:
        parser.error("Require 0 < warmup_steps < cosine_end_step <= num_steps.")

    if not 0 <= args.min_lr <= args.max_lr < math.inf:
        parser.error("Require finite learning rates with 0 <= min_lr <= max_lr.")

    # Validate optimizer args
    if not (0 <= args.beta1 < 1 and 0 <= args.beta2 < 1):
        parser.error("Both betas must be in [0, 1).")

    for name in ("eps", "weight_decay", "max_grad_norm"):
        if not 0 <= getattr(args, name) < math.inf:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative.")

    # Validate rope theta value
    if not 0 < args.rope_theta:
        parser.error("--rope-theta must be > 0.")

    return args


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float):
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    get_batch: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    num_batches: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    # Use try / finally so that model gets put back in train mode even on exception.
    try:
        # Aggregate cross entropy losses on validation data.
        losses = []
        for _ in range(num_batches):
            inputs, targets = get_batch()
            losses.append(cross_entropy_loss(model(inputs), targets))

        # Take mean over all calculated validation losses.
        mean_loss = torch.stack(losses).mean()
        return {"val/loss": mean_loss.item(), "val/perplexity": mean_loss.exp().item()}
    finally:
        model.train(was_training)


def log_progress(
    completed_steps: int,
    loss: torch.Tensor,
    lr: float,
    *,
    log_interval: int,
    val_interval: int,
    num_steps: int,
    evaluate: Callable[[], dict[str, float]],
    output_dir: Path,
    start_time: float,
    wandb_run: wandb.Run | None = None,
) -> None:
    # Check if we should run a validation pass.
    final_step = completed_steps == num_steps
    validate = final_step or completed_steps % val_interval == 0
    if not (validate or completed_steps % log_interval == 0):
        return

    metrics = {
        "train/loss": loss.item(),
        "train/lr": lr,
    }

    if validate:
        # On validate interval, run validation and update metrics with results
        metrics.update(evaluate())

    metrics["elapsed_seconds"] = time.perf_counter() - start_time

    # Log to stdout
    print(f"step={completed_steps} {metrics}", flush=True)

    # Write metrics artifact to local storage
    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps({"step": completed_steps, **metrics}) + "\n")

    # Write metrics to Weights and Biases if configured 
    if wandb_run is not None:
        wandb_run.log(metrics, step=completed_steps)


def save_progress(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_steps: int,
    *,
    checkpoint_interval: int,
    output_dir: Path,
) -> None:
    if completed_steps % checkpoint_interval != 0:
        return
    save_checkpoint(
        model,
        optimizer,
        completed_steps,
        output_dir / f"checkpoint_{completed_steps}.pt",
    )


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    start_step: int,
    num_steps: int,
    get_batch: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    lr_schedule: Callable[[int], float],
    gradient_clipping: Callable[[Iterator[Parameter]], None],
    log_progress: Callable[[int, torch.Tensor, float], None],
    save_progress: Callable[[torch.nn.Module, torch.optim.Optimizer, int], None],
) -> int:
    model.train()
    completed_steps = start_step

    for iteration in range(start_step, num_steps): 
        # Get the learning rate based on current iteration, pass to optimizer state
        lr = lr_schedule(iteration)
        set_learning_rate(optimizer, lr) 

        # Clear gradients.
        optimizer.zero_grad()

        # Get a batch of input and target data.
        inputs, targets = get_batch()

        # Run a forward pass and calculate loss.
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)

        # Run a backward pass to calculate gradients, clip large gradients.
        loss.backward()
        gradient_clipping(model.parameters())

        # Update model weights with one optimizer step. 
        optimizer.step()

        completed_steps = iteration + 1

        # Logging and checkpointing:
        log_progress(completed_steps, loss, lr)
        save_progress(model, optimizer, completed_steps)

    return completed_steps


def main() -> None:
    args = parse_args()
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    dtype = torch.float32

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_train = np.load(args.train_path, mmap_mode="r")
    data_val = np.load(args.val_path, mmap_mode="r")
    
    model = Transformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
        device=device,
        dtype=dtype,
    )
    optimizer = AdamW(
        params=model.parameters(), 
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    lr_schedule_fn = partial(
        cosine_annealing_lr_schedule,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmups=args.warmup_steps,
        final_iteration=args.cosine_end_step,
    )
    batch_fn = partial(
        get_batch,
        token_ids=data_train,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=device,
    )
    clipping_fn = partial(
        gradient_clipping,
        max_l2_norm=args.max_grad_norm,
    )
    val_batch_fn = partial(
        get_batch,
        token_ids=data_val,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=device,
    )
    evaluate_fn = partial(evaluate, model, val_batch_fn, args.val_batches)

    start_step = 0

    # Load model state, optimizer state, and iteration from checkpoint if
    # checkpoint path is supplied.
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, optimizer)

    # Checkpoint loading restores optimizer hyperparameters. Record the values
    # actually in use rather than any overridden CLI defaults.
    config = vars(args).copy()
    config.update(
        beta1=optimizer.param_groups[0]["betas"][0],
        beta2=optimizer.param_groups[0]["betas"][1],
        eps=optimizer.param_groups[0]["eps"],
        weight_decay=optimizer.param_groups[0]["weight_decay"],
        start_step=start_step,
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    run = None
    if args.wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=config,
        )

    try:
        log_fn = partial(
            log_progress,
            log_interval=args.log_interval,
            val_interval=args.val_interval,
            num_steps=args.num_steps,
            evaluate=evaluate_fn,
            output_dir=output_dir,
            start_time=time.perf_counter(),
            wandb_run=run,
        )
        save_fn = partial(
            save_progress,
            checkpoint_interval=args.checkpoint_interval,
            output_dir=output_dir,
        )
        completed_steps = train(
            model=model,
            optimizer=optimizer,
            start_step=start_step,
            num_steps=args.num_steps,
            get_batch=batch_fn,
            lr_schedule=lr_schedule_fn,
            gradient_clipping=clipping_fn,
            log_progress=log_fn,
            save_progress=save_fn,
        )
        save_checkpoint(
            model,
            optimizer,
            completed_steps,
            output_dir / "checkpoint_final.pt",
        )
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
