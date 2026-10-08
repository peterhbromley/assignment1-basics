"""Run from the project root with: python -m scripts.generate --help."""

import argparse
import json
from pathlib import Path

import torch

from llm.generation import decode
from llm.tokenizer import Tokenizer
from llm.transformer import Transformer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a completion from a trained checkpoint.")
    parser.add_argument("--model", type=Path, required=True, help="Path to a training checkpoint.")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--config", type=Path, help="Defaults to config.json beside the checkpoint.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cpu", help="cpu, cuda:0, or mps.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    config_path = args.config or args.model.with_name("config.json")
    config = json.loads(config_path.read_text())

    tokenizer = Tokenizer.from_pickle(args.tokenizer)
    model = Transformer.from_config(
        config,
        device=torch.device(args.device),
        dtype=torch.float32,
    )
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])

    del checkpoint
    model.eval()

    prompt_ids = torch.tensor(tokenizer.encode(args.prompt), dtype=torch.long, device=args.device)
    completion = decode(
        model,
        prompt_ids,
        context_length=config["context_length"],
        eos_token_id=tokenizer.token_to_id[b"<|endoftext|>"],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p_target=args.top_p,
    )
    print(args.prompt + tokenizer.decode(completion))


if __name__ == "__main__":
    main()
