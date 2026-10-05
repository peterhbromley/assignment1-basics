"""Run from the project root with: python -m scripts.tokenize --help."""

import argparse
from pathlib import Path

import numpy as np

from llm.tokenizer import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode a text file into a uint16 .npy dataset.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tinystories_bpe.pkl"))
    args = parser.parse_args()

    tokenizer = Tokenizer.from_pickle(args.tokenizer)

    # Read text line by line using the existing iterable encoder. The resulting
    # token array stays in RAM, but avoids building a large Python list of IDs.
    with args.input.open(encoding="utf-8", newline="") as text:
        tokens = np.fromiter(tokenizer.encode_iterable(text), dtype=np.uint16)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        np.save(output, tokens, allow_pickle=False)

    print(f"Saved {tokens.size:,} tokens ({tokens.nbytes:,} bytes of token data) to {args.output}")


if __name__ == "__main__":
    main()
