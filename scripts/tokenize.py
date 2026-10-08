"""Run from the project root with: python -m scripts.tokenize --help."""

import argparse
import cProfile
import pickle
import pstats
from pathlib import Path
from time import perf_counter

import numpy as np

from llm.constants import END_OF_TEXT_TOKEN
from llm.tokenizer import Tokenizer, train_bpe


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer or encode text with a saved tokenizer.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Encode the input into this uint16 .npy file.")
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tinystories_bpe.pkl"))
    parser.add_argument("--train", action="store_true", help="Train and save to --tokenizer instead of loading it.")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size when training.")
    parser.add_argument("--profile", action="store_true", help="Profile training in the main process.")
    args = parser.parse_args()

    if not args.train and args.output is None:
        parser.error("--output is required unless --train is specified")
    if args.profile and not args.train:
        parser.error("--profile requires --train")
    if args.train and args.vocab_size < 257:
        parser.error("--vocab-size must be at least 257 (256 bytes plus the end-of-text token)")
    if args.train and args.output is not None and args.vocab_size > 65536:
        parser.error("uint16 output supports a vocabulary size of at most 65536")

    if args.train:
        special_tokens = [END_OF_TEXT_TOKEN]
        profile = cProfile.Profile() if args.profile else None
        print(f"Training on {args.input} with vocabulary size {args.vocab_size:,}...", flush=True)
        start = perf_counter()
        if profile is not None:
            profile.enable()
        vocab, merges = train_bpe(str(args.input), args.vocab_size, special_tokens)
        if profile is not None:
            profile.disable()
        print(f"Training took {perf_counter() - start:.2f}s", flush=True)

        args.tokenizer.parent.mkdir(parents=True, exist_ok=True)
        with args.tokenizer.open("wb") as output:
            pickle.dump({"vocab": vocab, "merges": merges, "special_tokens": special_tokens}, output)
        print(f"Saved tokenizer to {args.tokenizer}")
        if profile is not None:
            pstats.Stats(profile).sort_stats("cumulative").print_stats(20)
        tokenizer = Tokenizer(vocab, merges, special_tokens)
    else:
        tokenizer = Tokenizer.from_pickle(args.tokenizer)

    if args.output is None:
        return

    # Read text line by line using the existing iterable encoder. The resulting
    # token array stays in RAM, but avoids building a large Python list of IDs.
    start = perf_counter()
    with args.input.open(encoding="utf-8", newline="") as text:
        tokens = np.fromiter(tokenizer.encode_iterable(text), dtype=np.uint16)
    elapsed = perf_counter() - start

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        np.save(output, tokens, allow_pickle=False)

    print(f"Saved {tokens.size:,} tokens ({tokens.nbytes:,} bytes of token data) to {args.output}")
    print(f"Encoding took {elapsed:.2f}s ({args.input.stat().st_size / elapsed:,.0f} bytes/second)")


if __name__ == "__main__":
    main()
