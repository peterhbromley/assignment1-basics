from collections import Counter, defaultdict
from multiprocessing import Pool
import os
import pickle
import regex as re
from typing import BinaryIO
from collections.abc import Iterable, Iterator

from llm.constants import END_OF_TEXT_TOKEN

# Regex-based pre-tokenizer from github.com/openai/tiktoken/pull/234/files
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize(text: str) -> dict[tuple[bytes, ...], int]:
    matches = re.finditer(PAT, text)
    freqs = Counter()
    for m in matches:
        m_bytes = tuple(bytes([b]) for b in m.group().encode("utf-8"))
        freqs[m_bytes] += 1
    return freqs


def pretokenize_chunk(path: str, start: int, end: int, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    if not special_tokens:
        return pretokenize(chunk)

    splits = re.split("|".join([re.escape(t) for t in special_tokens]), chunk)
    freqs = Counter()
    for text in splits:
        freqs.update(pretokenize(text))
    return freqs


def run_pretokenizer(path: str, num_processes: int, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, END_OF_TEXT_TOKEN.encode("utf-8"))
        start_end_pairs = [(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(
            pretokenize_chunk,
            [(path, start, end, special_tokens) for start, end in start_end_pairs],
        )

    freqs = Counter()
    for result in results:
        freqs.update(result)

    return freqs


def train_bpe(
    path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab_init_bytes = [bytes([i]) for i in range(256)] + [s.encode("utf-8") for s in special_tokens]
    vocab = {i: b for (i, b) in enumerate(vocab_init_bytes)}
    merges = []

    pretokens = run_pretokenizer(path, 8, special_tokens)

    byte_pair_counts = Counter()
    byte_pair_to_words = defaultdict(set)

    # Initialize pair counts and pair-to-word mapping (one full pass through pretokens)
    for word in pretokens:
        for l, r in list(zip(word, word[1:])):
            pair = (l, r)
            byte_pair_counts[pair] += pretokens[word]
            byte_pair_to_words[pair].add(word)

    while len(vocab) < vocab_size:
        # pair to merge is max count, ties broken lexicographically.
        # so key is count first, then the pair itself
        merge_pair = max(byte_pair_counts, key=lambda pair: (byte_pair_counts[pair], pair))
        l, r = merge_pair
        merged = l + r

        vocab[len(vocab)] = merged
        merges.append(merge_pair)

        # For each word that contains the merge pair, update the counts for adjacent
        # tokens and perform the merge.
        affected_words = byte_pair_to_words[merge_pair].copy()
        for affected_word in affected_words:
            affected_word_freq = pretokens.pop(affected_word)

            # Remove word counts and word mappings for the affected word
            for pair in list(zip(affected_word, affected_word[1:])):
                byte_pair_counts[pair] -= affected_word_freq
                byte_pair_to_words[pair].discard(affected_word)

            # Apply the merge
            new_word = []
            i = 0
            while i < len(affected_word):
                # If we are at the merge pair, then merge and append to the new word
                # Else keep the same and move on
                if i + 1 < len(affected_word) and affected_word[i] == l and affected_word[i + 1] == r:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(affected_word[i])
                    i += 1

            # Update the pretokens with merged word
            new_word = tuple(new_word)
            pretokens[new_word] = affected_word_freq

            for pair in list(zip(new_word, new_word[1:])):
                byte_pair_counts[pair] += pretokens[new_word]
                byte_pair_to_words[pair].add(new_word)

    return vocab, merges


def merge_pretoken(
    pretoken: tuple[bytes, ...], pair_to_merge_rank: dict[tuple[bytes, bytes], int]
) -> tuple[bytes, ...]:
    word = list(pretoken)
    done_merging = False
    while not done_merging:
        best_pair = min(
            (pair for pair in zip(word, word[1:]) if pair in pair_to_merge_rank),
            key=pair_to_merge_rank.get,
            default=None,
        )
        if best_pair is None:
            done_merging = True
        else:
            l, r = best_pair
            merged = l + r
            new_word = []
            i = 0
            while i < len(word):
                # If we match the merge pair, then merge and append to the new word
                if i + 1 < len(word) and word[i] == l and word[i + 1] == r:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
    return tuple(word)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens

        # Reverse mapping of vocab for quick lookups
        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}

        # Make a map from merge pair to merge rank to quickly lookup the priority
        # of an adjacent pair candidate
        self.pair_to_merge_rank = {pair: i for (i, pair) in enumerate(self.merges)}

        # Add any non-present special tokens
        if self.special_tokens:
            next_id = max(vocab) + 1
            for special_token in self.special_tokens:
                as_bytes = special_token.encode("utf-8")
                if as_bytes not in self.token_to_id:
                    self.vocab[next_id] = as_bytes
                    self.token_to_id[as_bytes] = next_id
                    next_id += 1

    @classmethod
    def from_pickle(cls, path: str):
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)

        return cls(
            vocab=data["vocab"],
            merges=data["merges"],
            special_tokens=data["special_tokens"],
        )

    def encode(self, text: str) -> list[int]:
        # Splits and retains special tokens
        if self.special_tokens:
            pattern = (
                "("
                + "|".join(
                    [
                        re.escape(t)
                        # Sort special tokens by length so that overlapping tokens
                        # (e.g. if we have ["<|endoftext|>", "<|endoftext|><|endoftext|>"])
                        # are accounted for before singletons
                        for t in sorted(self.special_tokens, key=len, reverse=True)
                    ]
                )
                + ")"
            )
            blocks = [s for s in re.split(pattern, text) if s != ""]
        else:
            blocks = [text]

        encoded = []
        for chunk in blocks:
            if self.special_tokens and chunk in self.special_tokens:
                encoded.append(self.token_to_id[chunk.encode("utf-8")])
                continue

            matches = re.finditer(PAT, chunk)
            for m in matches:
                pretoken = tuple(bytes([b]) for b in m.group().encode("utf-8"))
                merged = merge_pretoken(pretoken, self.pair_to_merge_rank)
                for token in merged:
                    encoded.append(self.token_to_id[token])

        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        out = b"".join([self.vocab[id] for id in ids])
        return out.decode("utf-8", errors="replace")


if __name__ == "__main__":
    import cProfile
    import pickle
    import pstats
    import random
    from pathlib import Path
    from time import perf_counter

    # Train and profile tokenizer on TinyStories
    start = perf_counter()
    with cProfile.Profile() as profile:
        vocab, merges = train_bpe(
            "data/TinyStoriesV2-GPT4-train.txt",
            10_000,
            [END_OF_TEXT_TOKEN],
        )
    print(f"Training took {perf_counter() - start:.2f}s")
    pstats.Stats(profile).sort_stats("cumulative").print_stats(20)

    # Save trained tokenizer
    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/tinystories_bpe.pkl", "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges, "special_tokens": [END_OF_TEXT_TOKEN]}, f)
    print("Saved tokenizer to artifacts/tinystories_bpe.pkl")

    # Run tokenizer on TinyStories sample docs, estimate compression ratio and
    # tokenization throughput
    documents = Path("data/TinyStoriesV2-GPT4-valid.txt").read_text(encoding="utf-8").split(END_OF_TEXT_TOKEN)
    documents = [doc for doc in documents if doc.strip()]
    sample = random.Random(42).sample(documents, 10)

    tokenizer = Tokenizer(vocab, merges, [END_OF_TEXT_TOKEN])
    total_bytes = sum(len(doc.encode("utf-8")) for doc in sample)

    start = perf_counter()
    total_tokens = sum(len(tokenizer.encode(doc)) for doc in sample)
    elapsed = perf_counter() - start
    bytes_per_second = total_bytes / elapsed

    print(f"TinyStories: {total_bytes / total_tokens:.2f} bytes/token")
    print(f"Encoded {total_bytes:,} bytes into {total_tokens:,} tokens in {elapsed:.3f}s")
    print(f"Throughput: {bytes_per_second:,.0f} bytes/second")
    print(f"Estimated time for 825 GB: {825e9 / bytes_per_second / 3600:.1f} hours")
