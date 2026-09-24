from collections import Counter
from multiprocessing import Pool
import os
import regex as re
from typing import BinaryIO

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


if __name__ == "__main__":
    path = "data/TinyStoriesV2-GPT4-valid.txt"
    freqs = run_pretokenizer(path, 4, [END_OF_TEXT_TOKEN])
