from collections import Counter, defaultdict

from llm.constants import END_OF_TEXT_TOKEN
from llm.pretokenization import run_pretokenizer


def train_bpe(
    path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab_init_bytes = (
        [bytes([i]) for i in range(256)] + [s.encode("utf-8") for s in special_tokens]
    )
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
                if i+1 < len(affected_word) and affected_word[i] == l and affected_word[i+1] == r:
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


if __name__ == "__main__":
    import cProfile
    import pickle
    import pstats
    from pathlib import Path
    from time import perf_counter

    start = perf_counter()
    with cProfile.Profile() as profile:
        vocab, merges = train_bpe(
            "data/TinyStoriesV2-GPT4-train.txt",
            10_000,
            [END_OF_TEXT_TOKEN],
        )
    print(f"Training took {perf_counter() - start:.2f}s")
    pstats.Stats(profile).sort_stats("cumulative").print_stats(20)

    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/tinystories_bpe.pkl", "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges, "special_tokens": [END_OF_TEXT_TOKEN]}, f)
    print("Saved tokenizer to artifacts/tinystories_bpe.pkl")
