import os
from collections import Counter, defaultdict
from multiprocessing import Pool

import regex as re

# Refer to https://github.com/openai/tiktoken/pull/234/changes
GPT2_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def _pretokenize_chunk(chunk: str) -> list[str]:
    return [m.group() for m in GPT2_PATTERN.finditer(chunk)]


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1. Read the text file
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    # 2. Split by special_tokens
    if special_tokens:
        special_pattern = "|".join(re.escape(t) for t in special_tokens)
        raw_chunks = re.split(f"({special_pattern})", text)
    else:
        raw_chunks = [text]

    # 3. Pretokenization
    special_set = set(special_tokens)
    chunks_to_process = [c for c in raw_chunks if c and c not in special_set]

    with Pool() as pool:
        results = pool.map(_pretokenize_chunk, chunks_to_process)

    words = []
    for result in results:
        words.extend(result)

    # 4. Initialization
    # Basic 256 bytes
    vocab = {i: bytes([i]) for i in range(256)}
    # Add special tokens to vocab
    for idx, token in enumerate(special_tokens):
        vocab[256 + idx] = token.encode("utf-8")
    byte_chunks = [list(word.encode("utf-8")) for word in words]

    # 5. Merge
    num_merges = vocab_size - 256 - len(special_tokens)
    merges = []

    counts = Counter()
    pair_to_chunks = defaultdict(set)  # pair -> set of chunk indices containing it

    for ci, chunk in enumerate(byte_chunks):
        for j in range(len(chunk) - 1):
            pair = (chunk[j], chunk[j + 1])
            counts[pair] += 1
            pair_to_chunks[pair].add(ci)

    # 6. Compute
    for i in range(num_merges):
        if not counts:
            break

        best_pair = max(counts, key=lambda p: (counts[p], vocab[p[0]], vocab[p[1]]))
        new_token_id = 256 + len(special_tokens) + i

        merges.append(best_pair)
        vocab[new_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        a, b = best_pair

        # Only process chunks that contain the best pair (skip all others)
        affected = list(pair_to_chunks.get(best_pair, []))

        for ci in affected:
            chunk = byte_chunks[ci]

            # Remove old pairs from counts and index
            for j in range(len(chunk) - 1):
                pair = (chunk[j], chunk[j + 1])
                counts[pair] -= 1
                if counts[pair] <= 0:
                    del counts[pair]
                if pair in pair_to_chunks:
                    pair_to_chunks[pair].discard(ci)
                    if not pair_to_chunks[pair]:
                        del pair_to_chunks[pair]

            # Build merged chunk
            new_chunk = []
            j = 0
            while j < len(chunk):
                if j < len(chunk) - 1 and chunk[j] == a and chunk[j + 1] == b:
                    new_chunk.append(new_token_id)
                    j += 2
                else:
                    new_chunk.append(chunk[j])
                    j += 1

            # Add new pairs to counts and index
            for j in range(len(new_chunk) - 1):
                pair = (new_chunk[j], new_chunk[j + 1])
                counts[pair] += 1
                pair_to_chunks[pair].add(ci)

            byte_chunks[ci] = new_chunk

    final_merges = [(vocab[p[0]], vocab[p[1]]) for p in merges]

    return vocab, final_merges
