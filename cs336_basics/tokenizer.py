import os
from collections import Counter

import regex as re


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

    # 3. Capture by gpt2pattern
    # Refer to https://github.com/openai/tiktoken/pull/234/changes
    gpt2_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    words = []
    for chunk in raw_chunks:
        if not chunk or chunk in special_tokens:
            continue
        words.extend(re.findall(gpt2_pattern, chunk))

    # 4. Initialization
    # Basic 256 bytes
    vocab = {i: bytes([i]) for i in range(256)}
    byte_chunks = [list(word.encode("utf-8")) for word in words]

    # 5. Merge
    num_merges = vocab_size - 256 - len(special_tokens)
    merges = []

    # Compute
    for i in range(num_merges):
        counts = Counter()
        for chunk in byte_chunks:
            if len(chunk) < 2:
                continue
            for j in range(len(chunk) - 1):
                counts[(chunk[j], chunk[j + 1])] += 1

        # Exit while Counter is empty
        if not counts:
            break

        best_pair = max(counts, key=lambda x: (counts[x], x))
        new_token_id = 256 + len(special_tokens) + i

        merges.append(best_pair)
        vocab[new_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        new_byte_chunks = []
        for chunk in byte_chunks:
            new_chunk = []
            idx = 0
            while idx < len(chunk):
                if idx < len(chunk) - 1 and (chunk[idx], chunk[idx + 1]) == best_pair:
                    new_chunk.append(new_token_id)
                    idx += 2
                else:
                    new_chunk.append(chunk[idx])
                    idx += 1
            new_byte_chunks.append(new_chunk)
        byte_chunks = new_byte_chunks

    final_merges = [(vocab[p[0]], vocab[p[1]]) for p in merges]

    return vocab, final_merges
