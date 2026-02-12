import os
from collections import Counter, defaultdict

import regex as re

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def bytes2unicode():
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapping = {b: chr(b) for b in printable}
    i = 0
    for x in range(256):
        if x not in mapping:
            mapping[x] = chr(i + 256)
            i += 1
    return mapping


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocab = {i: bytes([i]) for i in range(256)}
    num_merges = vocab_size - 256 - len(special_tokens)

    with open(input_path, encoding="utf-8") as file:
        text = file.read()

    if special_tokens:
        special_regex = "|".join(re.escape(token) for token in special_tokens)
        parts = re.split(f"({special_regex})", text)

        train_segments = [p for p in parts if p not in special_tokens]
    else:
        train_segments = [text]

    coarse_count = Counter()

    for segment in train_segments:
        for word in PAT.finditer(segment):
            coarse_count[tuple(bytes([byte]) for byte in word.group().encode("utf-8"))] += 1

    words_list = []
    count_list = []

    for word, freq in coarse_count.items():
        words_list.append(list(word))
        count_list.append(freq)

    freq_merges = defaultdict(int)
    idx_merges = defaultdict(set)

    for idx, word in enumerate(words_list):
        freq = count_list[idx]
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            freq_merges[pair] += freq
            idx_merges[pair].add(idx)

    merges = []

    for _ in range(num_merges):
        if not freq_merges:
            break

        most_pair = max(freq_merges.items(), key=lambda x: (x[1], x[0]))[0]

        if freq_merges[most_pair] <= 0:
            break

        merges.append(most_pair)
        new_token = most_pair[0] + most_pair[1]

        rel_idx = list(idx_merges[most_pair])

        for idx in rel_idx:
            word = words_list[idx]
            freq = count_list[idx]

            i = 0
            while i < len(word) - 1:
                if word[i] == most_pair[0] and word[i + 1] == most_pair[1]:
                    if i > 0:
                        prev_pair = (word[i - 1], word[i])
                        freq_merges[prev_pair] -= freq
                        if freq_merges == 0:
                            del freq_merges[prev_pair]

                    if i < len(word) - 2:
                        next_pair = (word[i + 1], word[i + 2])
                        freq_merges[next_pair] -= freq
                        if freq_merges == 0:
                            del freq_merges[next_pair]

                    word[i] = new_token
                    del word[i + 1]

                    if i > 0:
                        prev_pair = (word[i - 1], word[i])
                        freq_merges[prev_pair] += freq
                        idx_merges[prev_pair].add(idx)

                    if i < len(word) - 1:
                        next_pair = (word[i], word[i + 1])
                        freq_merges[next_pair] += freq
                        idx_merges[next_pair].add(idx)
                else:
                    i += 1

        if most_pair in freq_merges:
            del freq_merges[most_pair]
        if most_pair in idx_merges:
            del idx_merges[most_pair]

    for pair in merges:
        new_idx = len(vocab)
        vocab[new_idx] = pair[0] + pair[1]

    for spec_token in special_tokens:
        spec_bytes = spec_token.encode("utf-8")
        vocab[len(vocab)] = spec_bytes

    return vocab, merges
