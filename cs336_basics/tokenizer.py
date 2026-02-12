import os
from collections import Counter, defaultdict

import regex as re

# Refer to https://github.com/openai/tiktoken/pull/234/changes
GPT2_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1-3. Stream pretokenization into counts; avoid storing the full text
    word_counts: Counter[bytes] = Counter()

    def _consume_segment(segment: str, carry: str, flush: bool) -> str:
        if not segment and not carry:
            return ""
        buffer = carry + segment
        last_match = None
        for match in GPT2_PATTERN.finditer(buffer):
            if last_match is not None:
                word_counts[last_match.group().encode("utf-8")] += 1
            last_match = match
        if last_match is None:
            return buffer if not flush else ""
        if flush or last_match.end() < len(buffer):
            word_counts[last_match.group().encode("utf-8")] += 1
            return "" if last_match.end() == len(buffer) else buffer[last_match.end() :]
        return buffer[last_match.start() :]

    chunk_size = 1 << 20
    if special_tokens:
        special_pattern = "|".join(re.escape(t) for t in special_tokens)
        special_regex = re.compile(special_pattern)
        max_special_len = max(len(t) for t in special_tokens)
        buffer = ""
        carry = ""
        with open(input_path, encoding="utf-8") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                buffer += chunk
                scan_upto = len(buffer) - (max_special_len - 1)
                if scan_upto <= 0:
                    continue
                last_idx = 0
                for match in special_regex.finditer(buffer):
                    if match.start() >= scan_upto:
                        break
                    if match.start() > last_idx:
                        carry = _consume_segment(buffer[last_idx:match.start()], carry, flush=False)
                    carry = _consume_segment("", carry, flush=True)
                    last_idx = match.end()
                if last_idx > 0:
                    buffer = buffer[last_idx:]
                else:
                    carry = _consume_segment(buffer[:scan_upto], carry, flush=False)
                    buffer = buffer[scan_upto:]
        if buffer:
            carry = _consume_segment(buffer, carry, flush=False)
        if carry:
            carry = _consume_segment("", carry, flush=True)
    else:
        carry = ""
        with open(input_path, encoding="utf-8") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                carry = _consume_segment(chunk, carry, flush=False)
        if carry:
            carry = _consume_segment("", carry, flush=True)

    # 4. Initialization
    # Basic 256 bytes
    vocab = {i: bytes([i]) for i in range(256)}
    # Add special tokens to vocab
    for idx, token in enumerate(special_tokens):
        vocab[256 + idx] = token.encode("utf-8")

    # Collapse duplicate words to avoid per-occurrence storage
    word_symbols: list[list[int]] = []
    word_freqs: list[int] = []
    for word_bytes, freq in word_counts.items():
        word_symbols.append(list(word_bytes))
        word_freqs.append(freq)

    # 5. Merge
    num_merges = vocab_size - 256 - len(special_tokens)
    merges = []

    counts: Counter[tuple[int, int]] = Counter()
    pair_to_words: defaultdict[tuple[int, int], set[int]] = defaultdict(set)

    for wi, symbols in enumerate(word_symbols):
        freq = word_freqs[wi]
        for j in range(len(symbols) - 1):
            pair = (symbols[j], symbols[j + 1])
            counts[pair] += freq
            pair_to_words[pair].add(wi)

    # 6. Compute
    for i in range(num_merges):
        if not counts:
            break

        best_pair = max(counts, key=lambda p: (counts[p], vocab[p[0]], vocab[p[1]]))
        new_token_id = 256 + len(special_tokens) + i

        merges.append(best_pair)
        vocab[new_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        a, b = best_pair

        # Only process words that contain the best pair (skip all others)
        affected = list(pair_to_words.get(best_pair, []))

        for wi in affected:
            symbols = word_symbols[wi]
            freq = word_freqs[wi]

            # Remove old pairs from counts and index
            for j in range(len(symbols) - 1):
                pair = (symbols[j], symbols[j + 1])
                counts[pair] -= freq
                if counts[pair] <= 0:
                    del counts[pair]
                if pair in pair_to_words:
                    pair_to_words[pair].discard(wi)
                    if not pair_to_words[pair]:
                        del pair_to_words[pair]

            # Build merged chunk
            new_symbols = []
            j = 0
            while j < len(symbols):
                if j < len(symbols) - 1 and symbols[j] == a and symbols[j + 1] == b:
                    new_symbols.append(new_token_id)
                    j += 2
                else:
                    new_symbols.append(symbols[j])
                    j += 1

            # Add new pairs to counts and index
            for j in range(len(new_symbols) - 1):
                pair = (new_symbols[j], new_symbols[j + 1])
                counts[pair] += freq
                pair_to_words[pair].add(wi)

            word_symbols[wi] = new_symbols

    final_merges = [(vocab[p[0]], vocab[p[1]]) for p in merges]

    return vocab, final_merges
