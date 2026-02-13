import heapq
import os
import multiprocessing as mp
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator

import regex as re

# Refer to https://github.com/openai/tiktoken/pull/234/changes
GPT2_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def _iter_documents_single(input_path: str | os.PathLike, token: str, chunk_size: int) -> Iterator[str]:
    buffer = ""
    token_len = len(token)
    with open(input_path, encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            start = 0
            while True:
                idx = buffer.find(token, start)
                if idx == -1:
                    break
                yield buffer[start:idx]
                start = idx + token_len
            buffer = buffer[start:]
    if buffer:
        yield buffer


def _iter_documents_multi(input_path: str | os.PathLike, special_tokens: list[str], chunk_size: int) -> Iterator[str]:
    special_pattern = "|".join(re.escape(t) for t in special_tokens)
    special_regex = re.compile(special_pattern)
    buffer = ""
    with open(input_path, encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            last_idx = 0
            for match in special_regex.finditer(buffer):
                yield buffer[last_idx : match.start()]
                last_idx = match.end()
            buffer = buffer[last_idx:]
    if buffer:
        yield buffer


def _batch_documents(doc_iter: Iterable[str], max_docs: int = 2048, max_chars: int = 1 << 20) -> Iterator[list[str]]:
    batch: list[str] = []
    total_chars = 0
    for doc in doc_iter:
        if not doc:
            continue
        batch.append(doc)
        total_chars += len(doc)
        if len(batch) >= max_docs or total_chars >= max_chars:
            yield batch
            batch = []
            total_chars = 0
    if batch:
        yield batch


def _count_docs(docs: list[str]) -> Counter[bytes]:
    counts: Counter[bytes] = Counter()
    pattern = GPT2_PATTERN
    for doc in docs:
        for match in pattern.finditer(doc):
            counts[match.group().encode("utf-8")] += 1
    return counts


def _num_workers() -> int:
    env = os.getenv("BPE_NUM_WORKERS")
    if env:
        try:
            value = int(env)
            return max(1, value)
        except ValueError:
            pass
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1-3. Stream pretokenization into counts; avoid storing the full text
    word_counts: Counter[bytes] = Counter()
    chunk_size = 1 << 20

    if special_tokens:
        if len(special_tokens) == 1:
            doc_iter = _iter_documents_single(input_path, special_tokens[0], chunk_size)
        else:
            doc_iter = _iter_documents_multi(input_path, special_tokens, chunk_size)

        batches = _batch_documents(doc_iter)
        workers = _num_workers()
        if workers <= 1:
            for batch in batches:
                word_counts.update(_count_docs(batch))
        else:
            try:
                ctx = mp.get_context("fork")
            except ValueError:
                ctx = mp.get_context()
            with ctx.Pool(processes=workers) as pool:
                for counts in pool.imap_unordered(_count_docs, batches, chunksize=1):
                    word_counts.update(counts)
    else:

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
    def _inv_key(token_bytes: bytes) -> tuple[int, ...]:
        # Reverse lexicographic order with an explicit terminator so prefixes sort after longer bytes.
        return tuple(255 - b for b in token_bytes) + (256,)

    vocab = {i: bytes([i]) for i in range(256)}
    inv_vocab = [_inv_key(bytes([i])) for i in range(256)]
    # Add special tokens to vocab
    for idx, token in enumerate(special_tokens):
        token_bytes = token.encode("utf-8")
        vocab[256 + idx] = token_bytes
        inv_vocab.append(_inv_key(token_bytes))

    # Collapse duplicate words to avoid per-occurrence storage
    word_symbols: list[list[int]] = []
    word_freqs: list[int] = []
    for word_bytes, freq in word_counts.items():
        word_symbols.append(list(word_bytes))
        word_freqs.append(freq)
    word_next: list[list[int]] = []
    word_prev: list[list[int]] = []
    word_alive: list[list[bool]] = []
    for symbols in word_symbols:
        n = len(symbols)
        if n == 0:
            word_next.append([])
            word_prev.append([])
            word_alive.append([])
            continue
        word_next.append([i + 1 for i in range(n - 1)] + [-1])
        word_prev.append([-1] + [i for i in range(n - 1)])
        word_alive.append([True] * n)

    # 5. Merge
    num_merges = vocab_size - 256 - len(special_tokens)
    merges = []

    counts: Counter[tuple[int, int]] = Counter()
    pair_to_occurrences: defaultdict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)

    for wi, symbols in enumerate(word_symbols):
        freq = word_freqs[wi]
        for j in range(len(symbols) - 1):
            pair = (symbols[j], symbols[j + 1])
            counts[pair] += freq
            pair_to_occurrences[pair].append((wi, j))

    heap: list[tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, int]]] = []
    for pair, count in counts.items():
        heapq.heappush(
            heap,
            (-count, inv_vocab[pair[0]], inv_vocab[pair[1]], pair),
        )

    def _pop_best_pair() -> tuple[int, int] | None:
        while heap:
            neg_count, _inv_a, _inv_b, pair = heapq.heappop(heap)
            current = counts.get(pair)
            if current is None:
                continue
            if -neg_count != current:
                continue
            return pair
        return None

    def _adjust_pair(pair: tuple[int, int], delta: int) -> None:
        if delta == 0:
            return
        new_count = counts.get(pair, 0) + delta
        if new_count <= 0:
            if pair in counts:
                del counts[pair]
            return
        counts[pair] = new_count
        heapq.heappush(heap, (-new_count, inv_vocab[pair[0]], inv_vocab[pair[1]], pair))

    # 6. Compute
    for i in range(num_merges):
        if not counts:
            break
        best_pair = _pop_best_pair()
        if best_pair is None:
            break
        new_token_id = 256 + len(special_tokens) + i

        merges.append(best_pair)
        vocab[new_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        inv_vocab.append(_inv_key(vocab[new_token_id]))

        a, b = best_pair

        # Only process words that contain the best pair (skip all others)
        occurrences = pair_to_occurrences.pop(best_pair, [])
        affected_positions: dict[int, list[int]] = {}
        for wi, pos in occurrences:
            if not word_alive[wi][pos]:
                continue
            j = word_next[wi][pos]
            if j == -1:
                continue
            if word_symbols[wi][pos] != a or word_symbols[wi][j] != b:
                continue
            if word_prev[wi][j] != pos:
                continue
            affected_positions.setdefault(wi, []).append(pos)

        for wi, positions in affected_positions.items():
            positions.sort()
            freq = word_freqs[wi]
            for pos in positions:
                if not word_alive[wi][pos]:
                    continue
                j = word_next[wi][pos]
                if j == -1:
                    continue
                if word_symbols[wi][pos] != a or word_symbols[wi][j] != b:
                    continue
                if word_prev[wi][j] != pos:
                    continue

                prev_idx = word_prev[wi][pos]
                next_idx = word_next[wi][j]

                # Remove old pairs from counts
                _adjust_pair((a, b), -freq)
                if prev_idx != -1:
                    _adjust_pair((word_symbols[wi][prev_idx], a), -freq)
                if next_idx != -1:
                    _adjust_pair((b, word_symbols[wi][next_idx]), -freq)

                # Merge nodes pos and j
                word_symbols[wi][pos] = new_token_id
                word_alive[wi][j] = False
                word_prev[wi][j] = -1
                word_next[wi][j] = -1
                word_next[wi][pos] = next_idx
                if next_idx != -1:
                    word_prev[wi][next_idx] = pos

                # Add new pairs to counts and index
                if prev_idx != -1:
                    new_left = (word_symbols[wi][prev_idx], new_token_id)
                    _adjust_pair(new_left, freq)
                    pair_to_occurrences[new_left].append((wi, prev_idx))
                if next_idx != -1:
                    new_right = (new_token_id, word_symbols[wi][next_idx])
                    _adjust_pair(new_right, freq)
                    pair_to_occurrences[new_right].append((wi, pos))

    final_merges = [(vocab[p[0]], vocab[p[1]]) for p in merges]

    return vocab, final_merges
