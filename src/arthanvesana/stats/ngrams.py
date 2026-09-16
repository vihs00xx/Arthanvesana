from __future__ import annotations

import math
from collections import Counter


def count_ngrams(seqs: list[list[str]], n: int) -> Counter:
    counts: Counter = Counter()
    for seq in seqs:
        for i in range(len(seq) - n + 1):
            counts[tuple(seq[i : i + n])] += 1
    return counts


class NGramModel:
    def __init__(self, train: list[list[str]], n: int, k: float = 1.0):
        self.n = n
        self.k = k
        self.vocab = {s for seq in train for s in seq} | {"<UNK>"}
        self.vsize = len(self.vocab)
        padded = [["<S>"] * (n - 1) + self._map(seq) for seq in train]
        self.ngram_counts = count_ngrams(padded, n)
        self.context_counts = count_ngrams(padded, n - 1) if n > 1 else Counter()
        if n == 1:
            self.total = sum(self.ngram_counts.values())

    def _map(self, seq: list[str]) -> list[str]:
        return [s if s in self.vocab else "<UNK>" for s in seq]

    def logprob(self, seq: list[str]) -> float:
        seq = self._map(seq)
        if self.n == 1:
            denom = self.total + self.k * self.vsize
            return sum(
                math.log2((self.ngram_counts.get((s,), 0) + self.k) / denom)
                for s in seq
            )
        padded = ["<S>"] * (self.n - 1) + seq
        total = 0.0
        for i in range(len(seq)):
            ngram = tuple(padded[i : i + self.n])
            context = ngram[:-1]
            c_ng = self.ngram_counts.get(ngram, 0)
            c_ctx = self.context_counts.get(context, 0)
            total += math.log2((c_ng + self.k) / (c_ctx + self.k * self.vsize))
        return total

    def perplexity(self, test: list[list[str]]) -> float:
        log_sum = 0.0
        n_tokens = 0
        for seq in test:
            log_sum += self.logprob(seq)
            n_tokens += len(seq)
        if n_tokens == 0:
            return float("nan")
        return 2.0 ** (-log_sum / n_tokens)


def top_ngrams(seqs: list[list[str]], n: int, top: int = 20) -> list[tuple]:
    return count_ngrams(seqs, n).most_common(top)
