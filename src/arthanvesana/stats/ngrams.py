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
    def __init__(
        self,
        train: list[list[str]],
        n: int,
        k: float = 1.0,
        method: str = "laplace",
        lam: float = 0.5,
    ):
        self.n = n
        self.k = k
        self.method = method
        self.lam = lam
        self.vocab = {s for seq in train for s in seq} | {"<UNK>"}
        self.vsize = len(self.vocab)
        padded = [["<S>"] * (n - 1) + self._map(seq) for seq in train]
        self.orders: list[Counter] = [count_ngrams(padded, i) for i in range(n + 1)]
        self.followers: dict[int, dict[tuple, Counter]] = {}
        for order in range(1, n):
            fmap: dict[tuple, Counter] = {}
            for key, v in self.orders[order + 1].items():
                if key[-1] in self.vocab:
                    fmap.setdefault(key[:-1], Counter())[key[-1]] += v
            self.followers[order] = fmap
        if n == 1:
            self.total = sum(self.orders[1].values())

    def _map(self, seq: list[str]) -> list[str]:
        return [s if s in self.vocab else "<UNK>" for s in seq]

    def dist(self, context: tuple) -> dict:
        if self.method == "interp":
            return self._dist_interp(self.n, context)
        if self.method == "wittenbell":
            return self._dist_wb(self.n, context)
        return self._dist_laplace(self.n, context)

    def _dist_laplace(self, order: int, context: tuple) -> dict:
        if order == 1:
            denom = self.total + self.k * self.vsize
            return {
                w: (self.orders[1].get((w,), 0) + self.k) / denom
                for w in self.vocab
            }
        fc = self.followers[order - 1].get(context, Counter())
        total = sum(fc.values())
        denom = total + self.k * self.vsize
        return {w: (fc.get(w, 0) + self.k) / denom for w in self.vocab}

    def _dist_wb(self, order: int, context: tuple) -> dict:
        if order == 1:
            counts = {w: self.orders[1].get((w,), 0) for w in self.vocab}
            total = sum(counts.values())
            return self._wb_from_counts(counts, total)
        fc = self.followers[order - 1].get(context, Counter())
        total = sum(fc.values())
        if total == 0:
            return self._dist_wb(order - 1, context[1:])
        return self._wb_from_counts(dict(fc), total)

    def _wb_from_counts(self, counts: dict, total: int) -> dict:
        seen = {w for w, c in counts.items() if c > 0}
        n_types = len(seen)
        n_unseen = self.vsize - n_types
        out = {}
        for w in self.vocab:
            if w in seen:
                out[w] = counts[w] / (total + n_types)
            elif n_unseen:
                out[w] = n_types / (n_unseen * (total + n_types))
            else:
                out[w] = 0.0
        return out

    def _dist_interp(self, order: int, context: tuple) -> dict:
        if order == 1:
            counts = {w: self.orders[1].get((w,), 0) for w in self.vocab}
            total = sum(counts.values())
            if total == 0:
                return {w: 1.0 / self.vsize for w in self.vocab}
            out = {}
            for w in self.vocab:
                ml = counts[w] / total if total else 0.0
                out[w] = self.lam * ml + (1 - self.lam) / self.vsize
            return out
        fc = self.followers[order - 1].get(context, Counter())
        total = sum(fc.values())
        lower = self._dist_interp(order - 1, context[1:])
        if total == 0:
            return lower
        out = {}
        for w in self.vocab:
            ml = fc.get(w, 0) / total if total else 0.0
            out[w] = self.lam * ml + (1 - self.lam) * lower[w]
        return out

    def logprob(self, seq: list[str]) -> float:
        seq = self._map(seq)
        if self.n == 1:
            d = self.dist(())
            return sum(math.log2(d[s]) for s in seq)
        padded = ["<S>"] * (self.n - 1) + seq
        total = 0.0
        for i in range(len(seq)):
            context = tuple(padded[i : i + self.n - 1])
            total += math.log2(self.dist(context)[padded[i + self.n - 1]])
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
