from __future__ import annotations

import math
from collections import Counter

import numpy as np


def cooccurrence_counts(seqs: list[list[str]], window: int = 2) -> Counter:
    if not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    pairs: Counter = Counter()
    for seq in seqs:
        for i, target in enumerate(seq):
            lo = max(0, i - window)
            hi = min(len(seq), i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    pairs[(target, seq[j])] += 1
    return pairs


def ppmi_matrix(
    pairs: Counter, vocab: list[str], alpha: float = 0.75
) -> np.ndarray:
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be a finite positive number")
    index = {sign: i for i, sign in enumerate(vocab)}
    pair_total = sum(pairs.values())
    target_totals = Counter()
    context_totals = Counter()
    for (target, context), count in pairs.items():
        target_totals[target] += count
        context_totals[context] += count
    smoothed = {s: context_totals.get(s, 0) ** alpha for s in vocab}
    smooth_total = sum(smoothed.values())
    mat = np.zeros((len(vocab), len(vocab)))
    if pair_total == 0 or smooth_total == 0:
        return mat
    for (target, context), count in pairs.items():
        p_joint = count / pair_total
        pmi = math.log2(
            p_joint
            / ((target_totals[target] / pair_total) * (smoothed[context] / smooth_total))
        )
        if pmi > 0:
            mat[index[target], index[context]] = pmi
    return mat


def svd_embeddings(matrix: np.ndarray, dim: int = 50, power: float = 0.5) -> np.ndarray:
    if not isinstance(dim, int) or dim < 1:
        raise ValueError("dim must be a positive integer")
    dim = min(dim, min(matrix.shape))
    _, singular, right = np.linalg.svd(matrix, full_matrices=False)
    return right[:dim].T * (singular[:dim] ** power)


def train_embeddings(
    seqs: list[list[str]], window: int = 2, dim: int = 50,
    alpha: float = 0.75, power: float = 0.5,
) -> dict:
    vocab = sorted({sign for seq in seqs for sign in seq})
    if not vocab:
        raise ValueError("train_embeddings requires nonempty training sequences")
    pairs = cooccurrence_counts(seqs, window)
    matrix = ppmi_matrix(pairs, vocab, alpha)
    dim = min(dim, min(matrix.shape))
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    vectors = right[:dim].T * (singular[:dim] ** power)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return {
        "vocab": vocab,
        "vectors": vectors / norms,
        "word_scores": left[:, :dim] * singular[:dim],
        "context_scores": right[:dim].T,
        "window": window,
        "dim": dim,
        "alpha": alpha,
        "power": power,
    }


def cosine_neighbors(model: dict, sign: str, top_k: int = 10) -> list[tuple[str, float]]:
    if sign not in model["vocab"]:
        raise ValueError(f"sign {sign!r} not in embedding vocabulary")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    idx = model["vocab"].index(sign)
    sims = model["vectors"] @ model["vectors"][idx]
    order = sorted(
        (i for i in range(len(model["vocab"])) if i != idx),
        key=lambda i: (-sims[i], model["vocab"][i]),
    )
    return [(model["vocab"][i], float(sims[i])) for i in order[:top_k]]


def train_skipgram(
    seqs: list[list[str]], window: int = 2, dim: int = 50, seed: int = 0,
    epochs: int = 50,
) -> dict:
    from gensim.models import Word2Vec

    if not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    if not isinstance(dim, int) or dim < 1:
        raise ValueError("dim must be a positive integer")
    train = [list(seq) for seq in seqs if seq]
    if not train:
        raise ValueError("train_skipgram requires nonempty training sequences")
    model = Word2Vec(
        train, vector_size=dim, window=window, min_count=1, sg=1,
        negative=5, epochs=epochs, workers=1, seed=seed,
    )
    vocab = sorted(model.wv.key_to_index)
    vectors = np.array([model.wv[sign] for sign in vocab])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return {
        "vocab": vocab,
        "vectors": vectors / norms,
        "word_scores": np.array([model.syn1neg[model.wv.key_to_index[s]] for s in vocab]),
        "context_scores": np.array([model.wv[s] for s in vocab]),
        "window": window,
        "dim": dim,
        "seed": seed,
    }


def skipgram_restoration_ranks(
    train_records: list, test_records: list, window: int = 2, dim: int = 50,
    seed: int = 0,
) -> list[int | None]:
    train_seqs = [r["sequence"] for r in train_records]
    model = train_skipgram(train_seqs, window, dim, seed)
    index = {sign: i for i, sign in enumerate(model["vocab"])}
    counts = Counter(sign for seq in train_seqs for sign in seq)
    freq_rank = {s: r for r, s in enumerate(sorted(counts, key=lambda s: (-counts[s], s)), 1)}
    word = model["word_scores"]
    context = model["context_scores"]
    ranks = []
    for record in test_records:
        seq = record["sequence"]
        for pos, target in enumerate(seq):
            if target not in index:
                ranks.append(None)
                continue
            lo = max(0, pos - window)
            context_idx = [
                index[s] for k, s in enumerate(seq[lo:pos + window + 1])
                if lo + k != pos and s in index
            ]
            if not context_idx:
                ranks.append(freq_rank.get(target))
                continue
            query = context[context_idx].mean(axis=0)
            if not np.any(query):
                ranks.append(freq_rank.get(target))
                continue
            scores = word @ query
            ordered = sorted(
                range(len(model["vocab"])),
                key=lambda i: (-scores[i], model["vocab"][i]),
            )
            ranks.append(ordered.index(index[target]) + 1)
    return ranks


def neighbor_jaccard(first: list, second: list) -> float:
    a, b = set(first), set(second)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def embedding_restoration_ranks(
    train_records: list, test_records: list, window: int = 2, dim: int = 50,
    alpha: float = 0.75, power: float = 0.5,
) -> list[int | None]:
    train_seqs = [r["sequence"] for r in train_records]
    model = train_embeddings(train_seqs, window, dim, alpha, power)
    index = {sign: i for i, sign in enumerate(model["vocab"])}
    counts = Counter(sign for seq in train_seqs for sign in seq)
    freq_rank = {s: r for r, s in enumerate(sorted(counts, key=lambda s: (-counts[s], s)), 1)}
    word = model["word_scores"]
    context = model["context_scores"]
    ranks = []
    for record in test_records:
        seq = record["sequence"]
        for pos, target in enumerate(seq):
            if target not in index:
                ranks.append(None)
                continue
            lo = max(0, pos - window)
            context_idx = [
                index[s] for k, s in enumerate(seq[lo:pos + window + 1])
                if lo + k != pos and s in index
            ]
            if not context_idx:
                ranks.append(freq_rank.get(target))
                continue
            query = context[context_idx].mean(axis=0)
            if not np.any(query):
                ranks.append(freq_rank.get(target))
                continue
            scores = word @ query
            ordered = sorted(
                range(len(model["vocab"])),
                key=lambda i: (-scores[i], model["vocab"][i]),
            )
            ranks.append(ordered.index(index[target]) + 1)
    return ranks
