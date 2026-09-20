"""Compact structural models for the Sindhu-Sarasvatī sign sequences.

Two families, both described by structure rather than linguistic labels:

1. A smoothed relative-position model: the distribution of the sign at each
   (span length, zero-based position) bucket, smoothed toward the unigram.
2. A discrete hidden Markov model with 2-5 states, fitted by Baum-Welch, whose
   states are a compact summary of positional sequence structure. The states are
   NOT words, phrases, grammatical roles, or semantic classes.

Evaluation is grouped cross-validation with held-out log loss as the primary
metric. Log-space arithmetic is used throughout for numerical stability.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from arthanvesana.simulate.pipeline import assign_folds, records_from_seqs
from arthanvesana.stats.ngrams import NGramModel
from arthanvesana.stats.sampling import connected_groups

NEG_INF = -1e30


def _logsumexp(values, axis=None):
    arr = np.asarray(values, dtype=float)
    arr = np.where(arr <= NEG_INF, NEG_INF, arr)
    if axis is None:
        mx = float(np.max(arr))
        if not np.isfinite(mx):
            mx = 0.0
        return float(mx + math.log(float(np.sum(np.exp(arr - mx)))))
    mx = np.max(arr, axis=axis, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    shifted = np.exp(arr - mx)
    total = np.sum(shifted, axis=axis, keepdims=True)
    out = mx + np.log(np.maximum(total, 1e-300))
    return np.squeeze(out, axis=axis)


def relative_position_counts(seqs):
    """Counts of sign at each (span_length, position) bucket."""
    counts = Counter()
    for seq in seqs:
        for pos, sign in enumerate(seq):
            counts[(len(seq), pos, sign)] += 1
    return counts


def relative_position_logprob(seqs_fit):
    """Return a callable scoring a sequence's total log2 likelihood.

    Each (length, position) bucket is smoothed toward the empirical unigram so
    that unseen buckets degrade gracefully rather than failing.
    """
    counts = relative_position_counts(seqs_fit)
    unigram = Counter(s for seq in seqs_fit for s in seq)
    total_unigram = sum(unigram.values()) or 1
    vocab = sorted(unigram)
    v = len(vocab) or 1
    buckets = {}
    for (length, pos, sign), count in counts.items():
        buckets.setdefault((length, pos), Counter())[sign] = count
    smoothed = {}
    for key, bucket in buckets.items():
        total = sum(bucket.values()) or 1
        smoothed[key] = {
            s: (bucket.get(s, 0) + 1) / (total + v) for s in vocab
        }
    buckets = smoothed
    unigram_dist = {s: unigram[s] / total_unigram for s in vocab}

    def logprob(seq):
        total = 0.0
        for pos, sign in enumerate(seq):
            if sign not in unigram_dist:
                return None
            key = (len(seq), pos)
            dist = buckets.get(key)
            p = dist[sign] if dist is not None else unigram_dist[sign]
            if p <= 0:
                p = 1e-12
            total += math.log2(p)
        return total

    params = len(buckets) * v + v
    return logprob, {"n_buckets": len(buckets), "vocab_size": v, "n_parameters": params}


def _token_logloss(logprob_fn, seqs, fallback_bits):
    """Mean log2 loss per token; unseen signs take a documented penalty."""
    total = 0.0
    n = 0
    for seq in seqs:
        lp = logprob_fn(seq)
        if lp is None:
            total += fallback_bits * len(seq)
            n += len(seq)
        else:
            total -= lp
            n += len(seq)
    return (total / n) if n else None, n


def crossfit_compact(seqs, n_folds=5, seed=0, state_counts=(2, 3, 4, 5),
                     inner_frac=0.2):
    """Grouped nested cross-validation of compact models vs the bigram.

    For every outer fold: an inner split of the outer TRAIN partition selects the
    HMM state count by inner-validation log loss; the selected state count is
    refit on the full outer train; all models are then scored once on the outer
    test tokens. Returns per-model mean held-out bits/token plus perplexity,
    parameter counts, and the selected state counts.
    """
    records = records_from_seqs(seqs)
    groups = connected_groups(records, track="artifact", group_duplicates=True)
    gid_of = {}
    for g in groups:
        for i in g["indices"]:
            gid_of[i] = g["group_id"]
    assignment = assign_folds(
        {g["group_id"]: (
            sum(len(records[i]["sequence"]) for i in g["indices"]), len(g["indices"]))
         for g in groups},
        n_folds, seed,
    )
    vocab = sorted({s for seq in seqs for s in seq})
    fallback_bits = math.log2(max(2, len(vocab)))

    acc = {name: [0.0, 0] for name in ("bigram", "relative_position", "hmm")}
    selected_states = []
    hmm_params = []
    relpos_params = 0
    for f in range(n_folds):
        train = [records[i]["sequence"] for i in range(len(records))
                 if assignment[gid_of[i]] != f]
        test = [records[i]["sequence"] for i in range(len(records))
                if assignment[gid_of[i]] == f]
        if not train or not test:
            continue
        # --- bigram baseline
        bigram = NGramModel(train, 2, method="mkn")

        def bigram_logprob(seq, _model=bigram):
            mapped = _model._map(seq)
            total = 0.0
            for i, tok in enumerate(mapped):
                ctx = tuple((["<S>"] * (_model.n - 1) + mapped[:i])[-(_model.n - 1):])
                total += math.log2(max(_model.dist(ctx)[tok], 1e-12))
            return total

        loss, ntok = _token_logloss(bigram_logprob, test, fallback_bits)
        acc["bigram"][0] += loss * ntok
        acc["bigram"][1] += ntok
        # --- relative-position model
        relpos, relpos_stats = relative_position_logprob(train)
        loss, ntok = _token_logloss(relpos, test, fallback_bits)
        acc["relative_position"][0] += loss * ntok
        acc["relative_position"][1] += ntok
        relpos_params = relpos_stats["n_parameters"]
        # --- HMM: nested state-count selection on an inner split of train
        inner_fit, inner_valid = _inner_split(train, seed + f, inner_frac)
        best_states, best_loss = state_counts[0], None
        for k in state_counts:
            model = DiscreteHMM(n_states=k, n_iter=10, seed=seed + f).fit(inner_fit)
            loss, _ = _token_logloss(model.logprob, inner_valid, fallback_bits)
            if loss is not None and (best_loss is None or loss < best_loss):
                best_states, best_loss = k, loss
        selected_states.append(best_states)
        hmm = DiscreteHMM(n_states=best_states, n_iter=15, seed=seed + f).fit(train)
        loss, ntok = _token_logloss(hmm.logprob, test, fallback_bits)
        acc["hmm"][0] += loss * ntok
        acc["hmm"][1] += ntok
        hmm_params.append(hmm.n_parameters)

    out = {}
    for name, (total, ntok) in acc.items():
        if ntok:
            bits = total / ntok
            out[name] = {"bits_per_token": bits, "perplexity": 2 ** bits,
                         "n_tokens": ntok}
        else:
            out[name] = {"bits_per_token": None, "perplexity": None, "n_tokens": 0}
    out["selected_hmm_states"] = selected_states
    out["hmm_n_parameters"] = hmm_params
    out["relative_position_n_parameters"] = relpos_params
    out["n_groups"] = len(groups)
    return out


def _ctx(model, seq, i):
    mapped = model._map(seq)
    return tuple((["<S>"] * (model.n - 1) + mapped[:i])[-(model.n - 1):])


def _inner_split(train, seed, frac):
    import random as _random
    order = list(train)
    _random.Random(seed).shuffle(order)
    cut = max(1, int(len(order) * (1 - frac)))
    return order[:cut], order[cut:] or order[:1]


class DiscreteHMM:
    """Discrete HMM fitted by Baum-Welch (EM) in log space.

    The latent states summarize positional sequence structure. They carry no
    linguistic interpretation.
    """

    def __init__(self, n_states=3, n_iter=15, seed=0, smoothing=1e-2):
        self.n_states = n_states
        self.n_iter = n_iter
        self.seed = seed
        self.smoothing = smoothing
        self.vocab: list[str] = []
        self.index: dict[str, int] = {}
        self.log_pi = None
        self.log_a = None
        self.log_b = None
        self.n_parameters = 0

    def _emissions(self, seqs):
        idx = self.index
        rows = []
        for seq in seqs:
            rows.append(np.array([idx[s] for s in seq if s in idx], dtype=int))
        return [row for row in rows if row.size]

    def fit(self, seqs):
        seqs = [s for s in seqs if s]
        if not seqs:
            raise ValueError("DiscreteHMM requires nonempty training sequences")
        self.vocab = sorted({s for seq in seqs for s in seq})
        self.index = {s: i for i, s in enumerate(self.vocab)}
        v = len(self.vocab)
        n = max(1, min(self.n_states, v))
        rng = np.random.default_rng(self.seed)
        log_pi = np.log(np.ones(n) / n)
        log_a = np.log(np.ones((n, n)) / n)
        log_b = np.log(rng.random((n, v)) * 0.5 + 0.5 / v)
        log_b -= _logsumexp(log_b, axis=1)[:, None]
        rows = self._emissions(seqs)
        for _ in range(self.n_iter):
            pi_acc = np.zeros(n)
            a_acc = np.full((n, n), self.smoothing)
            b_acc = np.full((n, v), self.smoothing)
            for row in rows:
                alpha = self._forward(log_pi, log_a, log_b, row)
                beta = self._backward(log_a, log_b, row)
                gamma = alpha + beta
                gamma -= _logsumexp(gamma, axis=0)[None, :]
                pi_acc += np.exp(gamma[:, 0])
                for t in range(row.size):
                    b_acc[:, row[t]] += np.exp(gamma[:, t])
                for t in range(row.size - 1):
                    xi = (alpha[:, t][:, None] + log_a
                          + log_b[:, row[t + 1]][None, :] + beta[:, t + 1][None, :])
                    xi -= _logsumexp(xi)
                    a_acc += np.exp(xi)
            log_pi = np.log(np.maximum(pi_acc, 1e-300)) - math.log(pi_acc.sum())
            a_norm = a_acc / a_acc.sum(axis=1, keepdims=True)
            log_a = np.log(np.maximum(a_norm, 1e-300))
            b_norm = b_acc / b_acc.sum(axis=1, keepdims=True)
            log_b = np.log(np.maximum(b_norm, 1e-300))
        self.log_pi, self.log_a, self.log_b = log_pi, log_a, log_b
        self.n_parameters = (n - 1) + n * (n - 1) + n * (v - 1)
        return self

    def _forward(self, log_pi, log_a, log_b, row):
        T = row.size
        alpha = np.full((log_pi.size, T), NEG_INF)
        alpha[:, 0] = log_pi + log_b[:, row[0]]
        for t in range(1, T):
            alpha[:, t] = (
                _logsumexp(alpha[:, t - 1][:, None] + log_a, axis=0)
                + log_b[:, row[t]]
            )
        return alpha

    def _backward(self, log_a, log_b, row):
        T = row.size
        beta = np.zeros((log_a.shape[0], T))
        for t in range(T - 2, -1, -1):
            beta[:, t] = _logsumexp(
                log_a + log_b[:, row[t + 1]][None, :] + beta[:, t + 1][None, :],
                axis=1,
            )
        return beta

    def logprob(self, seq):
        """Total log2 likelihood of a sequence; None if it has unseen signs."""
        if not seq or any(s not in self.index for s in seq):
            return None
        row = np.array([self.index[s] for s in seq], dtype=int)
        alpha = self._forward(self.log_pi, self.log_a, self.log_b, row)
        return float(_logsumexp(alpha[:, -1]) / math.log(2))

    def viterbi_states(self, seq):
        """Most likely state path (used for state-stability analysis only)."""
        if not seq or any(s not in self.index for s in seq):
            return None
        row = np.array([self.index[s] for s in seq], dtype=int)
        T = row.size
        n = self.log_pi.size
        delta = np.full((n, T), NEG_INF)
        back = np.zeros((n, T), dtype=int)
        delta[:, 0] = self.log_pi + self.log_b[:, row[0]]
        for t in range(1, T):
            for s in range(n):
                scores = delta[:, t - 1] + self.log_a[:, s]
                best = int(np.argmax(scores))
                delta[s, t] = scores[best] + self.log_b[s, row[t]]
                back[s, t] = best
        path = [int(np.argmax(delta[:, T - 1]))]
        for t in range(T - 1, 0, -1):
            path.append(int(back[path[-1], t]))
        return list(reversed(path))