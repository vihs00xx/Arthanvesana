"""Compact structural models for the Sindhu-Sarasvatī sign sequences.

Model families, all described by structure rather than linguistic labels:

1. **Position models** with training-only hierarchical smoothing toward the
   global sign distribution. Three separately named variants:
   - ``exact``: bucket = (span length, zero-based index).
   - ``relative``: bucket = predefined normalized-position bin.
   - ``exact_complete``: bucket = (span length, index, start_complete,
     end_complete) — the optional completeness-aware version.
2. **A discrete hidden Markov model** with a selected state count, fitted by
   Baum-Welch from several deterministic initializations. Its states are a
   compact summary of positional sequence structure and are NOT words, phrases,
   grammatical roles, or semantic classes.
3. **An MKN bigram baseline.**

Evaluation is grouped cross-validation with held-out log loss as the primary
metric. The API takes **full records**, so original artifact identities and
grouping are preserved all the way through.

Unknown signs
-------------
One shared, training-only vocabulary policy is used by every model. The
vocabulary is built from the fitting data only, and ``<UNK>`` is always a member
with a small pseudo-count, so every model assigns it positive probability.
Likelihood is therefore evaluated over **mapped ``<UNK>`` outcomes** and is
directly comparable across models. A single unseen sign never discards the
known-token contributions of its sequence. Restoration scoring is separate: an
unseen original sign can never count as a correct restoration merely because
``<UNK>`` was predicted.
"""

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from arthanvesana.stats.grouping import (
    assign_folds,
    group_stats,
    leakage_diagnostics,
)
from arthanvesana.stats.ngrams import NGramModel
from arthanvesana.stats.sampling import connected_groups

NEG_INF = -1e30
UNK = "<UNK>"

#: Predefined normalized-position bins (fraction of the span already consumed).
RELATIVE_BINS = ((0.0, 0.2, "0-20%"), (0.2, 0.4, "20-40%"), (0.4, 0.6, "40-60%"),
                 (0.6, 0.8, "60-80%"), (0.8, 1.0001, "80-100%"))
SINGLETON_BIN = "singleton"

POSITION_MODES = ("exact", "relative", "exact_complete")


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


# ---------------------------------------------------------------------------
# Shared training-only vocabulary policy
# ---------------------------------------------------------------------------

class Vocabulary:
    """Training-only sign vocabulary with an always-present ``<UNK>``.

    ``unk_pseudo`` is the pseudo-count given to ``<UNK>`` in the global
    distribution. It is the single knob that makes unseen signs scoreable in
    every model, so likelihoods are comparable across model families.
    """

    def __init__(self, train_seqs, unk_pseudo=1.0):
        self.signs = sorted({s for seq in train_seqs for s in seq})
        self.index = {s: i for i, s in enumerate(self.signs)}
        self.unk_pseudo = float(unk_pseudo)
        counts = Counter(s for seq in train_seqs for s in seq)
        total = sum(counts.values()) + self.unk_pseudo
        self.global_dist = {s: counts[s] / total for s in self.signs}
        self.global_dist[UNK] = self.unk_pseudo / total
        self.size = len(self.signs) + 1  # + <UNK>

    def map(self, seq):
        return [s if s in self.index else UNK for s in seq]

    def outcomes(self):
        return [*self.signs, UNK]

    def global_prob(self, sign):
        return self.global_dist.get(sign, 0.0)


def training_corpus(records):
    return [r["sequence"] for r in records if r["sequence"]]


class NgramScorer:
    """Record-based adapter for :class:`NGramModel`.

    Gives the bigram the same ``logprob(record)`` interface as the other models,
    and floors zero probabilities at ``1e-12`` instead of raising. A single
    zero-probability context must not abort an entire evaluation, and every model
    in a comparison has to return a finite per-token likelihood.

    Unseen signs are already mapped to ``<UNK>`` by the wrapped model, which
    always has positive mass in its vocabulary.
    """

    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        # delegate _map / dist / n / vocab so the model stays usable as before
        return getattr(self.model, name)

    def logprob(self, record):
        seq = self.model._map(record["sequence"])
        if not seq:
            return 0.0
        if self.model.n == 1:
            dist = self.model.dist(())
            return sum(math.log2(max(dist[s], 1e-12)) for s in seq)
        padded = ["<S>"] * (self.model.n - 1) + seq
        total = 0.0
        for i in range(len(seq)):
            context = tuple(padded[i:i + self.model.n - 1])
            prob = self.model.dist(context)[padded[i + self.model.n - 1]]
            total += math.log2(max(prob, 1e-12))
        return total


# ---------------------------------------------------------------------------
# Position models
# ---------------------------------------------------------------------------

def relative_bin(pos, length):
    """Predefined normalized-position bin label."""
    if length <= 1:
        return SINGLETON_BIN
    frac = pos / (length - 1)
    for lo, hi, label in RELATIVE_BINS:
        if lo <= frac < hi:
            return label
    return RELATIVE_BINS[-1][2]


class PositionModel:
    """Hierarchically smoothed position model.

    ``p(w | bucket) = (count(bucket, w) + alpha * p_global(w)) / (n(bucket) + alpha)``

    An empty or rare bucket therefore degrades to the global sign distribution
    instead of failing, and ``<UNK>`` always keeps positive mass through
    ``p_global``. ``alpha`` is selected on inner grouped validation.
    """

    def __init__(self, mode="exact", alpha=1.0):
        if mode not in POSITION_MODES:
            raise ValueError(f"mode must be one of {POSITION_MODES}")
        self.mode = mode
        self.alpha = float(alpha)
        self.vocab = None
        self.counts = {}
        self.bucket_totals = Counter()
        self.n_buckets = 0
        self.n_parameters = 0

    def _key(self, length, pos, start_complete=True, end_complete=True):
        if self.mode == "relative":
            return (relative_bin(pos, length),)
        if self.mode == "exact_complete":
            return (length, pos, bool(start_complete), bool(end_complete))
        return (length, pos)

    def fit(self, records, vocab):
        self.vocab = vocab
        counts = {}
        totals = Counter()
        for record in records:
            seq = record["sequence"]
            for pos, sign in enumerate(seq):
                key = self._key(len(seq), pos,
                                record.get("start_complete", True),
                                record.get("end_complete", True))
                counts.setdefault(key, Counter())[sign] += 1
                totals[key] += 1
        self.counts = counts
        self.bucket_totals = totals
        self.n_buckets = len(counts)
        # one parameter per bucket-outcome pair plus the global distribution
        self.n_parameters = self.n_buckets * vocab.size + vocab.size
        return self

    def token_dist(self, length, pos, start_complete=True, end_complete=True):
        key = self._key(length, pos, start_complete, end_complete)
        bucket = self.counts.get(key)
        total = self.bucket_totals.get(key, 0)
        alpha = self.alpha
        denom = total + alpha
        out = {}
        for sign in self.vocab.outcomes():
            prior = self.vocab.global_prob(sign)
            observed = bucket.get(sign, 0) if bucket else 0
            out[sign] = (observed + alpha * prior) / denom if denom else prior
        return out

    def logprob(self, record):
        """Total log2 likelihood; never None, even with unseen signs.

        The sequence is mapped through the training-only vocabulary first, so an
        unseen sign is scored as ``<UNK>`` (which always has positive mass)
        rather than falling to the numerical floor.
        """
        seq = self.vocab.map(record["sequence"])
        if not seq:
            return 0.0
        total = 0.0
        for pos, sign in enumerate(seq):
            dist = self.token_dist(len(seq), pos,
                                   record.get("start_complete", True),
                                   record.get("end_complete", True))
            total += math.log2(max(dist.get(sign, 0.0), 1e-12))
        return total


# ---------------------------------------------------------------------------
# Discrete HMM
# ---------------------------------------------------------------------------

class DiscreteHMM:
    """Discrete HMM fitted by Baum-Welch (EM) in log space.

    ``<UNK>`` is part of the emission vocabulary with a pseudo-count, so a
    sequence containing unseen signs is scored token by token rather than
    discarded. Several deterministic initializations are available; the fit
    records its training likelihood per iteration and whether it converged.

    The latent states summarize positional sequence structure. They carry no
    linguistic interpretation.
    """

    INIT_STRATEGIES = ("uniform", "frequency_slice", "seeded_random")

    def __init__(self, n_states=3, n_iter=15, seed=0, smoothing=1e-2,
                 init="frequency_slice", tol=1e-4):
        self.n_states = n_states
        self.n_iter = n_iter
        self.seed = seed
        self.smoothing = smoothing
        self.init = init
        self.tol = tol
        self.vocab = []
        self.index = {}
        self.log_pi = None
        self.log_a = None
        self.log_b = None
        self.n_parameters = 0
        self.loglik_history = []
        self.converged = False
        self.iterations_run = 0

    def _init_log_b(self, rng, n, v, seqs):
        if self.init == "uniform":
            return np.log(np.ones((n, v)) / v)
        if self.init == "frequency_slice":
            # deterministic: assign each sign to a state by descending frequency
            counts = Counter(s for seq in seqs for s in seq)
            order = [s for s, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
            slices = np.array_split(np.arange(len(order)), n)
            arr = np.full((n, v), 1e-3)
            for state, sl in enumerate(slices):
                for j in sl:
                    arr[state, self.index[order[j]]] = 1.0
            return np.log(arr / arr.sum(axis=1, keepdims=True))
        return np.log(rng.random((n, v)) * 0.5 + 0.5 / v)

    def fit(self, records):
        seqs = [r["sequence"] for r in records if r["sequence"]]
        if not seqs:
            raise ValueError("DiscreteHMM requires nonempty training sequences")
        self.vocab = sorted({s for seq in seqs for s in seq}) + [UNK]
        self.index = {s: i for i, s in enumerate(self.vocab)}
        v = len(self.vocab)
        n = max(1, min(self.n_states, v))
        rng = np.random.default_rng(self.seed)
        log_pi = np.log(np.ones(n) / n)
        log_a = np.log(np.ones((n, n)) / n)
        log_b = self._init_log_b(rng, n, v, seqs)
        log_b -= _logsumexp(log_b, axis=1)[:, None]
        rows = [np.array([self.index.get(s, self.index[UNK]) for s in seq], dtype=int)
                for seq in seqs]
        rows = [row for row in rows if row.size]
        previous = None
        for it in range(self.n_iter):
            pi_acc = np.zeros(n)
            a_acc = np.full((n, n), self.smoothing)
            b_acc = np.full((n, v), self.smoothing)
            total_ll = 0.0
            for row in rows:
                alpha = self._forward(log_pi, log_a, log_b, row)
                beta = self._backward(log_a, log_b, row)
                gamma = alpha + beta
                gamma -= _logsumexp(gamma, axis=0)[None, :]
                total_ll += _logsumexp(alpha[:, -1])
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
            self.loglik_history.append(float(total_ll))
            self.iterations_run = it + 1
            if previous is not None and abs(total_ll - previous) < self.tol:
                self.converged = True
                break
            previous = total_ll
        self.log_pi, self.log_a, self.log_b = log_pi, log_a, log_b
        self.n_states_used = n
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

    def token_logprobs(self, record):
        """Per-token log2 likelihoods under the posterior; never None."""
        seq = record["sequence"]
        if not seq:
            return []
        row = np.array([self.index.get(s, self.index[UNK]) for s in seq], dtype=int)
        alpha = self._forward(self.log_pi, self.log_a, self.log_b, row)
        beta = self._backward(self.log_a, self.log_b, row)
        gamma = alpha + beta
        gamma -= _logsumexp(gamma, axis=0)[None, :]
        out = []
        for t in range(row.size):
            logp = _logsumexp(gamma[:, t] + self.log_b[:, row[t]]) / math.log(2)
            out.append(float(logp))
        return out

    def logprob(self, record):
        """Total log2 likelihood; never None, even with unseen signs."""
        seq = record["sequence"]
        if not seq:
            return 0.0
        row = np.array([self.index.get(s, self.index[UNK]) for s in seq], dtype=int)
        alpha = self._forward(self.log_pi, self.log_a, self.log_b, row)
        return float(_logsumexp(alpha[:, -1]) / math.log(2))

    def viterbi_states(self, seq):
        """Most likely state path (used for state-stability analysis only)."""
        if not seq:
            return None
        row = np.array([self.index.get(s, self.index[UNK]) for s in seq], dtype=int)
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


# ---------------------------------------------------------------------------
# Training budgets
# ---------------------------------------------------------------------------

def cap_records_by_group(records, groups, cap_records, seed):
    """Cap a training partition using WHOLE components only.

    Components are added largest-first with seeded tie-breaking until the record
    cap is reached, so the cap never splits a component and the realized size is
    reported by the caller. Returns ``(capped_records, realized)``.
    """
    if cap_records is None or len(records) <= cap_records:
        return list(records), {
            "cap": cap_records, "n_records": len(records),
            "capped": False, "n_groups": len(groups),
        }
    rng = random.Random(seed)
    tie = {g["group_id"]: rng.random() for g in groups}
    order = sorted(groups, key=lambda g: (-len(g["indices"]), tie[g["group_id"]]))
    chosen = []
    for group in order:
        if len(chosen) >= cap_records:
            break
        chosen.extend(group["indices"])
    selected = {i for i in chosen}
    capped = [r for i, r in enumerate(records) if i in selected]
    return capped, {
        "cap": cap_records, "n_records": len(capped), "capped": True,
        "n_groups": sum(1 for g in groups if set(g["indices"]) <= selected),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def token_logloss(model, records):
    """Mean log2 loss per token over mapped outcomes.

    Never substitutes a whole-sequence penalty: every model returns a per-token
    likelihood, so one unseen sign does not discard the known tokens.
    """
    total = 0.0
    n = 0
    for record in records:
        seq = record["sequence"]
        if not seq:
            continue
        lp = model.logprob(record)
        total -= lp
        n += len(seq)
    return (total / n) if n else None, n


def oov_rate(vocab, records):
    """Fraction of evaluated tokens whose original sign is outside the vocabulary."""
    total = unknown = 0
    for record in records:
        for sign in record["sequence"]:
            total += 1
            if sign not in vocab.index:
                unknown += 1
    return (unknown / total) if total else 0.0


def restoration_top1(model, vocab, records):
    """Top-1 restoration accuracy over ORIGINAL signs.

    Distinct from likelihood: an unseen original sign can never be counted as a
    correct restoration merely because ``<UNK>`` was predicted, and ``<UNK>`` is
    never a candidate. Every masked position is scored.
    """
    correct = total = 0
    candidates = vocab.outcomes()
    for record in records:
        seq = record["sequence"]
        for pos, sign in enumerate(seq):
            total += 1
            if sign not in vocab.index:
                continue  # unseen target: cannot be restored
            if isinstance(model, PositionModel):
                dist = model.token_dist(len(seq), pos,
                                        record.get("start_complete", True),
                                        record.get("end_complete", True))
            elif isinstance(model, DiscreteHMM):
                dist = _hmm_marginal(model, record, pos)
            else:  # NGramModel
                mapped = model._map(seq)
                ctx = tuple((["<S>"] * (model.n - 1) + mapped[:pos])[-(model.n - 1):])
                dist = model.dist(ctx)
            best = max((w for w in candidates if w != UNK),
                       key=lambda w: dist.get(w, 0.0))
            if best == sign:
                correct += 1
    return (correct / total) if total else None, total


def _hmm_marginal(model, record, pos):
    """Posterior over emissions at one position."""
    seq = record["sequence"]
    row = np.array([model.index.get(s, model.index[UNK]) for s in seq], dtype=int)
    alpha = model._forward(model.log_pi, model.log_a, model.log_b, row)
    beta = model._backward(model.log_a, model.log_b, row)
    gamma = alpha[:, pos] + beta[:, pos]
    gamma = gamma - _logsumexp(gamma)
    post = np.exp(gamma)
    out = {}
    for sign, idx in model.index.items():
        out[sign] = float(post @ np.exp(model.log_b[:, idx]))
    return out


# ---------------------------------------------------------------------------
# Nested grouped cross-validation
# ---------------------------------------------------------------------------

def inner_grouped_split(records, groups, seed, n_folds=2):
    """Carve an inner validation split from a training partition, grouped.

    Components move as units, so no inner-validation record shares a component
    with the inner-fitting data.
    """
    stats = group_stats(records, groups)
    assignment = assign_folds(stats, n_folds, seed)
    valid_fold = n_folds - 1
    gid_of = groups_of(groups)
    fit = [r for i, r in enumerate(records)
           if assignment[gid_of[i]] != valid_fold]
    valid = [r for i, r in enumerate(records)
             if assignment[gid_of[i]] == valid_fold]
    if not fit or not valid:
        cut = max(1, len(records) // 2)
        return list(records[:cut]), list(records[cut:]) or list(records[:1])
    return fit, valid


def groups_of(groups):
    """Map record index -> group id."""
    out = {}
    for group in groups:
        for i in group["indices"]:
            out[i] = group["group_id"]
    return out


def _select_alpha(records, groups, seed, vocab, alphas):
    fit, valid = inner_grouped_split(records, groups, seed)
    best, best_loss = alphas[0], None
    for alpha in alphas:
        model = PositionModel("exact", alpha).fit(fit, vocab)
        loss, _ = token_logloss(model, valid)
        if loss is not None and (best_loss is None or loss < best_loss):
            best, best_loss = alpha, loss
    return best


def _select_hmm(records, groups, seed, state_counts, inits, n_iter):
    fit, valid = inner_grouped_split(records, groups, seed)
    best = None
    trials = []
    for k in state_counts:
        for init in inits:
            model = DiscreteHMM(n_states=k, n_iter=n_iter, seed=seed, init=init).fit(fit)
            loss, _ = token_logloss(model, valid)
            trials.append({
                "n_states": k, "init": init, "inner_loss": loss,
                "converged": model.converged, "iterations": model.iterations_run,
                "train_loglik": (model.loglik_history[-1]
                                 if model.loglik_history else None),
            })
            if loss is not None and (best is None or loss < best[1]):
                best = ((k, init), loss)
    if best is None:
        return state_counts[0], inits[0], trials
    return best[0][0], best[0][1], trials


def crossfit_compact(records, n_folds=5, seed=0, state_counts=(2, 3, 4, 5),
                     inits=("frequency_slice", "uniform"),
                     alphas=(0.1, 1.0, 10.0), hmm_iter=8,
                     budget="full", cap_records=None):
    """Grouped nested cross-validation of compact models vs the bigram.

    ``records`` are full record dicts, so artifact identity and grouping survive
    the whole runner. Two budgets are supported:

    * ``budget="matched"`` — every model is trained on the SAME capped training
      subset, selected with whole components. This is the fair model-family
      comparison.
    * ``budget="full"`` — every model is trained on the full outer-training
      partition, except the HMM when ``cap_records`` is set for tractability;
      that limitation is reported rather than hidden.

    HMM state count and initialization are selected on an inner grouped split of
    the outer training partition only; the outer test never enters selection.
    """
    groups = connected_groups(records, track="artifact", group_duplicates=True)
    stats = group_stats(records, groups)
    assignment = assign_folds(stats, n_folds, seed)
    gid_of = groups_of(groups)
    leakage = leakage_diagnostics(records, groups, assignment)

    models = ("bigram", "position_exact", "position_relative",
              "position_exact_complete", "hmm")
    acc = {name: [0.0, 0] for name in models}
    oov = {name: [0, 0] for name in models}
    selected_states, selected_inits, hmm_params, hmm_diag = [], [], [], []
    alpha_selected, budgets = [], []

    for fold in range(n_folds):
        train = [r for i, r in enumerate(records)
                 if assignment[gid_of[i]] != fold]
        test = [r for i, r in enumerate(records)
                if assignment[gid_of[i]] == fold]
        if not train or not test:
            continue
        train_groups_all = connected_groups(train, track="artifact",
                                            group_duplicates=True)
        if budget == "matched":
            train_used, realized = cap_records_by_group(
                train, train_groups_all, cap_records, seed + fold)
        else:
            train_used, realized = list(train), {
                "cap": cap_records, "n_records": len(train), "capped": False,
                "n_groups": len(train_groups_all),
            }
        # Recompute grouping for the ACTUAL training subset: group indices from
        # the pre-cap list do not address the capped list, and using them would
        # silently mis-index the inner split.
        train_groups = connected_groups(train_used, track="artifact",
                                        group_duplicates=True)
        budgets.append({"fold": fold, "budget": budget, **realized})

        vocab = Vocabulary(training_corpus(train_used))
        fitted = {}
        # bigram
        bigram = NgramScorer(NGramModel(training_corpus(train_used), 2, method="mkn"))
        fitted["bigram"] = bigram
        # position models: alpha selected on an inner grouped split
        alpha = _select_alpha(train_used, train_groups, seed + fold, vocab, list(alphas))
        alpha_selected.append(alpha)
        for mode in ("exact", "relative", "exact_complete"):
            fitted[f"position_{mode}"] = PositionModel(mode, alpha).fit(train_used, vocab)
        # HMM: state count and init selected on an inner grouped split
        k, init, trials = _select_hmm(
            train_used, train_groups, seed + fold, list(state_counts), list(inits),
            hmm_iter)
        selected_states.append(k)
        selected_inits.append(init)
        hmm_diag.append({"fold": fold, "trials": trials})
        hmm_train = train_used
        if budget == "full" and cap_records is not None:
            hmm_train, hmm_realized = cap_records_by_group(
                train_used, train_groups, cap_records, seed + fold)
            budgets[-1]["hmm_capped"] = True
            budgets[-1]["hmm_n_records"] = hmm_realized["n_records"]
        hmm = DiscreteHMM(n_states=k, n_iter=hmm_iter * 2, seed=seed + fold,
                          init=init).fit(hmm_train)
        fitted["hmm"] = hmm
        hmm_params.append({
            "fold": fold, "n_states": k, "init": init,
            "n_parameters": hmm.n_parameters,
            "converged": hmm.converged,
            "iterations": hmm.iterations_run,
            "final_train_loglik": (hmm.loglik_history[-1]
                                   if hmm.loglik_history else None),
        })

        for name in models:
            loss, ntok = token_logloss(fitted[name], test)
            if loss is not None:
                acc[name][0] += loss * ntok
                acc[name][1] += ntok
            oov[name][0] += int(round(oov_rate(vocab, test) * _n_tokens(test)))
            oov[name][1] += _n_tokens(test)

    out = {}
    for name, (total, ntok) in acc.items():
        if ntok:
            bits = total / ntok
            out[name] = {"bits_per_token": bits, "perplexity": 2 ** bits,
                         "n_tokens": ntok,
                         "oov_rate": (oov[name][0] / oov[name][1])
                         if oov[name][1] else None}
        else:
            out[name] = {"bits_per_token": None, "perplexity": None,
                         "n_tokens": 0, "oov_rate": None}
    out["selected_hmm_states"] = selected_states
    out["selected_hmm_inits"] = selected_inits
    out["hmm_diagnostics"] = hmm_diag
    out["hmm_n_parameters"] = hmm_params
    out["selected_position_alpha"] = alpha_selected
    out["training_budgets"] = budgets
    out["budget"] = budget
    out["n_groups"] = len(groups)
    out["n_records"] = len(records)
    out["leakage"] = leakage
    out["vocabulary_policy"] = (
        "training-only vocabulary; <UNK> always present with a pseudo-count; "
        "likelihood evaluated over mapped <UNK> outcomes in every model")
    return out


def _n_tokens(records):
    return sum(len(r["sequence"]) for r in records)