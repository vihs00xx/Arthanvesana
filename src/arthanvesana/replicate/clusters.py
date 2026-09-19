from __future__ import annotations

import random

import numpy as np


def _kmeans_labels(vectors: np.ndarray, k: int, seed: int) -> list[int]:
    rng = np.random.RandomState(seed)
    first = rng.choice(len(vectors))
    centers = [vectors[first]]
    nearest = np.full(len(vectors), np.inf)
    for _ in range(1, k):
        dist = ((vectors - centers[-1]) ** 2).sum(axis=1)
        nearest = np.minimum(nearest, dist)
        total = nearest.sum()
        if total <= 0:
            centers.append(vectors[rng.choice(len(vectors))])
            continue
        centers.append(vectors[rng.choice(len(vectors), p=nearest / total)])
    centers = np.array(centers)
    labels = np.zeros(len(vectors), dtype=int)
    for _ in range(100):
        dist = ((vectors[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for c in range(k):
            if (labels == c).any():
                centers[c] = vectors[labels == c].mean(axis=0)
    return [int(v) for v in labels]


def _silhouette(vectors: np.ndarray, labels: list[int]) -> float:
    labels = np.asarray(labels)
    dist = np.sqrt(((vectors[:, None, :] - vectors[None, :, :]) ** 2).sum(axis=2))
    scores = []
    for i in range(len(vectors)):
        same = (labels == labels[i]) & (np.arange(len(vectors)) != i)
        other = labels != labels[i]
        if not same.any() or not other.any():
            continue
        intra = dist[i][same].mean()
        inter = min(dist[i][labels == c].mean() for c in set(labels) if c != labels[i])
        scores.append((inter - intra) / max(inter, intra))
    return float(sum(scores) / len(scores)) if scores else 0.0


def kmeans_sweep(
    vectors: np.ndarray, k_values: list[int], seed: int = 0
) -> dict:
    from sklearn.metrics import silhouette_score

    vectors = np.asarray(vectors, dtype=float)
    n = len(vectors)
    rows = []
    label_sets = {}
    for k in k_values:
        if not isinstance(k, int) or k < 2 or k >= n:
            raise ValueError("k must satisfy 2 <= k < n_signs")
        labels = _kmeans_labels(vectors, k, seed)
        try:
            score = float(silhouette_score(vectors, labels))
        except ValueError:
            score = _silhouette(vectors, labels)
        if len(set(labels)) < 2:
            score = 0.0
        rows.append({"k": k, "silhouette": score})
        label_sets[k] = labels
    best = max(rows, key=lambda r: r["silhouette"])
    return {"sweep": rows, "best_k": best["k"], "labels": label_sets}


def hierarchical_labels(vectors: np.ndarray, n_clusters: int) -> list[int]:
    from sklearn.cluster import AgglomerativeClustering

    if n_clusters < 2 or n_clusters >= len(vectors):
        raise ValueError("n_clusters must satisfy 2 <= n_clusters < n_signs")
    labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(vectors)
    return [int(v) for v in labels]


def cluster_purity(labels: list[int], roles: dict) -> dict:
    if len(labels) != len(roles):
        raise ValueError("labels must align one-to-one with roles in dict order")
    clusters: dict[int, list] = {}
    for sign, label in zip(roles, labels):
        clusters.setdefault(label, []).append(roles[sign])
    if not clusters:
        raise ValueError("cluster_purity requires nonempty labels and roles")
    from collections import Counter

    sizes = []
    weighted = 0.0
    for members in clusters.values():
        top = Counter(members).most_common(1)[0][1]
        weighted += top
        sizes.append(len(members))
    return {
        "purity": weighted / sum(sizes),
        "n_clusters": len(clusters),
        "n_signs": sum(sizes),
    }


def purity_permutation_null(
    labels: list[int], roles: dict, n_reps: int = 200, seed: int = 0
) -> dict:
    observed = cluster_purity(labels, roles)["purity"]
    rng = random.Random(seed)
    role_values = list(roles.values())
    nulls = []
    for _ in range(n_reps):
        shuffled = rng.sample(role_values, len(role_values))
        nulls.append(cluster_purity(labels, dict(zip(roles, shuffled)))["purity"])
    nulls.sort()
    return {
        "observed": observed,
        "null_mean": sum(nulls) / len(nulls),
        "null_95": nulls[min(int(0.95 * n_reps), n_reps - 1)],
        "empirical_p": sum(v >= observed for v in nulls) / n_reps,
        "n_reps": n_reps,
    }


def adjusted_rand(first: list[int], second: list[int]) -> float:
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(first, second))
