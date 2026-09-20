import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from arthanvesana.data.parse import to_tidy
from arthanvesana.replicate.embeddings import (
    cooccurrence_counts,
    cosine_neighbors,
    embedding_restoration_ranks,
    neighbor_jaccard,
    ppmi_matrix,
    svd_embeddings,
    train_embeddings,
)

ROOT = Path(__file__).resolve().parents[1]


def load_runner(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cooccurrence_counts_window():
    seqs = [["a", "b", "c"]]
    assert cooccurrence_counts(seqs, window=1) == Counter({
        ("a", "b"): 1, ("b", "a"): 1, ("b", "c"): 1, ("c", "b"): 1,
    })
    wide = cooccurrence_counts(seqs, window=2)
    assert wide[("a", "c")] == 1 and wide[("c", "a")] == 1
    try:
        cooccurrence_counts(seqs, window=0)
    except ValueError:
        pass
    else:
        raise AssertionError("window=0 must raise")


def test_ppmi_matches_hand_computation():
    pairs = Counter({("a", "b"): 4, ("a", "c"): 1, ("c", "c"): 1})
    mat = ppmi_matrix(pairs, ["a", "b", "c"], alpha=1.0)
    total = 6
    expect = math.log2((4 / total) / ((5 / total) * (4 / total)))
    assert mat[0, 1] == expect > 0
    assert mat[0, 2] == 0.0
    assert (mat >= 0.0).all()


def test_svd_embeddings_deterministic_and_shaped():
    rng = np.random.RandomState(0)
    mat = rng.rand(6, 6) + np.eye(6)
    first = svd_embeddings(mat, dim=3)
    second = svd_embeddings(mat, dim=3)
    assert first.shape == (6, 3)
    assert np.array_equal(np.abs(first), np.abs(second))
    assert svd_embeddings(mat, dim=100).shape == (6, 6)


def test_train_embeddings_neighbors_exclude_self():
    seqs = [["a", "b"], ["a", "b"], ["a", "c"], ["b", "c"], ["c", "d"]]
    model = train_embeddings(seqs, window=1, dim=2)
    assert model["vocab"] == ["a", "b", "c", "d"]
    neighbors = cosine_neighbors(model, "a", top_k=3)
    assert [n for n, _ in neighbors] == ["b", "c", "d"] or set(
        n for n, _ in neighbors
    ) == {"b", "c", "d"}
    assert all(n != "a" for n, _ in neighbors)
    sims = [s for _, s in neighbors]
    assert sims == sorted(sims, reverse=True)
    assert neighbor_jaccard(["a", "b"], ["b", "c"]) == 1 / 3
    assert neighbor_jaccard([], []) == 1.0


def test_embedding_restoration_predicts_observed_associate():
    train = [
        {"inscription_id": f"t{i}", "sequence": ["a", "b"]}
        for i in range(5)
    ] + [
        {"inscription_id": f"u{i}", "sequence": ["c", "d"]}
        for i in range(5)
    ]
    ranks = embedding_restoration_ranks(
        train, [{"inscription_id": "s1", "sequence": ["a", "b"]}],
        window=1, dim=2,
    )
    assert ranks == [1, 1]


def test_embedding_restoration_alignment_and_oov():
    train = [
        {"inscription_id": "t1", "sequence": ["a", "b", "c"]},
        {"inscription_id": "t2", "sequence": ["a", "b", "d"]},
    ]
    test = [{"inscription_id": "s1", "sequence": ["a", "b", "zzz"]}]
    ranks = embedding_restoration_ranks(train, test, window=1, dim=2)
    assert len(ranks) == 3
    assert ranks[2] is None
    assert all(r is None or r >= 1 for r in ranks)
    again = embedding_restoration_ranks(train, test, window=1, dim=2)
    assert ranks == again


def _toy_seqs():
    return (
        [["a", "b"]] * 20 + [["a", "c"]] * 20 + [["d", "b"]] * 20 + [["d", "c"]] * 20
    )


def test_skipgram_deterministic_and_ranked():
    from arthanvesana.replicate.embeddings import (
        skipgram_restoration_ranks,
        train_skipgram,
    )

    first = train_skipgram(_toy_seqs(), window=1, dim=4, seed=0)
    second = train_skipgram(_toy_seqs(), window=1, dim=4, seed=0)
    assert first["vocab"] == ["a", "b", "c", "d"]
    assert np.array_equal(first["vectors"], second["vectors"])
    train = [{"inscription_id": f"t{i}", "sequence": s} for i, s in enumerate(_toy_seqs())]
    ranks = skipgram_restoration_ranks(
        train, [{"inscription_id": "s1", "sequence": ["a", "b"]}], window=1, dim=4
    )
    assert ranks == [2, 1]
    assert ranks == skipgram_restoration_ranks(
        train, [{"inscription_id": "s1", "sequence": ["a", "b"]}], window=1, dim=4
    )
    with pytest.raises(ValueError):
        train_skipgram([], window=1, dim=4)


def test_cluster_purity_and_permutation_null():
    from arthanvesana.replicate.clusters import (
        adjusted_rand,
        cluster_purity,
        hierarchical_labels,
        kmeans_sweep,
        purity_permutation_null,
    )

    vectors = np.array([
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1],
        [5.0, 5.0], [5.1, 5.0], [5.0, 5.1], [5.1, 5.1],
    ])
    sweep = kmeans_sweep(vectors, [2, 3], seed=0)
    assert sweep["best_k"] == 2
    assert len(sweep["labels"][2]) == 8
    roles = {f"s{i}": "begin" if i < 4 else "end" for i in range(8)}
    purity = cluster_purity(sweep["labels"][2], roles)
    assert purity["purity"] == 1.0
    null = purity_permutation_null(sweep["labels"][2], roles, n_reps=50, seed=0)
    assert null["empirical_p"] <= 0.05
    hier = hierarchical_labels(vectors, 2)
    assert sorted(Counter(hier).values()) == [4, 4]
    with pytest.raises(ValueError):
        hierarchical_labels(vectors, 8)
    assert adjusted_rand([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    with pytest.raises(ValueError):
        cluster_purity([0, 0], roles)


def test_embeddings_runner_writes_reproducible_report(tmp_path):
    runner = load_runner("run_embeddings")
    raw = [
        {
            "id": str(i), "cisi": str(i), "site": "A" if i < 6 else "B",
            "direction": "L/R", "complete": True,
            "symbols": ["001", f"{100 + (i % 2):03d}", "003"],
        }
        for i in range(12)
    ]
    corpus = tmp_path / "corpus.csv"
    to_tidy(raw).to_csv(corpus, index=False)
    output = tmp_path / "results"
    summary = runner.main(["--corpus", str(corpus), "--output", str(output), "--repeats", "2"])
    saved = json.loads((output / "embeddings_summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    assert saved["manifest"]["seeds"] == [0, 1]
    assert saved["n_ok"] == 2
    assert len(saved["runs"]) == 2
    for run in saved["runs"]:
        assert run["bigram"]["n_masked"] == run["ppmi"]["n_masked"] == run["skipgram"]["n_masked"]
        assert len(run["inner_ppmi_sweep"]) == 6
        assert len(run["inner_skipgram_sweep"]) == 4
        assert run["selected_ppmi"]["window"] in (1, 2)
        assert run["selected_skipgram"]["window"] in (1, 2)
    assert saved["clustering"]["ppmi"]["best_k"] >= 2
    assert (output / "figures" / "umap_ppmi.png").exists()
    report = (output / "embeddings_report.txt").read_text(encoding="utf-8")
    assert "NOT confidence intervals" in report
    assert "nested" in report.lower()
