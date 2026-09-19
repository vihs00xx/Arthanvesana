import pytest

torch = pytest.importorskip("torch")

from arthanvesana.replicate.masked_lm import (
    build_vocab,
    count_parameters,
    masked_lm_ranks,
    train_masked_lm,
)


def test_build_vocab_reserves_specials():
    table = build_vocab([["b", "a"], ["c"]])
    assert table == {"<PAD>": 0, "<MASK>": 1, "a": 2, "b": 3, "c": 4}
    with pytest.raises(ValueError):
        build_vocab([])


def test_toy_memorization_and_determinism():
    train = [["a", "b"]] * 20 + [["c", "d"]] * 20
    first = train_masked_lm(
        train, train, dim=16, layers=1, heads=2, dropout=0.0,
        max_epochs=40, patience=40, seed=0,
    )
    assert first["epochs_run"] == 40
    assert first["n_parameters"] == count_parameters(first["model"])
    ranks = masked_lm_ranks(first, [["a", "b"], ["c", "d"]])
    assert ranks == [1, 1, 1, 1]
    second = train_masked_lm(
        train, train, dim=16, layers=1, heads=2, dropout=0.0,
        max_epochs=40, patience=40, seed=0,
    )
    assert masked_lm_ranks(second, [["a", "b"]]) == ranks[:2]


def test_ranks_alignment_and_oov():
    train = [["a", "b", "c"], ["a", "b", "d"]]
    bundle = train_masked_lm(
        train, None, dim=8, layers=1, heads=2, dropout=0.0,
        max_epochs=2, seed=0,
    )
    assert bundle["best_valid_loss"] == float("inf")
    ranks = masked_lm_ranks(bundle, [["a", "b", "zzz"]])
    assert len(ranks) == 3
    assert ranks[2] is None
    with pytest.raises(ValueError):
        train_masked_lm(train, None, dim=7, heads=2, seed=0)
