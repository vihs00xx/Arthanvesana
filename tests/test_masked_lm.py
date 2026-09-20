import pytest

torch = pytest.importorskip("torch")

from arthanvesana.replicate.masked_lm import (
    SPECIAL_TOKENS,
    build_vocab,
    count_parameters,
    masked_lm_ranks,
    train_masked_lm,
)


def test_build_vocab_reserves_specials():
    table = build_vocab([["b", "a"], ["c"]])
    assert table == {"<PAD>": 0, "<MASK>": 1, "<UNK>": 2, "a": 3, "b": 4, "c": 5}
    with pytest.raises(ValueError):
        build_vocab([])


def test_vocabulary_is_fit_only():
    fit = [["a", "b"]]
    valid = [["a", "zzz"]]
    bundle = train_masked_lm(fit, valid, dim=8, heads=2, max_epochs=2, seed=0)
    assert "zzz" not in bundle["table"]
    assert set(bundle["table"]) - set(SPECIAL_TOKENS) == {"a", "b"}
    assert bundle["fit_vocab_size"] == 2


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


def test_special_tokens_are_never_candidates():
    bundle = train_masked_lm(
        [["a", "b"], ["b", "a"]], None, dim=8, heads=2, max_epochs=2, seed=0,
    )
    ranks = masked_lm_ranks(bundle, [["a", "b"]])
    # candidates exclude <PAD>/<MASK>/<UNK>, so ranks cannot be inflated by them
    assert all(r is None or r <= 2 for r in ranks)


def test_oov_targets_excluded_from_loss_and_counted():
    fit = [["a", "b"]] * 4
    valid = [["a", "zzz"]] * 4
    bundle = train_masked_lm(
        fit, valid, dim=8, heads=2, dropout=0.0, max_epochs=3, patience=3, seed=0,
    )
    # Masked validation targets are a random mix of in-vocab ("a") and OOV ("zzz"):
    # OOV is reported rather than trained as if it were a sign.
    assert bundle["n_valid_masked"] > 0
    assert bundle["n_valid_oov"] > 0
    assert 0.0 < bundle["valid_oov_rate"] < 1.0