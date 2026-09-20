import importlib.util
import json
from pathlib import Path

from arthanvesana.data.parse import to_tidy
from arthanvesana.simulate import (
    crossfit_effect,
    empirical_profile,
    gen_markov,
    gen_shuffled_real,
    gen_trigram_mixture,
    gen_unigram,
)

ROOT = Path(__file__).resolve().parents[1]


def load_runner(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records():
    seqs = [["001", "002", "003"], ["001", "002", "004"], ["002", "003", "001"]] * 20
    return [{"inscription_id": f"i{k}", "sequence": s} for k, s in enumerate(seqs)]


def test_generators_match_lengths_and_vocabulary():
    profile = empirical_profile(_records())
    for gen in (
        lambda: gen_unigram(profile, 30, 0),
        lambda: gen_markov(profile, 30, 0),
        lambda: gen_trigram_mixture(profile, 30, 0, 0.0),
    ):
        seqs = gen()
        assert len(seqs) == 30
        # sampled lengths come from the empirical length list
        assert all(len(s) in set(profile["lengths"]) for s in seqs)
        # signs drawn from the empirical vocabulary
        assert set(s for seq in seqs for s in seq) <= set(profile["unigram"])


def test_shuffled_real_preserves_multiset():
    records = _records()
    seqs = gen_shuffled_real(records, 10, 0)
    source = [tuple(sorted(r["sequence"])) for r in records]
    for seq in seqs:
        assert tuple(sorted(seq)) in source


def test_crossfit_is_group_level_and_p_never_zero():
    profile = empirical_profile(_records())
    res = crossfit_effect(gen_unigram(profile, 40, 0), 5, 0, 200, 200)
    for key in ("macro_effect", "ci95", "p", "n_groups", "weighted_effect"):
        assert key in res
    assert res["n_groups"] > 0
    # (exceedances + 1) / (permutations + 1) is strictly positive
    assert res["p"] >= 1.0 / (200 + 1)
    lo, hi = res["ci95"]
    assert lo <= hi


def test_duplicate_tokens_do_not_multiply_independent_groups():
    # one huge exact-duplicate group plus two singletons
    seqs = [["001", "002", "003"]] * 30 + [["007", "008"], ["009", "010"]]
    res = crossfit_effect(seqs, 5, 0, 100, 100)
    # exact-sequence connectivity merges the 30 copies into ONE group
    assert res["n_groups"] == 3


def test_crossfit_deterministic_for_fixed_seed():
    profile = empirical_profile(_records())
    a = crossfit_effect(gen_markov(profile, 40, 1), 5, 3, 100, 100)
    b = crossfit_effect(gen_markov(profile, 40, 1), 5, 3, 100, 100)
    assert a == b


def test_power_runner_writes_reproducible_report(tmp_path):
    runner = load_runner("run_power_analysis")
    raw = [
        {
            "id": str(i), "cisi": str(i), "site": "A", "direction": "L/R",
            "complete": True, "symbols": ["001", f"{100 + (i % 3):03d}", "003"],
        }
        for i in range(30)
    ]
    corpus = tmp_path / "corpus.csv"
    to_tidy(raw).to_csv(corpus, index=False)
    output = tmp_path / "out"
    summary = runner.main([
        "--corpus", str(corpus), "--output", str(output),
        "--replicates", "2", "--sizes", "1.0",
        "--permutations", "50", "--bootstrap", "50",
    ])
    saved = json.loads((output / "power_analysis_summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    # zero-effect scenarios plus the trigram lambda grid are all present
    keys = set(summary["grid"])
    assert any(k.startswith("unigram|") for k in keys)
    assert any(k.startswith("trigram_mixture|lam=0.0|") for k in keys)
    assert any(k.startswith("trigram_mixture|lam=1.0|") for k in keys)
    # every p-value respects the +1 correction
    for cell in summary["cells"]:
        if cell["p"] is not None:
            assert cell["p"] >= 1.0 / (50 + 1)
    report = (output / "power_analysis_report.txt").read_text(encoding="utf-8")
    assert "false-positive rate" in report
    assert "calibration instruments" in report