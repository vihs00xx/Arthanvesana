import importlib.util
import json
from pathlib import Path

from arthanvesana.data.parse import to_tidy

ROOT = Path(__file__).resolve().parents[1]


def load_runner(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ngram_inference_runner_writes_reproducible_report(tmp_path):
    runner = load_runner("run_ngram_inference")
    raw = [
        {
            "id": str(i), "cisi": str(i), "site": "A" if i < 10 else "B",
            "direction": "L/R", "complete": True,
            "symbols": ["001", f"{100 + (i % 4):03d}", "003", "004", "005"],
        }
        for i in range(20)
    ]
    corpus = tmp_path / "corpus.csv"
    to_tidy(raw).to_csv(corpus, index=False)
    output = tmp_path / "results"
    summary = runner.main([
        "--corpus", str(corpus), "--output", str(output),
        "--repeats", "1", "--permutations", "500",
    ])
    saved = json.loads((output / "ngram_inference_summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    assert saved["manifest"]["seeds"] == [0]
    assert saved["manifest"]["corpus_sha256"] == runner.sha256_file(corpus)
    assert len(saved["manifest"]["source_sha256"]) > 1
    assert len(saved["runs"]) == 1
    run = saved["runs"][0]
    for key in (
        "mean_logloss_bigram", "mean_logloss_trigram",
        "mean_paired_diff", "perm_p", "n_tokens",
    ):
        assert key in run
    assert run["n_tokens"] > 0
    assert run["mean_logloss_bigram"] > 0
    assert run["mean_logloss_trigram"] > 0
    assert 0.0 <= run["perm_p"] <= 1.0
    assert 0.0 <= saved["pooled"]["perm_p"] <= 1.0
    assert saved["pooled"]["n_tokens"] == run["n_tokens"]
    assert saved["aggregate"]["mean_paired_diff"]["n"] == 1
    report = (output / "ngram_inference_report.txt").read_text(encoding="utf-8")
    assert "NOT confidence intervals" in report
    assert "exchangeability" in report
    assert "trigram - bigram" in report
