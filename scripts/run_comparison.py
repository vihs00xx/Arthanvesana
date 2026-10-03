"""First typological comparison: the Indus statistics on comparison corpora.

Runs the same measures on the Indus corpus and on each comparison corpus built
by ``build_comparison_corpora.py``. Corpora differ in size by up to three
orders of magnitude and most measures depend on sample size, so comparison
corpora are subsampled to the Indus analysable token count:

* ``token``: whole artifacts are drawn at random until the gap-split spans
  reach the Indus token total. Sequence lengths stay as the corpus has them.
* ``length``: spans are drawn to reproduce the Indus span-length histogram
  (at most the Indus count per length). Where a corpus has too few spans of a
  length, the achieved share is reported rather than hidden.

Each replicate r uses seed r for both the subsample and the artifact-grouped
train/test split; the Indus corpus is used whole, so its spread comes from the
split alone. Every measure is a descriptive statistic of a sign system's
structure; none of them by itself separates writing from non-linguistic
sign systems.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean, pstdev

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arthanvesana.data.parse import analysis_records  # noqa: E402
from arthanvesana.replicate.robustness import evaluate_split  # noqa: E402
from arthanvesana.replicate.zipf import mandelbrot_fit, rank_frequencies  # noqa: E402
from arthanvesana.stats.entropy import (  # noqa: E402
    conditional_entropy,
    mutual_information,
    unigram_entropy,
)
from arthanvesana.stats.ngrams import NGramModel  # noqa: E402
from arthanvesana.stats.position import positional_counts, positional_profile  # noqa: E402
from arthanvesana.stats.sampling import shuffled_corpus, split_records  # noqa: E402

INDUS = ROOT / "data" / "processed" / "corpus.csv"
COMPARISON = ROOT / "data" / "external" / "comparison"
OUT = ROOT / "outputs" / "comparison"
CORPORA = ("proto_cuneiform", "proto_elamite", "ur3_admin", "seal_legends")
POOL_TOKENS = 1_500_000  # artifacts kept from very large corpora before sampling
NUMERAL = re.compile(r"^[0-9]+(?:/[0-9]+)?\(.+\)$")


def load(path: Path, seed: int, pool_tokens: int | None = POOL_TOKENS) -> list[dict]:
    df = pd.read_csv(path, encoding="utf-8", dtype={"sign_code": str}, keep_default_na=False)
    if pool_tokens is not None and len(df) > pool_tokens:
        artifacts = df["artifact_id"].drop_duplicates().tolist()
        random.Random(seed).shuffle(artifacts)
        sizes = df.groupby("artifact_id").size()
        keep, total = set(), 0
        for artifact in artifacts:
            keep.add(artifact)
            total += int(sizes[artifact])
            if total >= pool_tokens:
                break
        df = df[df["artifact_id"].isin(keep)]
    return analysis_records(df, gap_policy="split", known_direction_only=True)


def without_numerals(records: list[dict]) -> list[dict]:
    """Drop numeral-group tokens (``3(N01)``, ``2(gesz2)``) from every span.

    Numerals dominate the early accounting corpora; this variant asks whether
    the comparison holds for the non-numeral signs alone. Span-edge flags are
    kept, so a span whose first sign was a numeral still counts as starting
    complete.
    """
    out = []
    for record in records:
        sequence = [s for s in record["sequence"] if not NUMERAL.match(s)]
        if sequence:
            out.append(dict(record, sequence=sequence))
    return out


def token_matched(records: list[dict], target: int, seed: int) -> list[dict]:
    by_artifact: dict = {}
    for record in records:
        by_artifact.setdefault(record["artifact_id"], []).append(record)
    keys = sorted(by_artifact)
    random.Random(seed).shuffle(keys)
    chosen, total = [], 0
    for key in keys:
        if total >= target:
            break
        chosen += by_artifact[key]
        total += sum(len(r["sequence"]) for r in by_artifact[key])
    return chosen


def length_matched(records: list[dict], histogram: Counter, seed: int) -> tuple[list[dict], float]:
    by_length: dict = {}
    for record in records:
        by_length.setdefault(len(record["sequence"]), []).append(record)
    rng = random.Random(seed)
    chosen = []
    for length, need in sorted(histogram.items()):
        pool = list(by_length.get(length, []))
        rng.shuffle(pool)
        chosen += pool[:need]
    wanted = sum(n * length for length, n in histogram.items())
    got = sum(len(r["sequence"]) for r in chosen)
    return chosen, got / wanted


def measures(records: list[dict], seed: int) -> dict:
    seqs = [r["sequence"] for r in records]
    complete = [r["sequence"] for r in records if r["start_complete"] and r["end_complete"]]
    counts = Counter(s for seq in seqs for s in seq)
    n_tokens = sum(counts.values())
    h1 = unigram_entropy(seqs)
    h2, h2mm = conditional_entropy(seqs, 1)
    _, h3mm = conditional_entropy(seqs, 2)
    null_h2 = fmean(conditional_entropy(n, 1)[0] for n in shuffled_corpus(seqs, seed, 10))

    train, test, _ = split_records(records, track="artifact", seed=seed)
    tr = [r["sequence"] for r in train]
    te = [r["sequence"] for r in test]
    ppl1 = NGramModel(tr, 1, method="mkn").perplexity(te)
    ppl2 = NGramModel(tr, 2, method="mkn").perplexity(te)
    restore = evaluate_split(train, test)["models"]

    fit = mandelbrot_fit(rank_frequencies(seqs))
    pos = positional_counts(complete)
    n_complete = len(complete)
    profile = positional_profile(complete, min_count=5)
    multi = [seq for seq in seqs if len(seq) >= 2]
    return {
        "spans": len(seqs),
        "tokens": n_tokens,
        "mean_span_length": n_tokens / len(seqs),
        "distinct_signs": len(counts),
        "hapax_share_of_signs": sum(v == 1 for v in counts.values()) / len(counts),
        "h1": h1,
        "h2_plugin": h2,
        "h2_mm": h2mm,
        "h3_mm": h3mm,
        "h2_over_h1": h2 / h1 if h1 else float("nan"),
        "h2_shuffle_drop": null_h2 - h2,
        "bigram_mi": mutual_information(seqs, 1),
        "zipf_b": fit["b"],
        "zipf_r2": fit["r_squared"],
        "ppl_unigram_mkn": ppl1,
        "ppl_bigram_mkn": ppl2,
        "bigram_gain_bits": math.log2(ppl1 / ppl2),
        "restore_top1_frequency": restore["frequency"]["top_1"],
        "restore_top1_context": restore["context"]["top_1"],
        "restore_top1_gain": restore["context"]["top_1"] - restore["frequency"]["top_1"],
        "complete_spans": n_complete,
        "top_end_sign_share": (pos["end"].most_common(1)[0][1] / n_complete) if n_complete else None,
        "top_begin_sign_share": (pos["begin"].most_common(1)[0][1] / n_complete) if n_complete else None,
        "end_biased_sign_share": (sum(r["p_end"] >= 0.5 for r in profile) / len(profile)) if profile else None,
        "begin_biased_sign_share": (sum(r["p_begin"] >= 0.5 for r in profile) / len(profile)) if profile else None,
        "adjacent_repeat_rate": (sum(any(a == b for a, b in zip(s, s[1:])) for s in multi) / len(multi)) if multi else None,
    }


def aggregate(rows: list[dict]) -> dict:
    out = {}
    for key in rows[0]:
        values = [r[key] for r in rows if r[key] is not None]
        if values and all(isinstance(v, (int, float)) for v in values):
            out[key] = {"mean": fmean(values), "sd": pstdev(values) if len(values) > 1 else 0.0}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--comparison", type=Path, default=COMPARISON)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--corpora", nargs="*", default=list(CORPORA))
    parser.add_argument("--drop-numerals", action="store_true",
                        help="remove numeral-group tokens from the comparison corpora")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    indus = load(INDUS, 0, pool_tokens=None)
    target = sum(len(r["sequence"]) for r in indus)
    histogram = Counter(len(r["sequence"]) for r in indus)
    results = {"indus_tokens": target, "replicates": args.replicates,
               "drop_numerals": args.drop_numerals, "corpora": {}}

    rows = [measures(indus, seed) for seed in range(args.replicates)]
    results["corpora"]["indus"] = {"full": aggregate(rows)}
    print("indus", json.dumps({k: round(v["mean"], 3) for k, v in results["corpora"]["indus"]["full"].items()}))

    for name in args.corpora:
        records = load(args.comparison / name / "corpus.csv", 0)
        if args.drop_numerals:
            records = without_numerals(records)
        entry = {"pool_spans": len(records), "pool_tokens": sum(len(r["sequence"]) for r in records)}
        token_rows, length_rows, coverage = [], [], []
        for seed in range(args.replicates):
            token_rows.append(measures(token_matched(records, target, seed), seed))
            sample, share = length_matched(records, histogram, seed)
            coverage.append(share)
            length_rows.append(measures(sample, seed))
        entry["token_matched"] = aggregate(token_rows)
        entry["length_matched"] = aggregate(length_rows)
        entry["length_matched_token_coverage"] = fmean(coverage)
        results["corpora"][name] = entry
        print(name, json.dumps({k: round(v["mean"], 3) for k, v in entry["token_matched"].items()}))
        with open(args.out / "comparison_summary.json", "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
    write_report(results, args.out)


ROWS = [
    ("distinct_signs", "Distinct signs", "{:.0f}"),
    ("mean_span_length", "Mean span length", "{:.2f}"),
    ("hapax_share_of_signs", "Signs seen once (share)", "{:.2f}"),
    ("h1", "H1 (bits)", "{:.2f}"),
    ("h2_mm", "H(s2|s1), Miller-Madow (bits)", "{:.2f}"),
    ("h2_over_h1", "H2 / H1", "{:.3f}"),
    ("h2_shuffle_drop", "H2 drop vs within-span shuffle (bits)", "{:.3f}"),
    ("bigram_mi", "Bigram mutual information (bits)", "{:.2f}"),
    ("zipf_b", "Zipf-Mandelbrot exponent b", "{:.2f}"),
    ("bigram_gain_bits", "Held-out bigram gain over unigram, MKN (bits/token)", "{:.2f}"),
    ("restore_top1_frequency", "Restoration top-1, frequency baseline", "{:.1%}"),
    ("restore_top1_context", "Restoration top-1, bigram context", "{:.1%}"),
    ("top_end_sign_share", "Share of complete spans ending in the commonest end sign", "{:.1%}"),
    ("top_begin_sign_share", "Share of complete spans starting with the commonest start sign", "{:.1%}"),
    ("end_biased_sign_share", "Signs (n>=5) with p_end >= 0.5", "{:.1%}"),
    ("adjacent_repeat_rate", "Spans (len>=2) with an adjacent repeated sign", "{:.1%}"),
]
NAMES = {"indus": "Indus", "proto_cuneiform": "Proto-cuneiform", "proto_elamite": "Proto-Elamite",
         "ur3_admin": "Ur III admin", "seal_legends": "Sumerian seals"}


def write_report(results: dict, out: Path) -> None:
    names = list(results["corpora"])
    lines = ["# Indus statistics on comparison corpora (first pass)", ""]
    if results["drop_numerals"]:
        lines += ["Numeral-group tokens removed from the comparison corpora.", ""]
    for mode, title in (("token_matched", "Token-matched subsamples"),
                        ("length_matched", "Length-matched subsamples")):
        lines += [f"## {title}", "",
                  "| measure | " + " | ".join(NAMES.get(n, n) for n in names) + " |",
                  "|---|" + "---|" * len(names)]
        for key, label, fmt in ROWS:
            cells = []
            for name in names:
                block = results["corpora"][name].get("full" if name == "indus" else mode, {})
                cells.append(fmt.format(block[key]["mean"]) if key in block else "n/a")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        if mode == "length_matched":
            lines.append("")
            lines.append("Token coverage of the Indus length histogram: " + ", ".join(
                f"{NAMES.get(n, n)} {results['corpora'][n]['length_matched_token_coverage']:.0%}"
                for n in names if n != "indus"))
        lines.append("")
    lines.append(f"Indus is the full analysable corpus ({results['indus_tokens']} tokens); "
                 f"each comparison column is the mean of {results['replicates']} subsamples.")
    (out / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
