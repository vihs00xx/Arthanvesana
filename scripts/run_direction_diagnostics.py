"""Direction diagnostics for the as-stored vs reading-order restoration gap.

Diagnostic only: explains the ~2.7-3.1 percentage-point advantage of physical
as-stored order over reading-order-normalized order. It does not change any
pipeline default and does not conclude that one order is correct.

Partitions are aligned across orderings with sequence-independent group keys
(artifact + inscription + span identity, group_duplicates=False) so both
orderings are compared on identical train/test membership. Exact-sequence
duplicate chaining is therefore not applied here; artifact-level grouping is
retained. This limitation is stated in the report.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from importlib.metadata import version
from pathlib import Path
from statistics import fmean

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arthanvesana.data.parse import analysis_records, parse_bool, sha256_file
from arthanvesana.replicate.restore import restoration_records
from arthanvesana.replicate.robustness import evaluate_split
from arthanvesana.stats.sampling import split_records

EDGE_SIGNS = {"700", "033", "032", "034"}
LENGTH_BUCKETS = (3, 5, 8)
MIN_STRATUM_TEST = 40


def _stable_key(record):
    artifact = record.get("artifact_id") or record.get("inscription_id")
    return f"{artifact}|{record['inscription_id']}|{record.get('span_index', 0)}"


def aligned_split(records, seed, train_frac=0.8):
    keys = [_stable_key(r) for r in records]
    return split_records(
        records, train_frac, seed, track="artifact",
        group_duplicates=False, group_keys=keys,
    )


def _split_top1(train, test):
    if not train or not any(r["sequence"] for r in test):
        return None
    try:
        result = evaluate_split(train, test)
    except ValueError:
        return None
    return {
        "top_1": result["models"]["context"]["top_1"],
        "top_5": result["models"]["context"]["top_5"],
        "top_10": result["models"]["context"]["top_10"],
        "mrr": result["models"]["context"]["mrr"],
        "oov_fraction": result["oov_fraction"],
        "n_test_records": result["n_test_records"],
        "n_masked": result["n_masked"],
    }


def _aggregate(blocks):
    blocks = [b for b in blocks if b is not None]
    if not blocks:
        return {"n_runs": 0}
    out = {"n_runs": len(blocks)}
    for key in ("top_1", "top_5", "top_10", "mrr", "oov_fraction"):
        vals = [b[key] for b in blocks]
        out[key] = {"mean": fmean(vals), "min": min(vals), "max": max(vals)}
    return out


def _subset(records, direction):
    return [r for r in records if r.get("direction") == direction]


def _reverse_records(records):
    """Reverse each sequence and swap start/end completeness so semantics hold."""
    return [
        dict(
            r,
            sequence=list(reversed(r["sequence"])),
            start_complete=parse_bool(r.get("end_complete")),
            end_complete=parse_bool(r.get("start_complete")),
        )
        for r in records
    ]

def experiment_a(frame, seeds):
    """Direction-specific evaluation under identical grouped conventions."""
    normalized = analysis_records(frame, gap_policy="split", reading_order=True,
                                 known_direction_only=True)
    stored = analysis_records(frame, gap_policy="split", reading_order=False,
                              known_direction_only=True)
    all_stored = analysis_records(frame, gap_policy="split", reading_order=False,
                                  known_direction_only=False)
    subsets = {
        "LR_normalized": _subset(normalized, "L/R"),
        "RL_as_stored": _subset(stored, "R/L"),
        "RL_after_reversal": _subset(normalized, "R/L"),
        "OTHER_as_stored": _subset(all_stored, "OTHER"),
    }
    out = {}
    for name, records in subsets.items():
        blocks = []
        for seed in seeds:
            train, test, _ = aligned_split(records, seed)
            blocks.append(_split_top1(train, test))
        out[name] = {"n_records": len(records), "aggregate": _aggregate(blocks)}
    return out


def _edge_top1(rank_pairs, positions):
    selected = [r for pos, r in rank_pairs if pos in positions]
    if not selected:
        return None
    return sum(r is not None and r <= 1 for r in selected) / len(selected)


def _boundary_probe(train, test, train_rev, test_rev):
    rows = []
    for label, tr, te in (("original", train, test), ("reversed", train_rev, test_rev)):
        if not tr or not any(r["sequence"] for r in te):
            continue
        try:
            records, _ = restoration_records(tr, te, mask_length=1)
        except ValueError:
            continue
        interior = [(row["position"], row["rank"]) for row in records
                    if 0 < row["position"] < row["length"] - 1]
        first = [(row["position"], row["rank"]) for row in records
                 if row["position"] == 0]
        last = [(row["position"], row["rank"]) for row in records
                if row["position"] == row["length"] - 1 and row["length"] > 1]
        rows.append({
            "orientation": label,
            "interior_top_1": _edge_top1(interior, {1}),
            "first_position_top_1": _edge_top1(first, {0}),
            "last_position_top_1": _edge_top1(last, {0}),
        })
    return {"probe": rows}


def experiment_b(reference, seeds):
    """Global reversal sanity check (flags swapped) on identical partitions."""
    reversed_records = _reverse_records(reference)
    rev_by_key = {_stable_key(r): r for r in reversed_records}
    per_seed = []
    boundary = []
    for seed in seeds:
        train, test, _ = aligned_split(reference, seed)
        train_keys = {_stable_key(r) for r in train}
        train_rev = [rev_by_key[_stable_key(r)] for r in reference
                     if _stable_key(r) in train_keys]
        test_rev = [rev_by_key[_stable_key(r)] for r in test]
        original = _split_top1(train, test)
        reversed_block = _split_top1(train_rev, test_rev)
        if original and reversed_block:
            per_seed.append({
                "seed": seed,
                "original_top_1": original["top_1"],
                "reversed_top_1": reversed_block["top_1"],
                "delta": original["top_1"] - reversed_block["top_1"],
            })
        boundary.append(_boundary_probe(train, test, train_rev, test_rev))
    deltas = [row["delta"] for row in per_seed]
    return {
        "per_seed": per_seed,
        "aggregate_delta": {
            "mean": fmean(deltas) if deltas else None,
            "min": min(deltas) if deltas else None,
            "max": max(deltas) if deltas else None,
            "n_runs": len(deltas),
        },
        "boundary_probe": [b for b in boundary if b["probe"]],
        "cause": (
            "Restoration is symmetric in the interior (the forward chain uses "
            "P(next|cur) and the reversed chain uses the same matrix), but the "
            "sequence START uses the <S> start distribution while the END contributes "
            "a uniform no-evidence backward term in restore._mask_distributions. "
            "Global reversal moves the <S> effect to the other edge, so boundary "
            "positions need not be exactly invariant."
        ),
    }


def experiment_c(normalized, stored, seed):
    """Cross-direction transfer with training sizes controlled."""
    classes_norm = {"L/R": _subset(normalized, "L/R"), "R/L": _subset(normalized, "R/L")}
    classes_stored = {"L/R": _subset(stored, "L/R"), "R/L": _subset(stored, "R/L")}
    pairs = [
        ("normalized_LR_to_RL", classes_norm["L/R"], classes_norm["R/L"]),
        ("normalized_RL_to_LR", classes_norm["R/L"], classes_norm["L/R"]),
        ("stored_LR_to_RL", classes_stored["L/R"], classes_stored["R/L"]),
        ("reversed_RL_train_to_LR", _reverse_records(classes_norm["R/L"]),
         classes_norm["L/R"]),
    ]
    out = {}
    rng = random.Random(seed)
    for name, train, test in pairs:
        train = list(train)
        if len(train) > len(test):
            train = rng.sample(train, len(test))
        out[name] = _split_top1(train, test) or {"error": "no data"}
    return out

def _length_bucket(length):
    for edge in LENGTH_BUCKETS:
        if length <= edge:
            return f"len<={edge}"
    return f"len>{LENGTH_BUCKETS[-1]}"


def _strata(records):
    """Named record filters for the normalized-vs-stored gap breakdown."""
    strata = {}

    def add(name, predicate):
        subset = [r for r in records if predicate(r)]
        if len(subset) >= MIN_STRATUM_TEST:
            strata[name] = subset

    for site in sorted({r.get("site") for r in records if r.get("site")}):
        add(f"site={site}", lambda r, s=site: r.get("site") == s)
    for art in sorted({r.get("artefact_type") for r in records if r.get("artefact_type")}):
        add(f"artefact={art}", lambda r, a=art: r.get("artefact_type") == a)
    for bucket in [f"len<={e}" for e in LENGTH_BUCKETS] + [f"len>{LENGTH_BUCKETS[-1]}"]:
        add(bucket, lambda r, b=bucket: _length_bucket(len(r["sequence"])) == b)
    add("complete", lambda r: parse_bool(r.get("complete")))
    add("start_and_end_complete",
        lambda r: parse_bool(r.get("start_complete")) and parse_bool(r.get("end_complete")))
    for direction in ("L/R", "R/L"):
        add(f"direction={direction}", lambda r, d=direction: r.get("direction") == d)
    add("has_edge_sign", lambda r: bool(EDGE_SIGNS & set(r["sequence"])))
    add("no_edge_sign", lambda r: not (EDGE_SIGNS & set(r["sequence"])))
    return strata


def experiment_d(normalized, stored, seeds):
    """Break the normalized-vs-stored top-1 difference down by stratum."""
    strata = _strata(normalized)
    stored_by_key = {_stable_key(r): r for r in stored}
    out = {}
    for name, subset in strata.items():
        deltas = []
        norm_vals = []
        stored_vals = []
        for seed in seeds:
            train_n, test_n, _ = aligned_split(subset, seed)
            train_keys = {_stable_key(r) for r in train_n}
            test_keys = {_stable_key(r) for r in test_n}
            train_s = [stored_by_key[k] for k in train_keys if k in stored_by_key]
            test_s = [stored_by_key[k] for k in test_keys if k in stored_by_key]
            block_n = _split_top1(train_n, test_n)
            block_s = _split_top1(train_s, test_s)
            if block_n and block_s:
                norm_vals.append(block_n["top_1"])
                stored_vals.append(block_s["top_1"])
                deltas.append(block_n["top_1"] - block_s["top_1"])
        if deltas:
            out[name] = {
                "n_records": len(subset),
                "n_runs": len(deltas),
                "normalized_top_1": fmean(norm_vals),
                "stored_top_1": fmean(stored_vals),
                "delta_normalized_minus_stored": fmean(deltas),
            }
    return out

def render_report(summary):
    lines = [
        "Direction diagnostics: as-stored vs reading-order restoration gap",
        "",
        "Diagnostic only. The pipeline default is unchanged, and a higher score for",
        "one ordering is NOT treated as evidence that the other order is wrong.",
        "Partitions are aligned across orderings with sequence-independent group keys",
        "(artifact+inscription+span), so exact-sequence duplicate chaining is not",
        "applied here; artifact-level grouping is retained. This is a limitation.",
        "",
        "A. Direction-specific evaluation (context bigram top-1):",
    ]
    for name, block in summary["A_direction_specific"].items():
        agg = block["aggregate"]
        if agg.get("n_runs"):
            top1 = agg["top_1"]
            lines.append(
                f"  {name}: n={block['n_records']} records, "
                f"top-1 {top1['mean']:.4f} [{top1['min']:.4f}, {top1['max']:.4f}], "
                f"OOV {agg['oov_fraction']['mean']:.4f}"
            )
        else:
            lines.append(f"  {name}: n={block['n_records']} records, skipped")
    b = summary["B_reversal"]
    lines.extend([
        "",
        "B. Global reversal sanity check (reverse every sequence; swap start/end flags):",
    ])
    agg = b["aggregate_delta"]
    if agg["n_runs"]:
        lines.append(
            f"  mean top-1 (original minus reversed) = {agg['mean']:+.4f} "
            f"over {agg['n_runs']} seeds [{agg['min']:+.4f}, {agg['max']:+.4f}]"
        )
    lines.append(f"  cause: {b['cause']}")
    for probe in b["boundary_probe"][:1]:
        for row in probe["probe"]:
            lines.append(
                f"  probe {row['orientation']}: interior top-1 {row['interior_top_1']}, "
                f"first-position top-1 {row['first_position_top_1']}, "
                f"last-position top-1 {row['last_position_top_1']}"
            )
    lines.extend(["", "C. Cross-direction transfer (train->test, sizes controlled):"])
    for name, block in summary["C_transfer"].items():
        if "error" in block:
            lines.append(f"  {name}: {block['error']}")
        else:
            lines.append(
                f"  {name}: top-1 {block['top_1']:.4f}, OOV {block['oov_fraction']:.4f}"
            )
    lines.extend([
        "",
        "D. Normalized-minus-stored top-1 difference by stratum (exploratory):",
        "  stratum | n_records | normalized | stored | delta (norm - stored)",
    ])
    for name, row in sorted(summary["D_stratified"].items(),
                            key=lambda kv: kv[1]["delta_normalized_minus_stored"]):
        lines.append(
            f"  {name} | {row['n_records']} | {row['normalized_top_1']:.4f} | "
            f"{row['stored_top_1']:.4f} | {row['delta_normalized_minus_stored']:+.4f}"
        )
    lines.extend([
        "",
        "Interpretation limits: the as-stored advantage may reflect direction",
        "interpretation, editorial conventions, artifact-type composition, duplicate",
        "structure, physical layout regularities, or a real difference between",
        "direction classes. These diagnostics narrow the candidates but do not",
        "establish which transcription order is archaeologically correct.",
        "SD/intervals describe split variability, NOT confidence intervals.",
    ])
    return "\n".join(lines) + "\n"

def main(argv=None):
    parser = argparse.ArgumentParser(description="Direction diagnostics for restoration order")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "processed" / "corpus.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "direction_diagnostics")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.corpus, encoding="utf-8", dtype={"sign_code": str})
    seeds = list(range(args.seed, args.seed + args.repeats))
    normalized = analysis_records(frame, gap_policy="split", reading_order=True,
                                 known_direction_only=True)
    stored = analysis_records(frame, gap_policy="split", reading_order=False,
                             known_direction_only=True)
    summary = {
        "manifest": {
            "corpus_sha256": sha256_file(args.corpus),
            "source_sha256": {
                "scripts/run_direction_diagnostics.py": sha256_file(Path(__file__).resolve()),
                "src/arthanvesana/replicate/restore.py": sha256_file(
                    ROOT / "src" / "arthanvesana" / "replicate" / "restore.py"),
            },
            "python": platform.python_version(),
            "dependencies": {n: version(n) for n in ("numpy", "pandas", "scipy")},
            "seeds": seeds, "gap_policy": "split",
            "alignment": "sequence-independent group keys (artifact+inscription+span)",
            "note": "diagnostic only; does not change pipeline defaults",
        },
        "A_direction_specific": experiment_a(frame, seeds),
        "B_reversal": experiment_b(normalized, seeds),
        "C_transfer": experiment_c(normalized, stored, args.seed),
        "D_stratified": experiment_d(normalized, stored, seeds),
    }
    report = render_report(summary)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "direction_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (args.output / "direction_diagnostics_report.txt").write_text(report, encoding="utf-8")
    print(report)
    return summary


if __name__ == "__main__":
    main()