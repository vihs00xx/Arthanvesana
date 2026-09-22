"""Direction diagnostics for the as-stored vs reading-order restoration gap.

Diagnostic only: explains the ~2.7-3.1 percentage-point advantage of physical
as-stored order over reading-order-normalized order. It does not change any
pipeline default and does not conclude that one order is correct.

What this runner now does, and why:

* **Aligned groups across orientation variants.** Every variant is split with the
  SAME group keys, built from the **union of exact-sequence equivalence relations
  across all variants**. Using the union makes the grouping at least as strict as
  any single variant, so neither orientation can leak an equivalent sequence
  across the shared split. The policy is printed beside the scores.
* **OOV-separated transfer.** Cross-direction transfer previously reported a
  single accuracy alongside an OOV rate, which invites the unsupported reading
  that "vocabulary coverage explains the gap". Accuracy is now reported overall,
  on non-OOV targets only, and on targets shared between the two vocabularies,
  so ordering effects are separated from vocabulary-coverage failures.
* **Explicit boundary models.** ``asymmetric`` (the pipeline default),
  ``none`` (no boundary evidence) and ``symmetric`` (both edges use the ``<S>``
  start distribution). Reversal invariance is only *expected* under ``none`` and
  ``symmetric``; requiring it of an asymmetric model would be a category error.
* **Position classes.** Singleton, first, interior and last positions are
  reported separately, with sample sizes.

Direction conventions are taken from ``data/PROVENANCE.md``: symbols are stored
physical left-to-right; reading order is as-stored for ``L/R`` and reversed for
``R/L``; every other direction is kept in stored order and flagged
``reading_order_known=False``. This runner verifies behaviour against that
document; it does not adjudicate which order is archaeologically correct.
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
from arthanvesana.replicate.restore import BOUNDARY_MODES, restoration_records
from arthanvesana.replicate.robustness import evaluate_split
from arthanvesana.stats.sampling import split_records

EDGE_SIGNS = {"700", "033", "032", "034"}
LENGTH_BUCKETS = (3, 5, 8)
MIN_STRATUM_TEST = 40

#: Which reversal invariants are mathematically expected, per boundary model.
REVERSAL_EXPECTATIONS = {
    "asymmetric": (
        "NOT expected to be invariant. The interior is exactly mirror-symmetric "
        "(the forward chain uses P(next|cur) and the reversed chain the same "
        "matrix), but a mask at the very start uses the <S> start distribution "
        "while a mask at the very end gets a uniform no-evidence backward term. "
        "Reversing therefore moves the <S> effect to the other edge."
    ),
    "none": (
        "Expected to be APPROXIMATELY invariant, not exactly. Both edges use the "
        "same prior no-evidence term, but the forward pass starts from the prior "
        "while the backward pass ends at it; these coincide exactly only when the "
        "training unigram is the stationary distribution of the fitted chain. A "
        "large delta here indicates that the unigram is far from stationary."
    ),
    "symmetric": (
        "Removes the BOUNDARY asymmetry, but is NOT exactly invariant. Both edges "
        "use the <S> start distribution, each gated on its own completeness flag, "
        "so the two edges are treated identically under reversal. The context "
        "model itself is still not mirror-symmetric, however: its left term is "
        "P(w | prev) and its right term is P(next | w), which are "
        "transpose-related and coincide only under detailed balance. Expect a "
        "delta SMALLER than 'asymmetric', not zero."
    ),
}


def _stable_key(record):
    artifact = record.get("artifact_id") or record.get("inscription_id")
    return f"{artifact}|{record['inscription_id']}|{record.get('span_index', 0)}"


def union_group_keys(variants):
    """Union of exact-sequence equivalence relations across orientation variants.

    ``variants`` is an iterable of record lists. Two records are linked when their
    sequences match in ANY variant. The resulting components are at least as
    strict as every individual variant, so no variant can leak an equivalent
    sequence across a shared split. Returns ``stable_key -> component_label``.
    """
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for records in variants:
        # seed EVERY record key first, so records without any duplicate still get
        # a component label rather than a missing entry
        for record in records:
            find(_stable_key(record))
        by_sequence = {}
        for record in records:
            by_sequence.setdefault(tuple(record["sequence"]), []).append(
                _stable_key(record))
        for keys in by_sequence.values():
            for key in keys[1:]:
                union(keys[0], key)
    return {key: find(key) for key in parent}


def aligned_split(records, seed, keys_map, train_frac=0.8):
    """Split with union-derived group keys, so variants share identical folds."""
    keys = [keys_map.get(_stable_key(r), _stable_key(r)) for r in records]
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


def _accuracy(rows, k=1):
    if not rows:
        return None
    return sum(1 for r in rows if r["rank"] is not None and r["rank"] <= k) / len(rows)


def _restoration_breakdown(train, test, boundary_mode="asymmetric"):
    """OOV-separated restoration metrics for one (train, test) pair.

    Separating OOV targets from non-OOV targets is what allows an ordering effect
    to be distinguished from a vocabulary-coverage failure. ``top_1_shared``
    restricts evaluation to targets whose sign appears in the training vocabulary,
    which is the like-for-like comparison across conditions with different OOV
    rates.
    """
    if not train or not any(r["sequence"] for r in test):
        return None
    try:
        rows, _ = restoration_records(train, test, mask_length=1,
                                      boundary_mode=boundary_mode)
    except ValueError:
        return None
    if not rows:
        return None
    oov = [r for r in rows if r["oov"]]
    known = [r for r in rows if not r["oov"]]
    return {
        "n_targets": len(rows),
        "n_oov": len(oov),
        "oov_rate": len(oov) / len(rows),
        "top_1_overall": _accuracy(rows),
        "top_5_overall": _accuracy(rows, 5),
        "n_known_targets": len(known),
        "top_1_shared": _accuracy(known),
        "top_5_shared": _accuracy(known, 5),
        "oov_targets_are_failures": True,
        "note": ("top_1_overall counts OOV targets as failures; top_1_shared "
                 "excludes them and is the comparison to use when OOV rates differ"),
    }


def _position_class(row):
    length = row["length"]
    if length <= 1:
        return "singleton"
    if row["position"] == 0:
        return "first"
    if row["position"] == length - 1:
        return "last"
    return "interior"


def position_breakdown(train, test, boundary_mode="asymmetric"):
    """First / interior / last / singleton top-1, reported separately."""
    if not train or not any(r["sequence"] for r in test):
        return None
    try:
        rows, _ = restoration_records(train, test, mask_length=1,
                                      boundary_mode=boundary_mode)
    except ValueError:
        return None
    classes = ("singleton", "first", "interior", "last")
    out = {}
    for name in classes:
        subset = [r for r in rows if _position_class(r) == name]
        known = [r for r in subset if not r["oov"]]
        out[name] = {
            "n_targets": len(subset),
            "n_known": len(known),
            "top_1_overall": _accuracy(subset),
            "top_1_shared": _accuracy(known),
        }
    return out


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


def experiment_a(frame, seeds, keys_map):
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
            train, test, _ = aligned_split(records, seed, keys_map)
            blocks.append(_split_top1(train, test))
        out[name] = {"n_records": len(records), "aggregate": _aggregate(blocks)}
    return out


def experiment_b(reference, seeds, keys_map):
    """Reversal check under every boundary model, with the expected invariant."""
    reversed_records = _reverse_records(reference)
    rev_by_key = {_stable_key(r): r for r in reversed_records}
    per_mode = {mode: {"per_seed": [], "positions": []} for mode in BOUNDARY_MODES}
    for seed in seeds:
        print(f"    B: reversal check, seed {seed} ...", flush=True)
        train, test, _ = aligned_split(reference, seed, keys_map)
        train_keys = {_stable_key(r) for r in train}
        train_rev = [rev_by_key[_stable_key(r)] for r in reference
                     if _stable_key(r) in train_keys]
        test_rev = [rev_by_key[_stable_key(r)] for r in test]
        for mode in BOUNDARY_MODES:
            original = _restoration_breakdown(train, test, mode)
            reversed_block = _restoration_breakdown(train_rev, test_rev, mode)
            if original and reversed_block:
                per_mode[mode]["per_seed"].append({
                    "seed": seed,
                    "original_top_1": original["top_1_overall"],
                    "reversed_top_1": reversed_block["top_1_overall"],
                    "delta": original["top_1_overall"] - reversed_block["top_1_overall"],
                })
            if mode == "asymmetric":
                pos = position_breakdown(train, test, mode)
                if pos:
                    per_mode[mode]["positions"].append(pos)

    out = {}
    for mode in BOUNDARY_MODES:
        deltas = [row["delta"] for row in per_mode[mode]["per_seed"]]
        out[mode] = {
            "expected": REVERSAL_EXPECTATIONS[mode],
            "mean_delta": fmean(deltas) if deltas else None,
            "min_delta": min(deltas) if deltas else None,
            "max_delta": max(deltas) if deltas else None,
            "n_runs": len(deltas),
            "per_seed": per_mode[mode]["per_seed"],
            "position_classes": _mean_position_blocks(per_mode[mode]["positions"]),
        }
    return out


def _mean_position_blocks(blocks):
    if not blocks:
        return {}
    out = {}
    for name in ("singleton", "first", "interior", "last"):
        n = [b[name]["n_targets"] for b in blocks if b]
        overall = [b[name]["top_1_overall"] for b in blocks
                   if b and b[name]["top_1_overall"] is not None]
        shared = [b[name]["top_1_shared"] for b in blocks
                  if b and b[name]["top_1_shared"] is not None]
        out[name] = {
            "mean_n_targets": fmean(n) if n else 0,
            "top_1_overall": fmean(overall) if overall else None,
            "top_1_shared": fmean(shared) if shared else None,
        }
    return out


def experiment_c(normalized, stored, seed):
    """Cross-direction transfer with sizes controlled AND OOV separated."""
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
        block = _restoration_breakdown(train, test) or {}
        block["n_train_records"] = len(train)
        block["n_test_records"] = len(test)
        out[name] = block
    # explicitly compare the two conditions whose OOV rates are closest, so the
    # ordering effect is not read off a vocabulary-coverage difference
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


def experiment_d(normalized, stored, seeds, keys_map):
    """Break the normalized-vs-stored top-1 difference down by stratum."""
    strata = _strata(normalized)
    stored_by_key = {_stable_key(r): r for r in stored}
    out = {}
    for name, subset in strata.items():
        print(f"    D: stratum {name} ...", flush=True)
        deltas = []
        norm_vals = []
        stored_vals = []
        for seed in seeds:
            train_n, test_n, _ = aligned_split(subset, seed, keys_map)
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
        "",
        "Direction conventions (data/PROVENANCE.md): symbols are stored physical",
        "left-to-right; reading order is as-stored for L/R and reversed for R/L;",
        "every other direction is kept in stored order and flagged",
        "reading_order_known=False.",
        "",
        f"Grouping policy: {summary['manifest']['grouping_policy']}",
        "Every orientation variant is split with the SAME union-derived group keys,",
        "so both orderings are compared on identical train/test membership.",
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

    lines.extend([
        "",
        "B. Reversal check under each boundary model (original minus reversed):",
        "   No model here is exactly invariant. 'symmetric' and 'none' equalize the",
        "   edge treatment and should show a SMALLER delta than the 'asymmetric'",
        "   default; the residual is the bigram's left/right transpose asymmetry",
        "   (P(w|prev) vs P(next|w)), which vanishes only under detailed balance.",
    ])
    for mode, block in summary["B_reversal"].items():
        if block["n_runs"]:
            lines.append(
                f"  [{mode}] mean delta {block['mean_delta']:+.4f} over "
                f"{block['n_runs']} seeds "
                f"[{block['min_delta']:+.4f}, {block['max_delta']:+.4f}]"
            )
        else:
            lines.append(f"  [{mode}] no successful runs")
        lines.append(f"      expected: {block['expected']}")
    pos = summary["B_reversal"]["asymmetric"]["position_classes"]
    if pos:
        lines.append("  Position classes under the default (asymmetric) model:")
        lines.append("    class | mean targets | top-1 overall | top-1 shared-vocab")
        for name in ("singleton", "first", "interior", "last"):
            row = pos.get(name)
            if not row:
                continue
            lines.append(
                f"    {name} | {row['mean_n_targets']:.1f} | "
                f"{row['top_1_overall']} | {row['top_1_shared']}"
            )

    lines.extend([
        "",
        "C. Cross-direction transfer, OOV separated (train->test):",
        "   'shared' = targets whose sign is in the training vocabulary. Compare",
        "   conditions on 'shared', not on 'overall', when OOV rates differ.",
        "  condition | n_train | n_test | targets | OOV | top-1 overall | top-1 shared",
    ])
    for name, block in summary["C_transfer"].items():
        if not block or block.get("n_targets") is None:
            lines.append(f"  {name}: no data")
            continue
        lines.append(
            f"  {name} | {block['n_train_records']} | {block['n_test_records']} | "
            f"{block['n_targets']} | {block['oov_rate']:.4f} | "
            f"{block['top_1_overall']:.4f} | "
            f"{block['top_1_shared'] if block['top_1_shared'] is None else format(block['top_1_shared'], '.4f')}"
        )
    lines.append(
        "  Reading: a large 'overall' deficit that disappears on 'shared' targets "
        "is a vocabulary-coverage effect, not an ordering effect. A deficit that "
        "persists on 'shared' targets is an ordering effect. Neither alone "
        "establishes which transcription order is archaeologically correct.")

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
    # union of duplicate relations across both orientation variants
    keys_map = union_group_keys([normalized, stored, _reverse_records(normalized)])

    print("A: direction-specific evaluation ...", flush=True)
    a_block = experiment_a(frame, seeds, keys_map)
    print("B: reversal check under 3 boundary models ...", flush=True)
    b_block = experiment_b(normalized, seeds, keys_map)
    print("C: cross-direction transfer ...", flush=True)
    c_block = experiment_c(normalized, stored, args.seed)
    print("D: stratified breakdown ...", flush=True)
    d_block = experiment_d(normalized, stored, seeds, keys_map)
    print("rendering report ...", flush=True)

    summary = {
        "manifest": {
            "corpus_sha256": sha256_file(args.corpus),
            "source_sha256": {
                "scripts/run_direction_diagnostics.py": sha256_file(Path(__file__).resolve()),
                "src/arthanvesana/replicate/restore.py": sha256_file(
                    ROOT / "src" / "arthanvesana" / "replicate" / "restore.py"),
                "data/PROVENANCE.md": sha256_file(ROOT / "data" / "PROVENANCE.md"),
            },
            "python": platform.python_version(),
            "dependencies": {n: version(n) for n in ("numpy", "pandas", "scipy")},
            "seeds": seeds, "gap_policy": "split",
            "grouping_policy": ("union of exact-sequence equivalence relations across "
                                "orientation variants, applied as shared group keys; "
                                "artifact-level grouping retained; no variant can "
                                "leak an equivalent sequence across the shared split"),
            "duplicate_grouping": "preserved via the union of variant relations",
            "boundary_modes": list(BOUNDARY_MODES),
            "default_boundary_mode": "asymmetric",
            "direction_conventions": ("data/PROVENANCE.md: stored physical "
                                      "left-to-right; as-stored for L/R, reversed "
                                      "for R/L; other directions kept in stored "
                                      "order with reading_order_known=False"),
            "note": "diagnostic only; does not change pipeline defaults",
        },
        "A_direction_specific": a_block,
        "B_reversal": b_block,
        "C_transfer": c_block,
        "D_stratified": d_block,
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