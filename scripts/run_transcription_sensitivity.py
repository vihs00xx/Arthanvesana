"""Same-family transcription sensitivity.

Tests whether the headline result (context bigram beats frequency/position)
survives same-family transcription conventions. Matches records by CISI,
restricts to unambiguous one-to-one matches, and uses IDENTICAL artifact-level
train/test assignments for both transcriptions (built from stable catalog
identities, not source-specific sequences).

Agreement between these ICIT-derived sources is NOT independent
inter-annotator agreement. Raw external sequences are not redistributed;
only per-CISI relation labels and aggregate counts are reported.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from statistics import fmean

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from arthanvesana.data.parse import analysis_records, sha256_file
from arthanvesana.replicate.robustness import evaluate_split
from arthanvesana.stats.sampling import split_records
from scripts.run_audit import normalize_gcode

EXT_DIRECTION_MAP = {"R-L": "L/R", "L-R": "R/L"}


def external_sequences(external):
    """Parse the external CSV into per-CISI sequence rows in reading order."""
    out = {}
    for _, row in external.iterrows():
        cisi = row["cisi_number"]
        if cisi == "unknown":
            continue
        codes = [normalize_gcode(c) for c in str(row["sign_sequence"]).split()]
        direction = EXT_DIRECTION_MAP.get(str(row["reading_direction"]), "OTHER")
        if direction == "R/L":
            codes = list(reversed(codes))
        out.setdefault(cisi, []).append({
            "cisi": cisi, "sequence": codes, "direction": direction,
            "site": str(row.get("site", "unknown")),
            "object_type": str(row.get("object_type", "unknown")),
        })
    return out


def relation(mine, theirs):
    """Audit-relation classification for one matched pair (raw comparison)."""
    if theirs == mine:
        return "exact"
    if [x for x in theirs if x != "000"] == [x for x in mine if x != "000"]:
        return "gap_placement_only"
    if theirs == mine[2:]:
        return "first_edge"
    if theirs == mine[:-2]:
        return "last_edge"
    return "other"


def match_records(frame, external):
    """Match primary to external by CISI; 1:1 unambiguous matches only.

    Returns (primary_records, external_records, exclusions). Each external
    record mirrors its primary counterpart's record-level fields (ids, artifact,
    direction, completeness flags) with the external sign sequence attached.
    """
    ours = {}
    for r in analysis_records(frame, gap_policy="split", known_direction_only=False):
        if r.get("cisi"):
            ours.setdefault(r["cisi"], []).append(r)
    theirs = external_sequences(external)
    primary, ext, exclusions = [], [], {"missing": 0, "duplicate": 0, "empty": 0}
    for cisi in sorted(set(ours) | set(theirs)):
        mine = ours.get(cisi, [])
        ext_rows = theirs.get(cisi, [])
        if not mine or not ext_rows:
            exclusions["missing"] += 1
            continue
        if len(ext_rows) != 1 or len({tuple(r["sequence"]) for r in mine}) != 1:
            # multiple external rows OR multiple primary variants for one CISI:
            # ambiguous sides/fragments, excluded by design
            exclusions["duplicate"] += 1
            continue
        ext_seq = ext_rows[0]["sequence"]
        if not ext_seq:
            exclusions["empty"] += 1
            continue
        for r in mine:
            primary.append(dict(r))
        base = mine[0]
        ext.append({
            "inscription_id": base["inscription_id"],
            "artifact_id": base["artifact_id"],
            "artifact_group": base["artifact_group"],
            "site": base["site"],
            "direction": base["direction"],
            "complete": base["complete"],
            "start_complete": base["start_complete"],
            "end_complete": base["end_complete"],
            "sequence": list(ext_seq),
            "exact": base["sequence"] == ext_seq,
            "audit_relation": relation(base["sequence"], ext_seq),
            "cisi": cisi,
        })
    return primary, ext, exclusions



def aligned_evaluate(primary, ext, seeds):
    """Evaluate both transcriptions on identical test sets.

    Splits the PRIMARY records; the external records are assigned to train/test
    by the same inscription ids, so both transcriptions receive the same test
    set. Partitions come from stable catalog identities (inscription id), not
    source-specific sequences.
    """
    out = []
    for seed in seeds:
        train_p, test_p, diagnostics = split_records(
            primary, 0.8, seed, track="artifact",
            group_duplicates=False,
            group_keys=[r["inscription_id"] for r in primary],
        )
        test_ids = {r["inscription_id"] for r in test_p}
        train_e = [r for r in ext if r["inscription_id"] not in test_ids]
        test_e = [r for r in ext if r["inscription_id"] in test_ids]
        res_p = res_e = None
        try:
            res_p = evaluate_split(train_p, test_p)
        except ValueError:
            pass
        try:
            res_e = evaluate_split(train_e, test_e)
        except ValueError:
            pass
        out.append({
            "seed": seed, "split": diagnostics,
            "test_ids": sorted(test_ids),
            "primary": res_p, "external": res_e,
        })
    return out


def _side_metrics(runs, side):
    ok = [r[side] for r in runs if r[side] is not None]
    if not ok:
        return {"n_runs": 0}
    block = {"n_runs": len(ok)}
    for model in ("frequency", "position", "context"):
        for metric in ("top_1", "top_5", "top_10", "mrr"):
            vals = [r["models"][model][metric] for r in ok]
            block[f"{model}_{metric}"] = {"mean": fmean(vals)}
    block["oov"] = {"mean": fmean(r["oov_fraction"] for r in ok)}
    for comp in ("context_vs_frequency", "context_vs_position"):
        vals = [r["paired_deltas"][comp]["top_1"] for r in ok]
        block[f"{comp}_delta"] = {
            "mean": fmean(vals),
            "n_positive": sum(v > 0 for v in vals),
            "n_zero": sum(v == 0 for v in vals),
            "n_negative": sum(v < 0 for v in vals),
        }
    return block



def render_report(summary):
    lines = [
        "Same-family transcription sensitivity",
        "",
        "Scientific question: does contextual information outperform frequency and",
        "position under BOTH same-family transcriptions?",
        f"Matched CISI: {summary['matched_cisi']}; primary spans: {summary['n_primary']}.",
        "Exclusions: " + "; ".join(
            f"{k}={summary['exclusions'][k]}" for k in sorted(summary["exclusions"])
        ),
        "Partitions are identical for both transcriptions (stable catalog ids).",
        "Agreement between these ICIT-derived sources is NOT independent",
        "inter-annotator agreement.",
        "",
    ]
    for side in ("primary", "external"):
        block = summary[side]
        if block.get("n_runs"):
            lines.append(
                f"{side}: context top-1 {block['context_top_1']['mean']:.4f}, "
                f"frequency {block['frequency_top_1']['mean']:.4f}, "
                f"position {block['position_top_1']['mean']:.4f}, "
                f"OOV {block['oov']['mean']:.4f}"
            )
            lines.append(
                f"  ctx-freq {block['context_vs_frequency_delta']['mean']:+.4f} "
                f"(w/t/l {block['context_vs_frequency_delta']['n_positive']}/"
                f"{block['context_vs_frequency_delta']['n_zero']}/"
                f"{block['context_vs_frequency_delta']['n_negative']}); "
                f"ctx-pos {block['context_vs_position_delta']['mean']:+.4f}"
            )
        else:
            lines.append(f"{side}: no successful runs")
    lines.extend([
        "",
        "Exact-agreement vs disagreement spans (descriptive):",
    ])
    for name, blk in summary["exact_vs_disagree"].items():
        lines.append(f"  {name}: n_test={blk['n_test_records']} {blk['note']}")
    lines.extend(["", "By audit relation (descriptive):"])
    for name, blk in summary["by_relation"].items():
        lines.append(f"  {name}: n_test={blk['n_test_records']} {blk['note']}")
    lines.extend([
        "",
        "These are artificial single-sign masks, not verified restorations.",
        "Raw external sequences are not redistributed.",
    ])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Same-family transcription sensitivity")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "processed" / "corpus.csv")
    parser.add_argument("--external", type=Path,
                        default=ROOT / "data" / "external" / "indus_website_real_corpus.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "transcription_sensitivity")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not args.external.exists():
        raise FileNotFoundError(
            f"External transcription file missing: {args.external}. "
            "Expected data/external/indus_website_real_corpus.csv "
            "(joyboseroy/indus_decipher, config indus_website). "
            "Provide it with --external; nothing is downloaded."
        )
    frame = pd.read_csv(args.corpus, encoding="utf-8", dtype={"sign_code": str})
    external = pd.read_csv(args.external, encoding="utf-8", dtype=str)
    seeds = list(range(args.seed, args.seed + args.repeats))
    primary, ext, exclusions = match_records(frame, external)
    runs = aligned_evaluate(primary, ext, seeds)

    summary = {
        "manifest": {
            "corpus_sha256": sha256_file(args.corpus),
            "external_sha256": sha256_file(args.external),
            "source_sha256": {
                "scripts/run_transcription_sensitivity.py": sha256_file(
                    Path(__file__).resolve()),
            },
            "python": platform.python_version(),
            "dependencies": {n: version(n) for n in ("numpy", "pandas", "scipy")},
            "seeds": seeds,
            "design": "identical artifact-level test sets from stable catalog ids",
            "note": "same-family comparison; not inter-annotator agreement",
        },
        "matched_cisi": len({r["cisi"] for r in ext}),
        "n_primary": len(primary),
        "n_external": len(ext),
        "exclusions": exclusions,
        "primary": _side_metrics(runs, "primary"),
        "external": _side_metrics(runs, "external"),
        "exact_vs_disagree": _breakdown(ext),
        "by_relation": _breakdown(ext, key="audit_relation"),
        "runs": [
            {"seed": r["seed"], "test_ids": r["test_ids"],
             "primary_ok": r["primary"] is not None,
             "external_ok": r["external"] is not None}
            for r in runs
        ],
    }
    report = render_report(summary)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "transcription_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (args.output / "transcription_sensitivity_report.txt").write_text(report, encoding="utf-8")
    print(report)
    return summary


def _breakdown(ext, key="exact"):
    """Span counts per agreement label, shared by both transcriptions."""
    out = {}
    for r in ext:
        label = str(r.get("audit_relation") == "exact") if key == "exact" else str(r[key])
        out.setdefault(label, set()).add(r["inscription_id"])
    return {k: {"n_test_records": len(v),
                "note": "descriptive; per-label metrics inherit the aligned design"}
            for k, v in sorted(out.items())}


if __name__ == "__main__":
    main()