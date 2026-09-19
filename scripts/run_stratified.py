from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from arthanvesana.data.parse import analysis_records, sha256_file
from arthanvesana.replicate.robustness import evaluate_split
from arthanvesana.stats.sampling import split_records
from scripts.build_metadata import attach_metadata

MOTIF_GROUPS = {
    "bull": lambda m: m is not None and ("Bull" in m),
    "other_known": lambda m: m is not None and m != "unknown" and "Bull" not in m,
    "unknown": lambda m: m is None or m == "unknown",
}

DIRECTION_MAP = {"R-L": "L/R", "L-R": "R/L"}


def direction_agreement(frame: pd.DataFrame, sidecar: pd.DataFrame) -> dict:
    ours = frame.drop_duplicates("cisi", keep="first").set_index("cisi")["direction"]
    known = sidecar[sidecar["cisi"].isin(set(ours.index))]
    pairs = [
        (ours.loc[cisi], DIRECTION_MAP.get(ext))
        for cisi, ext in zip(known["cisi"], known["ext_direction"])
    ]
    comparable = [(a, b) for a, b in pairs if a in ("L/R", "R/L") and b is not None]
    agree = sum(a == b for a, b in comparable)
    return {
        "n_compared": len(comparable),
        "n_agree": agree,
        "agreement": agree / len(comparable) if comparable else None,
    }


def render_report(summary):
    lines = [
        "Metadata-stratified evaluation",
        "",
        "External motif and direction fields joined by CISI number; sidecar",
        "covers 41.6% of corpus inscriptions. The external damage flag is",
        "uniformly False and excluded; line counts are uniformly 1.",
        "",
        f"Direction cross-check: {summary['direction']['n_agree']}/"
        f"{summary['direction']['n_compared']} agree "
        f"({summary['direction']['agreement']:.3f}).",
        "External R-L (as-stored) maps to our L/R; L-R maps to our R/L.",
        "",
        "Motif groups (single seed-0 artifact-grouped split each):",
        "Group | Spans | Context top-1 | Frequency top-1 | Position top-1 | OOV",
    ]
    for name, result in summary["motif_groups"].items():
        status = result["status"]
        if status != "ok":
            lines.append(f"{name} | skipped ({result['reason']})")
            continue
        models = result["evaluation"]["models"]
        lines.append(
            f"{name} | {result['n_test']} | {models['context']['top_1']:.4f} | "
            f"{models['frequency']['top_1']:.4f} | {models['position']['top_1']:.4f} | "
            f"{result['evaluation']['oov_fraction']:.4f}"
        )
    lines.extend([
        "",
        "Groups are descriptive subsets, not independent experiments; small",
        "groups carry high split noise. Full metrics and split identities are",
        "in stratified_summary.json.",
    ])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Metadata-stratified restoration check")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "processed" / "corpus.csv")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data" / "processed" / "inscription_metadata.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "stratified")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.corpus, encoding="utf-8", dtype=str)
    sidecar = pd.read_csv(args.metadata, encoding="utf-8", dtype=str)
    records = attach_metadata(
        analysis_records(frame, gap_policy="split", known_direction_only=True), sidecar
    )
    agreement = direction_agreement(frame, sidecar)
    groups = {}
    for name, predicate in MOTIF_GROUPS.items():
        subset = [r for r in records if predicate(r["ext_motif"])]
        if len(subset) < 10:
            groups[name] = {"status": "skipped", "reason": "fewer than 10 spans"}
            continue
        train, test, diagnostics = split_records(
            subset, 0.8, args.seed, track="artifact", group_duplicates=True
        )
        try:
            evaluation = evaluate_split(train, test)
        except ValueError as exc:
            groups[name] = {"status": "skipped", "reason": str(exc)}
            continue
        groups[name] = {
            "status": "ok", "reason": None, "evaluation": evaluation,
            "n_train": len(train), "n_test": len(test),
            "split": diagnostics,
        }
    summary = {
        "manifest": {
            "corpus_sha256": sha256_file(args.corpus),
            "metadata_sha256": sha256_file(args.metadata),
            "python": platform.python_version(),
            "dependencies": {n: version(n) for n in ("numpy", "pandas", "scipy")},
            "seed": args.seed,
        },
        "direction": agreement,
        "motif_groups": groups,
    }
    report = render_report(summary)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "stratified_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (args.output / "stratified_report.txt").write_text(report, encoding="utf-8")
    print(report)
    return summary


if __name__ == "__main__":
    main()
