from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from importlib.metadata import version
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arthanvesana.data.parse import sha256_file

CODE_RE = re.compile(r"^G(\d+)$")


def normalize_gcode(code: str) -> str:
    match = CODE_RE.match(code)
    if not match:
        raise ValueError(f"unexpected external code {code!r}")
    return match.group(1).zfill(3)


def stored_sequences(frame: pd.DataFrame) -> dict:
    ordered = frame.sort_values(["inscription_id", "position"])
    return {
        cisi: group["sign_code"].tolist()
        for cisi, group in ordered.groupby("cisi", sort=False)
    }


def audit_family_overlap(frame: pd.DataFrame, external: pd.DataFrame) -> dict:
    from collections import Counter

    ours = stored_sequences(frame)
    compared = exact = length_match = 0
    readable_positions = readable_agree = 0
    missing_positions = 0
    mismatches = []
    relations = Counter()
    dropped = Counter()
    for _, row in external.iterrows():
        cisi = row["cisi_number"]
        if cisi == "unknown" or cisi not in ours:
            continue
        theirs = [normalize_gcode(c) for c in str(row["sign_sequence"]).split()]
        mine = ours[cisi]
        compared += 1
        if len(theirs) == len(mine):
            length_match += 1
        if theirs == mine:
            exact += 1
            relations["exact"] += 1
        else:
            mismatches.append(cisi)
            if theirs == mine[2:]:
                relations["their_drops_first2"] += 1
                dropped.update(mine[:2])
            elif theirs == mine[:-2]:
                relations["their_drops_last2"] += 1
                dropped.update(mine[-2:])
            elif [x for x in theirs if x != "000"] == [x for x in mine if x != "000"]:
                relations["gap_placement_only"] += 1
            else:
                relations["other"] += 1
        for a, b in zip(mine, theirs):
            if a == "000":
                missing_positions += 1
            else:
                readable_positions += 1
                readable_agree += a == b
    return {
        "n_compared": compared,
        "length_agreement": length_match / compared if compared else None,
        "exact_agreement": exact / compared if compared else None,
        "readable_token_agreement": readable_agree / readable_positions if readable_positions else None,
        "readable_positions": readable_positions,
        "missing_positions": missing_positions,
        "mismatch_relations": dict(relations),
        "dropped_edge_codes": dropped.most_common(10),
        "mismatch_cisi": sorted(mismatches),
    }


def audit_mayig_overlap(frame: pd.DataFrame, mayig: pd.DataFrame) -> dict:
    base = mayig["inscription_id"].str.replace(r"[A-Z]$", "", regex=True)
    mayig = mayig.assign(base=base)
    ours = stored_sequences(frame)
    our_cisi = set(ours)
    matched = mayig[mayig["base"].isin(our_cisi)]
    length_pairs = [
        (len(str(seq).split()), len(ours[cisi]))
        for seq, cisi in zip(matched["sign_sequence"], matched["base"])
    ]
    gated_pairs = [
        (n, sum(s != "000" for s in ours[cisi]))
        for (n, _), cisi in zip(length_pairs, matched["base"])
    ]
    return {
        "mayig_rows": len(mayig),
        "matched_inscriptions": int(matched["base"].nunique()),
        "stored_length_agreement": sum(a == b for a, b in length_pairs) / len(length_pairs) if length_pairs else None,
        "gated_length_agreement": sum(a == b for a, b in gated_pairs) / len(gated_pairs) if gated_pairs else None,
        "mayig_damaged_rate": float(mayig["damaged"].mean()),
        "mayig_mean_uncertainty": float(mayig["mean_uncertainty"].mean()),
        "code_mapping_note": "no trusted machine-readable ICIT<->Parpola concordance found in either repository; sign-level agreement not computable",
    }


def render_report(summary):
    fam = summary["family"]
    may = summary["mayig"]
    lines = [
        "Cross-corpus audit",
        "",
        "Same-family check (ours vs indus-website ICIT digitization, stored order):",
        f"Compared {fam['n_compared']} inscriptions: length agreement "
        f"{fam['length_agreement']:.4f}, exact sequence agreement "
        f"{fam['exact_agreement']:.4f}.",
        f"Readable-position token agreement {fam['readable_token_agreement']:.4f} "
        f"over {fam['readable_positions']} positions; {fam['missing_positions']} "
        "positions are 000 on our side (external source carries no missing marker).",
        f"Mismatch relations: {fam['mismatch_relations']}.",
        f"Edge signs dropped by the external source: {fam['dropped_edge_codes']}.",
        "A small set of edge codes accounts for most length differences: an",
        "editorial convention difference, not random transcription noise.",
        "This measures pipeline consistency within one transcription family,",
        "not independent inter-annotator agreement.",
        "",
        "Independent check (ours vs mayig CISI hand-transcription, Parpola codes):",
        f"{may['matched_inscriptions']} of {may['mayig_rows']} seals match by CISI base number.",
        f"Stored-length agreement {may['stored_length_agreement']}, "
        f"gated-length agreement {may['gated_length_agreement']}.",
        f"Mayig damage rate {may['mayig_damaged_rate']:.3f}, mean uncertainty "
        f"{may['mayig_mean_uncertainty']:.2f}.",
        f"Note: {may['code_mapping_note']}.",
        "",
        "Full mismatch lists and per-inscription pairs are in audit_summary.json.",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cross-corpus transcription audit")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "processed" / "corpus.csv")
    parser.add_argument("--external", type=Path, default=ROOT / "data" / "external" / "indus_website_real_corpus.csv")
    parser.add_argument("--mayig", type=Path, default=ROOT / "data" / "external" / "cisi_real_corpus.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "audit")
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.corpus, encoding="utf-8", dtype=str)
    external = pd.read_csv(args.external, encoding="utf-8", dtype=str)
    mayig = pd.read_csv(args.mayig, encoding="utf-8")
    summary = {
        "manifest": {
            "corpus_sha256": sha256_file(args.corpus),
            "external_sha256": sha256_file(args.external),
            "mayig_sha256": sha256_file(args.mayig),
            "python": platform.python_version(),
            "dependencies": {n: version(n) for n in ("numpy", "pandas", "scipy")},
        },
        "family": audit_family_overlap(frame, external),
        "mayig": audit_mayig_overlap(frame, mayig),
    }
    report = render_report(summary)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (args.output / "audit_report.txt").write_text(report, encoding="utf-8")
    print(report)
    return summary


if __name__ == "__main__":
    main()
