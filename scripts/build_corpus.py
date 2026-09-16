"""Build the tidy corpus: raw JSON -> data/processed/corpus.csv + summary.

Usage (from repo root):
    .venv\\Scripts\\python scripts\\build_corpus.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anvesa.data.parse import (  # noqa: E402
    inscription_sequences,
    load_raw,
    sha256_file,
    to_tidy,
)
from anvesa.data.validate import validate_raw, validate_tidy  # noqa: E402

RAW = ROOT / "data" / "raw" / "sanitized_corpus.json"
PROCESSED = ROOT / "data" / "processed"
CSV = PROCESSED / "corpus.csv"
SUMMARY = PROCESSED / "corpus_summary.json"


def main() -> None:
    records = load_raw(RAW, verify=True)
    raw_report = validate_raw(records)
    df = to_tidy(records)
    tidy_report = validate_tidy(df)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False, encoding="utf-8")

    seqs = inscription_sequences(df)  # reading order, gated, non-empty
    gated_lengths = [len(s) for s in seqs]
    summary = {
        "sha256": sha256_file(RAW),
        "raw": raw_report,
        "tidy": tidy_report,
        "analyzable_sequences": len(seqs),
        "gated_length_mean": sum(gated_lengths) / len(gated_lengths),
        "gated_length_max": max(gated_lengths),
        "sites": int(df["site"].nunique()),
        "artefact_types": int(df["artefact_type"].nunique()),
    }
    with open(SUMMARY, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2))
    print(f"wrote {CSV.name} ({len(df)} rows) to data/processed/")


if __name__ == "__main__":
    main()
