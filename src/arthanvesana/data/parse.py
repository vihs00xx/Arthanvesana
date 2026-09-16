from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

EXPECTED_SHA256 = "345241b13fedada87b4783c24cd241123491bbd7edaf5bf636f9cdb36c01da68"
MISSING = "000"
_CODE_RE = re.compile(r"^\d{3}$")

REQUIRED_KEYS = (
    "id",
    "cisi",
    "site",
    "artefact_type",
    "direction",
    "complete",
    "symbols",
    "source",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(path: str | Path, verify: bool = True) -> list[dict]:
    path = Path(path)
    if verify:
        got = sha256_file(path)
        if got.lower() != EXPECTED_SHA256.lower():
            raise ValueError(
                f"SHA-256 mismatch for {path}: got {got}, "
                f"expected {EXPECTED_SHA256}."
            )
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)
    if not isinstance(records, list):
        raise ValueError("Expected top-level JSON list of inscription records.")
    return records


def normalize_direction(raw: str | None) -> str:
    s = (raw or "").strip()
    if s == "L/R":
        return "L/R"
    if s in ("R/L", "R/l"):
        return "R/L"
    return "OTHER"


def reading_order_known(direction: str) -> bool:
    return direction in ("L/R", "R/L")


def sequence_in_reading_order(symbols: list[str], direction: str) -> list[str]:
    if direction == "R/L":
        return list(reversed(symbols))
    return list(symbols)


def gate(sequence: list[str]) -> list[str]:
    return [s for s in sequence if s != MISSING]


def to_tidy(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        direction = normalize_direction(rec.get("direction"))
        symbols = rec.get("symbols", [])
        for pos, code in enumerate(symbols, start=1):
            rows.append(
                {
                    "inscription_id": rec.get("id"),
                    "cisi": rec.get("cisi"),
                    "site": rec.get("site"),
                    "artefact_type": rec.get("artefact_type"),
                    "direction_raw": rec.get("direction"),
                    "direction": direction,
                    "reading_order_known": reading_order_known(direction),
                    "complete": rec.get("complete"),
                    "stored_length": len(symbols),
                    "position": pos,
                    "sign_code": code,
                    "is_missing": code == MISSING,
                }
            )
    df = pd.DataFrame(
        rows,
        columns=[
            "inscription_id",
            "cisi",
            "site",
            "artefact_type",
            "direction_raw",
            "direction",
            "reading_order_known",
            "complete",
            "stored_length",
            "position",
            "sign_code",
            "is_missing",
        ],
    )
    return df


def inscription_sequences(
    df: pd.DataFrame,
    *,
    reading_order: bool = True,
    drop_missing: bool = True,
    drop_empty: bool = True,
    known_direction_only: bool = False,
) -> list[list[str]]:
    frame = df
    if known_direction_only:
        frame = frame[frame["reading_order_known"]]
    seqs: list[list[str]] = []
    for _, group in frame.groupby("inscription_id", sort=False):
        group = group.sort_values("position")
        symbols = group["sign_code"].tolist()
        direction = group["direction"].iloc[0]
        if reading_order:
            symbols = sequence_in_reading_order(symbols, direction)
        if drop_missing:
            symbols = gate(symbols)
        if symbols or not drop_empty:
            seqs.append(symbols)
    return seqs
