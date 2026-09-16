"""Corpus validation. EXPECTED encodes the deposit's published counts; any
failure means the upstream input changed or the parser is wrong.
"""

from __future__ import annotations

import pandas as pd

from .parse import MISSING, REQUIRED_KEYS, _CODE_RE

EXPECTED = {
    "inscriptions": 5704,
    "tokens": 19976,
    "distinct_codes_incl_missing": 714,
    "missing_tokens": 1911,
    "all_missing_inscriptions": 168,
    "analyzable_inscriptions": 5536,
    "gated_sign_tokens": 18065,
    "distinct_signs_gated": 713,
}


def _fail(check: str, detail: str) -> ValueError:
    return ValueError(f"validation failed [{check}]: {detail}")


def validate_raw(records: list[dict]) -> dict:
    if len(records) != EXPECTED["inscriptions"]:
        raise _fail(
            "row-count",
            f"got {len(records)} records, expected {EXPECTED['inscriptions']}",
        )
    ids = [r.get("id") for r in records]
    if len(set(ids)) != len(ids):
        raise _fail("unique-ids", "duplicate inscription ids present")
    for i, rec in enumerate(records):
        missing_keys = [k for k in REQUIRED_KEYS if k not in rec]
        if missing_keys:
            raise _fail("required-keys", f"row {i} missing {missing_keys}")
        symbols = rec.get("symbols")
        if not isinstance(symbols, list) or not symbols:
            raise _fail("symbols", f"row {i} ({rec.get('id')}) has empty symbols")
        bad = [s for s in symbols if not _CODE_RE.match(str(s))]
        if bad:
            raise _fail(
                "code-format", f"row {i} ({rec.get('id')}) has malformed codes {bad}"
            )
    return {"records": len(records), "unique_ids": len(set(ids))}


def validate_tidy(df: pd.DataFrame) -> dict:
    report: dict = {}
    report["token_rows"] = len(df)
    if len(df) != EXPECTED["tokens"]:
        raise _fail(
            "token-count", f"got {len(df)} rows, expected {EXPECTED['tokens']}"
        )
    n_codes = df["sign_code"].nunique()
    report["distinct_codes"] = n_codes
    if n_codes != EXPECTED["distinct_codes_incl_missing"]:
        raise _fail("distinct-codes", f"got {n_codes}")
    n_missing = int(df["is_missing"].sum())
    report["missing_tokens"] = n_missing
    if n_missing != EXPECTED["missing_tokens"]:
        raise _fail("missing-tokens", f"got {n_missing}")
    per_ins = df.groupby("inscription_id")["is_missing"]
    n_all_missing = int((per_ins.mean() == 1.0).sum())
    report["all_missing_inscriptions"] = n_all_missing
    if n_all_missing != EXPECTED["all_missing_inscriptions"]:
        raise _fail("all-missing", f"got {n_all_missing}")
    report["analyzable_inscriptions"] = int(df["inscription_id"].nunique()) - n_all_missing
    if report["analyzable_inscriptions"] != EXPECTED["analyzable_inscriptions"]:
        raise _fail("analyzable", f"got {report['analyzable_inscriptions']}")
    gated = df[~df["is_missing"]]
    report["gated_sign_tokens"] = len(gated)
    if len(gated) != EXPECTED["gated_sign_tokens"]:
        raise _fail("gated-tokens", f"got {len(gated)}")
    n_signs = gated["sign_code"].nunique()
    report["distinct_signs_gated"] = n_signs
    if n_signs != EXPECTED["distinct_signs_gated"]:
        raise _fail("distinct-signs", f"got {n_signs}")
    if (df["position"] < 1).any():
        raise _fail("positions", "non-positive position values")
    return report
