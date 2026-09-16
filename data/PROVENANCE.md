# Data provenance

## Source corpus

- **Upstream repository:** https://github.com/ShaktiOSindia/indus-sign-regimes-deposit
- **File:** `sanitized_corpus.json`
- **Upstream commit:** `e48b3ec1e90f368079f5126790613cded6f6f56c` (2026-08-12)
- **SHA-256:** `345241b13fedada87b4783c24cd241123491bbd7edaf5bf636f9cdb36c01da68`
- **Acquired:** 2026-09-16 via shallow clone of the upstream repo (the raw
  `raw.githubusercontent.com` URL for the pinned commit returned 404, so the
  file was copied from the clone and verified byte-for-byte against the SHA-256
  published in the deposit README).

Reproduce with: `.venv\Scripts\python src\arthanvesana\data\download.py`

## Licensing / redistribution

The upstream deposit contains **no LICENSE file**, so the corpus file is kept
in `data/raw/` (gitignored) and is **not redistributed** in this repository.
Only derived artifacts (`data/processed/corpus.csv`, summary statistics) are
tracked. Cite the upstream deposit and Parpola et al.'s CISI when using it.

## Published properties (deposit README, verified by `scripts/build_corpus.py`)

| Property | Value |
|---|---|
| Catalogued inscriptions | 5,704 |
| Analyzable (excl. 168 all-`000` rows) | 5,536 |
| Sign tokens, `000` gated | 18,065 |
| Distinct signs, `000` gated | 713 |
| Sites / artefact types | 77 / 32 |

## Conventions adopted from upstream (`chr_lib`)

- Symbols stored physical left-to-right; reading order is as-stored for
  direction `L/R`, reversed for `R/L`.
- ICIT `000` is a missing-data placeholder, removed ("gated") before analysis.
- **Deviation:** upstream reverses stored order for *every* non-`L/R`
  direction (including `-`, `NR`, `BUS`, `SYM`, `T/B`). We reverse only for
  unambiguous `R/L` and flag the rest with `reading_order_known=False`.
