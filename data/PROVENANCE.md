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

The raw file is already present locally; no download is needed for normal runs.
On a fresh machine, obtain `sanitized_corpus.json` from the upstream repository
at the pinned commit above and place it at `data/raw/sanitized_corpus.json`.
Run `python scripts/build_corpus.py` from the project root to verify its SHA-256,
validate the records, and rebuild the processed CSV and summary.

## Licensing / redistribution

The upstream README currently declares the corpus **GPL-3.0** and scripts **MIT**.
The raw corpus is kept in `data/raw/` (gitignored) and is not redistributed here.
Derived artifacts (`data/processed/corpus.csv`, summary statistics) are tracked;
whether redistribution of the processed CSV meets the upstream licensing
requirements remains a question for the maintainer. This project's Apache 2.0
license does not override upstream data licensing. Cite the upstream deposit
and Parpola et al.'s CISI when using it.

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
