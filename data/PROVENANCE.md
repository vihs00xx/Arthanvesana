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

## External metadata (sidecar)

- **Source:** https://huggingface.co/datasets/joyboseroy/indus_decipher
  (`indus_website_real_corpus.csv`, config `indus_website`), derived from
  yajnadevam/indus-website's ICIT digitization; acquired 2026-09-19 into
  `data/external/` (gitignored).
- **Sidecar:** `data/processed/inscription_metadata.csv`, built by
  `python scripts/build_metadata.py`: 2,375 rows keyed by `cisi`, all matched
  to corpus inscriptions (41.6% coverage); motif known for 1,514.
- **Limitations:** the external `damaged` flag is uniformly False (unusable);
  `line_count` is uniformly 1 and `object_type` uniformly unknown. The sidecar
  carries metadata only, no sign sequences.

## Cross-corpus audit sources

- **Same-family:** `indus_website_real_corpus.csv` from the same HF dataset
  (2,375 overlapping inscriptions). Compared in stored order in
  `scripts/run_audit.py`.
- **Independent:** `cisi_real_corpus.csv` (179 Mohenjo-daro unicorn seals,
  Parpola codes, per-grapheme damage and uncertainty) from mayig's CISI
  digitization via the same HF dataset. No trusted machine-readable
  ICIT↔Parpola concordance exists in either repository, so only catalog
  overlap and sequence-length agreement are computable, not sign-level
  agreement.

## Licensing / redistribution

The upstream README declares the corpus **GPL-3.0** and scripts **MIT** (verified
2026-09-20 against the pinned commit; the deposit states the corpus is GPL-3.0
because it redistributes substantial GPL-3.0 upstream content from
`yajnadevam/indus-website`'s `population-script.sql`).

**Resolution (this project):**

- The raw corpus stays in `data/raw/` (gitignored) and is **not redistributed**
  here. Users obtain it from the upstream deposit at the pinned commit and
  rebuild via `scripts/build_corpus.py`.
- **Derived artifacts that encode sign sequences** (`data/processed/corpus.csv`)
  are derivative of the GPL-3.0 corpus and therefore remain under **GPL-3.0**
  when redistributed. They are tracked in this repository for reproducibility
  only; anyone redistributing them must do so under GPL-3.0 with attribution to
  the upstream deposit and Parpola et al.'s CISI.
- **Aggregate statistics** (reported numbers, entropy/perplexity values, model
  metrics, figures) are facts/measurements and are not creative expression
  subject to the corpus license; they may be cited freely with attribution.
- This project's **code and analysis** remain **Apache 2.0**, which does not
  override upstream data licensing. The Apache-2.0 code and the GPL-3.0 derived
  data are separable: the code runs on any corpus the user supplies.

This interpretation is documented for transparency; it is not legal advice.
When in doubt, do not redistribute the processed CSV — regenerate it from the
upstream raw file with `scripts/build_corpus.py`.

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
