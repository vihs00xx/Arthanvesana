# Arthānveṣaṇa

**AI-assisted structural analysis of the Sindhu-Sarasvatī script using statistical baselines, sign embeddings, and self-supervised learning.**


Inspired by the Sanskrit idea of searching and inquiry, **Arthānveṣaṇa** ("search for meaning") reflects this project's purpose: to investigate ancient signs through evidence, uncover structural patterns, and generate testable hypotheses.

## Status

Corpus ingestion and validation, statistical baselines with shuffled controls, grouped evaluation, PPMI/SVD and skip-gram sign embeddings, clustering with permutation-null checks, PCA/UMAP visualization, and a masked-sign transformer are implemented. A metadata sidecar, motif-stratified evaluation, and a cross-corpus transcription audit are also included.

## Source corpus

- Upstream: [ShaktiOSindia/indus-sign-regimes-deposit](https://github.com/ShaktiOSindia/indus-sign-regimes-deposit)
- File: `sanitized_corpus.json` (5,704 catalogued inscriptions; ~18k sign tokens; 713 distinct signs; `000` marks an illegible sign, not a sign)
- Upstream commit: `e48b3ec1e90f368079f5126790613cded6f6f56c` (2026-08-12)
- SHA-256: `345241b13fedada87b4783c24cd241123491bbd7edaf5bf636f9cdb36c01da68`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install -e .
```

Run scripts from the repository root, e.g. `python scripts/build_corpus.py`.

## Testing

Install development tools with `python -m pip install -r requirements-dev.txt` (or `python -m pip install -e ".[dev]"`).

```bash
python -m pytest
python -m ruff check .
python -m mypy
```

Type checking currently covers five statistical core modules. Run `python scripts/run_stats.py`, `python scripts/run_replication.py`, and `python scripts/run_upgrade.py` to regenerate local reports. Defaults use known-direction, gap-split spans and artifact/duplicate-grouped holdouts; these results are not directly comparable to the earlier record-level splits.

## Restoration robustness

Run `python scripts/run_robustness.py --repeats 10` to compare the context model with training-only frequency and observed-position baselines. The same artifact/duplicate-grouped partitions are evaluated as readable spans, complete inscriptions, unique sequences, and complete unique sequences. Each model predicts the same masked positions; unseen target signs count as failures.

The runner also holds out each site with at least 100 eligible inscriptions, removing linked artifacts and duplicate groups from training. Local `outputs/robustness/` files contain a text report and JSON with all split identities, paired differences, overlap checks, source/data hashes, and dependency versions. Reported standard deviations describe split variability, not confidence intervals over independent experiments. Use `--help` for input, output, seed, split-fraction, and site-threshold options.

## Sign embeddings

Run `python scripts/run_embeddings.py --repeats 10` to train PPMI/SVD and skip-gram sign vectors and compare embedding-based restoration against the bigram on identical grouped splits and masked positions. The runner also clusters both vector sets (k-means sweep plus hierarchical), tests cluster alignment with positional roles against a permutation null, and writes PCA/UMAP figures colored by role. Local `outputs/embeddings/` files contain a text report and JSON with the config sweeps, per-run metrics, paired differences, stability checks, cluster assignments, and full-data neighborhoods. Reported standard deviations describe split variability, not confidence intervals.

## Masked-sign transformer

Run `python scripts/run_transformer.py --repeats 10` to train a tiny transformer encoder (1–2 layers, dim 32–64) with a mask-one-sign objective and compare it against the bigram on identical grouped splits and masked positions. Training uses early stopping on a grouped validation split; parameter counts and epochs are reported alongside accuracy. Requires torch (CPU wheel, see `requirements.txt`). Local `outputs/transformer/` files contain a text report and JSON with the config selection, per-run metrics, paired differences, and training diagnostics. Reported standard deviations describe split variability, not confidence intervals.

## Data enrichment and audit

Run `python scripts/build_metadata.py` to join external motif and direction fields (HuggingFace `joyboseroy/indus_decipher`, gitignored under `data/external/`) into `data/processed/inscription_metadata.csv`, keyed by CISI number. Run `python scripts/run_stratified.py` for motif-stratified restoration and a direction cross-check, and `python scripts/run_audit.py` for the cross-corpus transcription audit (same-family agreement plus an independent length-level check against the mayig CISI transcription). Local `outputs/stratified/` and `outputs/audit/` hold the reports. See `data/PROVENANCE.md` for sources and limitations.

## License

Apache 2.0
