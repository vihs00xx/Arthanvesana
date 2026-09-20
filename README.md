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

Type checking currently covers five statistical core modules under `strict` (`src/arthanvesana/stats/entropy.py`, `src/arthanvesana/replicate/llr.py`, `src/arthanvesana/replicate/entropy2.py`, `src/arthanvesana/replicate/metrics.py`, `src/arthanvesana/replicate/segment.py`). The newer `simulate` and `compact` modules import numpy and are only partially annotated; they are not yet under `strict`, and numpy's bundled stubs require Python ≥3.12 to parse while the checked configuration targets 3.11. Full annotation of those modules is outstanding work. Run `python scripts/run_stats.py`, `python scripts/run_replication.py`, and `python scripts/run_upgrade.py` to regenerate local reports. Defaults use known-direction, gap-split spans and artifact/duplicate-grouped holdouts; these results are not directly comparable to the earlier record-level splits.

## Restoration robustness

Run `python scripts/run_robustness.py --repeats 10` to compare the context model with training-only frequency and observed-position baselines. The same artifact/duplicate-grouped partitions are evaluated as readable spans, complete inscriptions, unique sequences, and complete unique sequences. Each model predicts the same masked positions; unseen target signs count as failures.

The runner also holds out each site with at least 100 eligible inscriptions, removing linked artifacts and duplicate groups from training. Local `outputs/robustness/` files contain a text report and JSON with all split identities, paired differences, overlap checks, source/data hashes, and dependency versions. Reported standard deviations describe split variability, not confidence intervals over independent experiments. Use `--help` for input, output, seed, split-fraction, and site-threshold options.

## Sign embeddings

Run `python scripts/run_embeddings.py --repeats 10` to train PPMI/SVD and skip-gram sign vectors and compare embedding-based restoration against the bigram on identical grouped splits and masked positions. The runner also clusters both vector sets (k-means sweep plus hierarchical), tests cluster alignment with positional roles against a permutation null, and writes PCA/UMAP figures colored by role. Local `outputs/embeddings/` files contain a text report and JSON with the config sweeps, per-run metrics, paired differences, stability checks, cluster assignments, and full-data neighborhoods. Reported standard deviations describe split variability, not confidence intervals.

## Masked-sign transformer

Run `python scripts/run_transformer.py --repeats 10` to train a tiny transformer encoder (1–2 layers, dim 32–64) with a mask-one-sign objective and compare it against the bigram on identical grouped splits and masked positions. Training uses early stopping on a grouped validation split; parameter counts and epochs are reported alongside accuracy. Requires torch (CPU wheel, see `requirements.txt`). Local `outputs/transformer/` files contain a text report and JSON with the config selection, per-run metrics, paired differences, and training diagnostics. Reported standard deviations describe split variability, not confidence intervals.

## Data enrichment and audit

Run `python scripts/build_metadata.py` to join external motif and direction fields (HuggingFace `joyboseroy/indus_decipher`, gitignored under `data/external/`) into `data/processed/inscription_metadata.csv`, keyed by CISI number. Run `python scripts/run_stratified.py` for motif-stratified restoration and a direction cross-check, and `python scripts/run_audit.py` for the cross-corpus transcription audit (same-family agreement plus an independent length-level check against the mayig CISI transcription). Local `outputs/stratified/` and `outputs/audit/` hold the reports. See `data/PROVENANCE.md` for sources and limitations.

## Group audit, sequence-order sensitivity, and grouped n-gram inference

Run `python scripts/run_group_audit.py` to audit the connected artifact/inscription/duplicate components used by every grouped split. It reports component-size percentiles, the largest components and which identity relation created them, and warns when any component exceeds one fold's target size. Exact-sequence connectivity merges many records into large components, which is why grouped test sets vary in size; grouping is deliberately not weakened to balance folds.

Run `python scripts/run_sensitivity.py --repeats 10` for the 2×2×2 preprocessing matrix covering **sequence-order processing** (`reading_order_normalized` vs `physical_as_stored`), **direction inclusion**, and **completeness filtering**. This is not a comparison of independent transcription traditions; both sequence-order levels order the *same* transcription.

Run `python scripts/run_ngram_inference.py` for grouped cross-fitted bigram-vs-trigram log-loss inference: connected components are indivisible groups assigned to five folds, every record gets exactly one out-of-fold prediction, token differences are aggregated within each group before inference, and the primary estimand is the macro group-level effect with a group-level sign-flip randomization p-value `(exceedances+1)/(permutations+1)` (never 0) and a cluster-bootstrap interval. The earlier token-level pooled p-value is removed.

## Direction diagnostics

Run `python scripts/run_direction_diagnostics.py --repeats 10` to investigate the ~2.7–3.1 percentage-point advantage of physical as-stored order over reading-order-normalized order. It reports direction-specific evaluation, a global reversal sanity check (interior masks are exactly mirror-symmetric; boundary positions are not, because the span start uses the `<S>` distribution while the end takes a uniform no-evidence term), cross-direction transfer with controlled training sizes, and a stratified breakdown of the gap. Diagnostic only: no pipeline default changes.

## Same-family transcription sensitivity

Run `python scripts/run_transcription_sensitivity.py --repeats 10` to test whether context beats frequency and position under *both* same-family ICIT transcriptions. Records are matched by CISI (one-to-one only, all exclusions reported) and both transcriptions receive identical artifact-level test sets built from stable catalog ids. Agreement between these ICIT-derived sources is **not** independent inter-annotator agreement.

## Power calibration and compact models

Run `python scripts/run_power_analysis.py` to calibrate what the corrected inference can detect at this corpus size. Synthetic corpora matched to empirical properties (inscription count, length distribution, vocabulary, unigram frequencies) are passed through the same grouped cross-fitted pipeline at 0.5×/1×/2× size, with zero-effect scenarios for the false-positive rate and a controlled trigram-effect parameter for power. Use `--replicates 100` for the full grid. The generators are calibration instruments, not models of the Indus production process or of natural language.

Run `python scripts/run_compact_models.py` to compare a smoothed relative-position model and a discrete HMM (2–5 states, state count selected on an inner split of each outer training partition) against the bigram under grouped nested cross-validation, with held-out bits/token and perplexity as the primary metrics. The HMM states are an economical latent-state representation of positional sequence structure; they are not words, phrases, grammatical roles, or semantic classes.

## License

Apache 2.0
