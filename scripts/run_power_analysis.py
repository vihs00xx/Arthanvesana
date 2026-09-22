"""Synthetic calibration for the grouped bigram-vs-trigram inference.

This runner answers three *different* questions and never treats them as
interchangeable:

1. **Predictive difference** — do the fitted bigram and trigram differ in
   held-out performance, in either direction? Assessed with the two-sided
   group-level sign-flip test (``reject_difference``).
2. **Predictive improvement** — does the trigram improve held-out performance?
   Assessed with the one-sided positive sign-flip test (``reject_positive``).
   A significantly *negative* effect can never satisfy this indicator.
3. **Structural departure** — is the observed gain unusually large relative to
   a specified first-order generative null passed through the complete
   estimation pipeline? Assessed by re-fitting both models and re-running the
   whole cross-fitting pipeline on corpora drawn from the fitted first-order
   null, then comparing the observed positive effect with the resulting null
   distribution (``structural`` block).

Terminology rule enforced here: a rate is labelled a **false-positive rate**
only for a scenario that actually satisfies the null hypothesis of that
particular decision rule. A first-order generator can produce a genuine
*predictive disadvantage* for a more complex fitted model, so a two-sided
rejection under such a generator is not a false discovery of higher-order
structure. Every scenario therefore carries an explicit ``null_for`` block, and
the report prints the rule a rate is (and is not) valid for.

Synthetic generators are calibration instruments, not models of the Indus
production process or of natural language.

Runs are resumable: every replicate is written to its own JSON file and the
summary is rebuilt from whatever replicates exist, so an interrupted long run
can be continued with the same command.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
import zlib
from importlib.metadata import version
from pathlib import Path
from statistics import fmean, stdev

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arthanvesana.data.parse import analysis_records, sha256_file
from arthanvesana.simulate import (
    crossfit_effect,
    empirical_profile,
    gen_hmm_slots,
    gen_markov,
    gen_position_only,
    gen_shuffled_real,
    gen_trigram_mixture,
    gen_unigram,
)

#: Grid scenarios. ``scenario_info`` gives the null status of each one.
SCENARIOS = ("unigram", "position_only", "markov", "hmm_slots", "shuffled_real")
#: Higher-order strengths; spaced finer around the observed real effect (+0.028).
TRI_LAMBDAS = (0.0, 0.25, 0.30, 0.35, 0.40, 0.5, 1.0)
#: The fitted first-order null used for the structural test.
NULL_SCENARIO = "markov"

SCENARIO_INFO = {
    "unigram": {
        "generator": "i.i.d. draws from the empirical unigram",
        "kind": "control",
        "null_for_improvement": True,
        "null_for_difference": False,
        "note": ("No sequential dependence at all. Satisfies the null of the "
                 "improvement rule (the trigram cannot genuinely improve), but "
                 "the two fitted models are not exchangeable in expectation, so "
                 "it is not a null for the difference rule."),
    },
    "position_only": {
        "generator": "empirical starters / enders at the edges, unigram inside",
        "kind": "named control",
        "null_for_improvement": True,
        "null_for_difference": False,
        "note": ("Retains length and position dependence by construction. A named "
                 "control, NOT a mathematically pure first-order null."),
    },
    NULL_SCENARIO: {
        "generator": "first-order Markov chain sampling the empirical bigrams",
        "kind": "fitted first-order null",
        "null_for_improvement": True,
        "null_for_difference": False,
        "note": ("The fitted first-order null: preserves length distribution, "
                 "vocabulary and first-order transitions; contains no genuine "
                 "higher-order structure. The fitted trigram is still "
                 "over-parameterized relative to it, so it genuinely loses — "
                 "which is why this scenario is a null for the improvement rule "
                 "but not for the difference rule."),
    },
    "hmm_slots": {
        "generator": "deterministic positional-slot sampler (not a stochastic HMM)",
        "kind": "named alternative",
        "null_for_improvement": False,
        "null_for_difference": False,
        "note": ("Deterministic position-dependent emissions. Position within a span "
                 "is correlated with order, so a fitted trigram genuinely extracts "
                 "this structure rather than merely fitting noise. Measured at the "
                 "real corpus scale its positive-rejection rate is 1.000 with a "
                 "POSITIVE mean effect, so this scenario does NOT satisfy the "
                 "improvement null: its 'positive' column is POWER, not a "
                 "false-positive rate. It is retained as a named alternative."),
    },
    "shuffled_real": {
        "generator": "within-inscription shuffles of real sequences",
        "kind": "named control",
        "null_for_improvement": True,
        "null_for_difference": False,
        "note": ("Destroys order but retains the real sign composition and length "
                 "distribution, so it can retain dependencies from length and "
                 "composition. Separately named control, not a pure first-order "
                 "null."),
    },
    "trigram_mixture": {
        "generator": "(1-lam) * bigram + lam * trigram transition mixture",
        "kind": "calibrated alternative",
        "null_for_improvement": None,  # depends on lam
        "null_for_difference": False,
        "note": ("lam=0 is the zero-higher-order cell and is a null for the "
                 "improvement rule; lam>0 is an alternative used to estimate "
                 "power. Spacing is finer near the observed real effect."),
    },
}

def cell_seed(base_seed, cell, replicate):
    """Deterministic per-replicate seed derived from the cell label.

    Deriving the seed from the cell (rather than from the loop position) keeps a
    resumed run reproducible and gives every scenario/size/strength cell its own
    independent random stream.
    """
    return ((base_seed * 1_000_000_000
             + (zlib.crc32(cell.encode("utf-8")) % 1_000_000) * 1000
             + replicate) % (2 ** 63))


def _fmt(value, spec="+.5f", missing="-"):
    """Format a possibly-missing statistic without crashing the report."""
    if value is None:
        return missing
    return format(value, spec)


def _mcse(values):
    if len(values) < 2:
        return 0.0
    return stdev(values) / math.sqrt(len(values))


def wilson(k, n, z=1.959963984540054):
    """Wilson score interval for a binomial rate; None when n == 0."""
    if n <= 0:
        return None
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def scenario_sequences(name, profile, real_records, n, seed, lam=0.0):
    """Generate one synthetic corpus for a scenario."""
    if name == "unigram":
        return gen_unigram(profile, n, seed)
    if name == "position_only":
        return gen_position_only(profile, n, seed)
    if name == NULL_SCENARIO:
        return gen_markov(profile, n, seed)
    if name == "hmm_slots":
        seqs, _states = gen_hmm_slots(profile, n, seed)
        return seqs
    if name == "shuffled_real":
        return gen_shuffled_real(real_records, n, seed)
    if name == "trigram_mixture":
        return gen_trigram_mixture(profile, n, seed, lam)
    raise ValueError(f"unknown scenario {name}")


def realized_properties(seqs, profile):
    """Realized vs target properties of one synthetic corpus.

    Reports only what the generators actually control. Artifact relationships,
    duplicate/component structure, missingness and completeness are NOT
    generated, so they are reported as absent rather than as matched.
    """
    tokens = [s for seq in seqs for s in seq]
    counts = {}
    for sign in tokens:
        counts[sign] = counts.get(sign, 0) + 1
    vocab = set(counts)
    target_vocab = set(profile["unigram"])
    lengths = sorted(len(s) for s in seqs)
    target_lengths = sorted(profile["lengths"])
    return {
        "n_inscriptions": len(seqs),
        "target_n_inscriptions": profile["n_inscriptions"],
        "n_tokens": len(tokens),
        "target_n_tokens": sum(profile["lengths"]),
        "vocab_size": len(vocab),
        "target_vocab_size": len(target_vocab),
        "vocab_coverage_of_target": (
            len(vocab & target_vocab) / len(target_vocab) if target_vocab else None
        ),
        "mean_length": fmean(lengths) if lengths else None,
        "target_mean_length": fmean(target_lengths) if target_lengths else None,
        "max_length": lengths[-1] if lengths else None,
        "target_max_length": target_lengths[-1] if target_lengths else None,
        "singleton_sign_fraction": (
            sum(1 for c in counts.values() if c == 1) / len(counts) if counts else None
        ),
        "artifact_relationships_generated": False,
        "duplicate_components_generated": False,
        "missingness_generated": False,
        "note": ("artifact, duplicate, missingness and completeness structure are "
                 "NOT generated; only count, length, vocabulary and frequency "
                 "properties are sampled from the empirical corpus"),
    }


def _cell_key(scenario, lam, size):
    return f"{scenario}|lam={lam}|size={size}"


def _replicate_path(out_dir: Path, cell: str, rep: int) -> Path:
    safe = cell.replace("|", "__").replace("=", "-").replace(".", "p")
    return out_dir / "replicates" / safe / f"{rep:04d}.json"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def run_cell(out_dir, cell, scenario, lam, size, n, replicates, seed_base,
             permutations, bootstrap, resume, verbose=True):
    """Run (or reuse) every replicate of one cell; returns the records."""
    records = []
    for rep in range(replicates):
        path = _replicate_path(out_dir, cell, rep)
        if resume:
            cached = _read_json(path)
            if cached is not None and cached.get("status") == "ok":
                records.append(cached)
                continue
        seed = cell_seed(seed_base, cell, rep)
        started = time.perf_counter()
        entry = {
            "scenario": scenario, "lambda": lam, "size": size,
            "replicate": rep, "n_inscriptions": n, "seed": seed,
        }
        try:
            seqs = scenario_sequences(scenario, PROFILE, REAL_RECORDS, n, seed, lam)
            effect = crossfit_effect(seqs, 5, seed, permutations, bootstrap)
            entry.update(effect)
            entry["realized"] = realized_properties(seqs, PROFILE)
            entry["status"] = "ok" if effect.get("macro_effect") is not None \
                else "skipped"
            if entry["status"] == "skipped":
                entry["reason"] = "no scorable group"
        except (ValueError, FloatingPointError, OverflowError) as exc:
            entry["status"] = "failed"
            entry["reason"] = f"{type(exc).__name__}: {exc}"
            for key in ("macro_effect", "ci95", "p_two_sided", "p_positive",
                        "p_negative"):
                entry.setdefault(key, None)
            entry.setdefault("reject_difference", False)
            entry.setdefault("reject_positive", False)
            entry.setdefault("reject_negative", False)
        entry["seconds"] = time.perf_counter() - started
        _write_json(path, entry)
        records.append(entry)
        if verbose:
            print(f"  {cell} rep {rep + 1}/{replicates} "
                  f"[{entry['status']}] {entry['seconds']:.1f}s", flush=True)
    return records


def summarize(records):
    """Summarise one cell across the three decision rules.

    Every rate is reported with the denominator it was computed over, and the
    rule it belongs to. Rates are only *named* false-positive rates where the
    scenario satisfies that rule's null; the labelling is added by the caller.
    """
    n_runs = len(records)
    ok = [r for r in records if r.get("macro_effect") is not None]
    failed = [r for r in records if r.get("status") == "failed"]
    skipped = [r for r in records if r.get("status") == "skipped"]
    effects = [r["macro_effect"] for r in ok]
    n_ok = len(ok)
    k_diff = sum(1 for r in ok if r.get("reject_difference"))
    k_pos = sum(1 for r in ok if r.get("reject_positive"))
    k_neg = sum(1 for r in ok if r.get("reject_negative"))
    return {
        "n_runs": n_runs,
        "n_evaluable": n_ok,
        "n_failed": len(failed),
        "n_skipped": len(skipped),
        "failure_reasons": sorted({r.get("reason", "") for r in failed}),
        "mean_effect": fmean(effects) if effects else None,
        "mcse_effect": _mcse(effects) if n_ok > 1 else None,
        "sd_effect": stdev(effects) if n_ok > 1 else None,
        "min_effect": min(effects) if effects else None,
        "max_effect": max(effects) if effects else None,
        "effect_quantiles": (
            {
                "p05": float(np.percentile(effects, 5)),
                "p50": float(np.percentile(effects, 50)),
                "p95": float(np.percentile(effects, 95)),
            } if effects else None
        ),
        "difference": {
            "rule": "two-sided sign-flip on the macro group mean",
            "k": k_diff, "n": n_ok,
            "rate": k_diff / n_ok if n_ok else None,
            "ci95": wilson(k_diff, n_ok),
        },
        "positive_improvement": {
            "rule": "one-sided positive sign-flip on the macro group mean",
            "k": k_pos, "n": n_ok,
            "rate": k_pos / n_ok if n_ok else None,
            "ci95": wilson(k_pos, n_ok),
        },
        "negative": {
            "rule": "one-sided negative sign-flip on the macro group mean",
            "k": k_neg, "n": n_ok,
            "rate": k_neg / n_ok if n_ok else None,
            "ci95": wilson(k_neg, n_ok),
        },
    }


def structural_test(observed_effect, calibration_effects, evaluation_effects,
                    alpha=0.05):
    """Threshold-and-evaluate structural test with disjoint simulation sets.

    The threshold is the ``1 - alpha`` quantile of the **calibration** null
    effects; the **evaluation** null effects are an independent simulation set
    used only to estimate the rule's false-positive rate. Choosing a threshold
    and estimating its size on the same simulations would understate the error.

    The p-value is a Monte Carlo exceedance probability with the plus-one
    correction, so it is never zero. It is conditional on the fitted null model
   : it is evidence that the observed gain is unusual relative to *that* null,
    not evidence for a linguistic mechanism.
    """
    null_all = list(calibration_effects) + list(evaluation_effects)
    out = {
        "null_scenario": NULL_SCENARIO,
        "n_calibration": len(calibration_effects),
        "n_evaluation": len(evaluation_effects),
        "alpha": alpha,
        "observed_effect": observed_effect,
        "threshold": None,
        "false_positive_rate": None,
        "false_positive_ci95": None,
        "p_value": None,
        "decision": None,
        "limitations": [
            "the null is fitted to finite data and only preserves count, length, "
            "vocabulary and first-order transition structure",
            "generated corpora have no artifact or duplicate structure, which the "
            "real corpus does, so the null is not a perfect reproduction",
            "the p-value is conditional on this fitted null and is not proof of a "
            "linguistic mechanism",
        ],
    }
    if not calibration_effects:
        return out
    threshold = float(np.quantile(calibration_effects, 1 - alpha))
    out["threshold"] = threshold
    if evaluation_effects:
        k = sum(1 for e in evaluation_effects if e > threshold)
        n = len(evaluation_effects)
        out["false_positive_rate"] = k / n
        out["false_positive_ci95"] = wilson(k, n)
    if null_all and observed_effect is not None:
        exceed = sum(1 for e in null_all if e >= observed_effect)
        p = (exceed + 1) / (len(null_all) + 1)
        out["p_value"] = p
        out["decision"] = {
            "structural_departure": bool(p < alpha),
            "rule": "reject 'no structural departure' when p < alpha",
        }
    return out


PROFILE = None
REAL_RECORDS = None


def main(argv=None):
    global PROFILE, REAL_RECORDS
    parser = argparse.ArgumentParser(
        description="Synthetic calibration for bigram-vs-trigram inference")
    parser.add_argument("--corpus", type=Path,
                        default=ROOT / "data" / "processed" / "corpus.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "power_analysis")
    parser.add_argument("--replicates", type=int, default=24,
                        help="replicates per cell; the brief's full grid uses 100")
    parser.add_argument("--sizes", type=float, nargs="*", default=[0.5, 1.0, 2.0])
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    parser.add_argument("--lambdas", type=float, nargs="*", default=list(TRI_LAMBDAS))
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--null-replicates", type=int, default=0,
                        help="replicates per null half for the structural test; "
                             "0 disables the structural test")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore cached replicates and rerun every cell")
    parser.add_argument("--cache-only", action="store_true",
                        help=("write replicate caches but not the summary/report; "
                              "used to fan the grid across processes, followed by a "
                              "normal run that rebuilds the summary from cache"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    verbose = not args.quiet
    frame = pd.read_csv(args.corpus, encoding="utf-8", dtype={"sign_code": str})
    REAL_RECORDS = analysis_records(
        frame, gap_policy="split", known_direction_only=True)
    PROFILE = empirical_profile(REAL_RECORDS)
    n_emp = PROFILE["n_inscriptions"]
    total_tokens = sum(PROFILE["lengths"])
    resume = not args.no_resume

    started_all = time.perf_counter()
    cells: dict[str, list] = {}
    cell_meta: dict[str, dict] = {}
    for scenario in args.scenarios:
        lambdas = args.lambdas if scenario == "trigram_mixture" else [0.0]
        for lam in lambdas:
            for size in args.sizes:
                n = max(10, int(n_emp * size))
                cell = _cell_key(scenario, lam, size)
                cell_meta[cell] = {
                    "scenario": scenario, "lambda": lam, "size": size,
                    "n_inscriptions": n,
                }
                if verbose:
                    print(f"cell {cell} (n={n})", flush=True)
                cells[cell] = run_cell(
                    args.output, cell, scenario, lam, size, n, args.replicates,
                    args.seed, args.permutations, args.bootstrap, resume, verbose)

    # Observed effect on the real corpus, cached because it is the same for
    # every cell and is the statistic the structural test asks about.
    observed_path = args.output / "observed_effect.json"
    observed = _read_json(observed_path) if resume else None
    if observed is None or observed.get("status") != "ok":
        if verbose:
            print("computing observed effect on the real corpus ...", flush=True)
        seqs = [r["sequence"] for r in REAL_RECORDS if r["sequence"]]
        obs = crossfit_effect(seqs, 5, args.seed, args.permutations,
                              args.bootstrap)
        observed = {"status": "ok" if obs.get("macro_effect") is not None
                    else "skipped", **obs}
        _write_json(observed_path, observed)
    observed_effect = observed.get("macro_effect")

    # --- structural test: disjoint calibration and evaluation null sets
    structural = {"enabled": bool(args.null_replicates), "null_scenario": NULL_SCENARIO}
    if args.null_replicates:
        n_null = max(10, int(n_emp * 1.0))
        null_blocks = {}
        for half, seed_base in (
            ("calibration", 2_000_000_000),
            ("evaluation", 3_000_000_000),
        ):
            key = f"structural_null_{half}|lam=0.0|size=1.0"
            null_blocks[half] = run_cell(
                args.output, key, NULL_SCENARIO, 0.0, 1.0, n_null,
                args.null_replicates, args.seed + seed_base, args.permutations,
                args.bootstrap, resume, verbose)
        structural = structural_test(
            observed_effect,
            [r["macro_effect"] for r in null_blocks["calibration"]
             if r.get("macro_effect") is not None],
            [r["macro_effect"] for r in null_blocks["evaluation"]
             if r.get("macro_effect") is not None],
            args.alpha,
        )
        structural["enabled"] = True

    summary_grid = {cell: summarize(recs) for cell, recs in sorted(cells.items())}
    elapsed = time.perf_counter() - started_all

    # attach the rule a rate is valid for, and never call a rate a
    # false-positive rate unless the scenario satisfies that rule's null
    for cell, block in summary_grid.items():
        meta = cell_meta[cell]
        info = SCENARIO_INFO[meta["scenario"]]
        lam = meta["lambda"]
        improvement_null = info["null_for_improvement"]
        if meta["scenario"] == "trigram_mixture":
            improvement_null = lam == 0.0
        block.update({
            "scenario": meta["scenario"], "lambda": lam, "size": meta["size"],
            "n_inscriptions": meta["n_inscriptions"],
            "generator": info["generator"],
            "scenario_kind": info["kind"],
            "scenario_note": info["note"],
            "positive_rate_is_false_positive_rate": bool(improvement_null),
            "difference_rate_is_false_positive_rate": False,
            "rate_labels": {
                "difference": ("rejection rate under the two-sided difference "
                               "rule; NOT a false-positive rate here, because the "
                               "fitted trigram genuinely differs from the bigram "
                               "under this generator"),
                "positive_improvement": (
                    "FALSE-POSITIVE RATE for the improvement rule (this scenario "
                    "has no genuine higher-order structure)"
                    if improvement_null else
                    "POWER under the named alternative"
                    if (lam > 0 or meta["scenario"] == "hmm_slots") else
                    "descriptive"),
                "negative": "rate at which the trigram significantly loses",
            },
        })

    manifest = {
        "corpus_sha256": sha256_file(args.corpus),
        "source_sha256": {
            "src/arthanvesana/simulate/generators.py": sha256_file(
                ROOT / "src" / "arthanvesana" / "simulate" / "generators.py"),
            "src/arthanvesana/simulate/pipeline.py": sha256_file(
                ROOT / "src" / "arthanvesana" / "simulate" / "pipeline.py"),
            "src/arthanvesana/stats/grouping.py": sha256_file(
                ROOT / "src" / "arthanvesana" / "stats" / "grouping.py"),
            "scripts/run_power_analysis.py": sha256_file(Path(__file__).resolve()),
        },
        "python": platform.python_version(),
        "dependencies": {n: version(n) for n in ("numpy", "pandas", "scipy")},
        "seed": args.seed,
        "replicates": args.replicates,
        "sizes": args.sizes,
        "scenarios": args.scenarios,
        "lambdas": args.lambdas,
        "permutations": args.permutations,
        "bootstrap": args.bootstrap,
        "null_replicates": args.null_replicates,
        "alpha": args.alpha,
        "empirical_tokens": total_tokens,
        "empirical_inscriptions": n_emp,
        "empirical_vocab": len(PROFILE["unigram"]),
        "grouping_policy": "artifact_grouped (artifact/inscription/exact-sequence "
                           "connected components, duplicates merged)",
        "elapsed_seconds": elapsed,
        "resumable": {
            "enabled": resume,
            "layout": "replicates/<cell>/<rep>.json",
            "note": "rerun the same command to continue an interrupted grid",
        },
        "generator_matching": {
            "sampled_from_empirical": [
                "inscription count", "length distribution", "vocabulary size",
                "unigram / sign-frequency distribution",
                "first-order transitions (markov and mixture generators only)",
            ],
            "not_preserved": [
                "artifact relationships", "duplicate / component structure",
                "missingness and completeness",
            ],
        },
        "note": ("generators are calibration instruments, not models of the Indus "
                 "production process or of natural language"),
    }

    summary = {
        "manifest": manifest,
        "observed": observed,
        "structural": structural,
        "grid": summary_grid,
    }

    lines = [
        "Synthetic calibration for the grouped bigram-vs-trigram inference",
        "",
        f"Replicates per cell: {args.replicates}; sizes: {args.sizes}; "
        f"permutations: {args.permutations}; alpha: {args.alpha}.",
        f"Resumable: {'yes' if resume else 'no'} "
        "(per-replicate files under replicates/).",
        "",
        "THREE SEPARATE QUESTIONS (never interchangeable):",
        "  difference         = two-sided sign-flip on the macro group mean",
        "  positive_improvement = one-sided positive sign-flip (trigram better)",
        "  structural         = exceedance vs a fitted first-order null passed",
        "                       through the complete estimation pipeline",
        "",
        "A rate is called a false-positive rate ONLY where the scenario satisfies",
        "the null of that rule. The 'difference' column is a rejection rate and is",
        "NEVER a false-positive rate in this grid: under a first-order generator the",
        "fitted, over-parameterized trigram genuinely loses, which is a real",
        "predictive difference rather than a false discovery of higher-order",
        "structure.",
        "",
    ]
    header = ("cell | n_eval/n_run | mean effect | MCSE | difference | "
              "positive | negative | FP-valid")
    lines.append(header)
    for cell, blk in summary_grid.items():
        if blk["n_evaluable"] == 0:
            lines.append(f"{cell} | 0/{blk['n_runs']} | - | - | - | - | - | -")
            continue
        lines.append(
            f"{cell} | {blk['n_evaluable']}/{blk['n_runs']} | "
            f"{_fmt(blk['mean_effect'])} | {_fmt(blk['mcse_effect'], '.5f')} | "
            f"{_fmt(blk['difference']['rate'], '.3f')} | "
            f"{_fmt(blk['positive_improvement']['rate'], '.3f')} | "
            f"{_fmt(blk['negative']['rate'], '.3f')} | "
            f"{'yes' if blk['positive_rate_is_false_positive_rate'] else 'no'}"
        )
    lines.append("")
    lines.append("Failed / skipped simulations:")
    any_failure = False
    for cell, blk in summary_grid.items():
        if blk["n_failed"] or blk["n_skipped"]:
            any_failure = True
            lines.append(f"  {cell}: failed={blk['n_failed']} "
                         f"skipped={blk['n_skipped']} "
                         f"reasons={blk['failure_reasons']}")
    if not any_failure:
        lines.append("  none")
    lines.append("")
    if structural.get("enabled"):
        lines.append("STRUCTURAL TEST (fitted first-order null: "
                     f"{structural['null_scenario']})")
        lines.append(f"  observed real-corpus macro effect: {observed_effect}")
        lines.append(f"  null replicates: calibration={structural['n_calibration']}, "
                     f"evaluation={structural['n_evaluation']}")
        lines.append(f"  threshold at alpha={structural['alpha']}: "
                     f"{structural['threshold']}")
        lines.append(f"  false-positive rate of the rule on the independent "
                     f"evaluation half: {structural['false_positive_rate']}")
        lines.append(f"  Monte Carlo p for the observed effect: "
                     f"{structural['p_value']} (plus-one, never zero)")
        lines.append(f"  decision: {structural['decision']}")
        lines.append("  limitations: " + "; ".join(structural["limitations"]))
    else:
        lines.append("STRUCTURAL TEST: not run (pass --null-replicates N). "
                     "Without it the higher-order claim stays provisional; a "
                     "two-sided rejection rate is NOT evidence of higher-order "
                     "structure in either direction.")
    lines.extend(["", "Interpretation: the 'positive' column is the false-positive "
                      "rate of the improvement rule ONLY in the cells whose scenario "
                      "satisfies that rule's null (unigram, markov, shuffled_real, "
                      "and trigram_mixture at lam=0) and power in the lam>0 and "
                      "hmm_slots cells. The 'difference' column is not a "
                      "false-positive rate anywhere in this grid. Generators are "
                      "calibration instruments; no claim about the Indus production "
                      "process follows."])
    report = "\n".join(lines) + "\n"

    args.output.mkdir(parents=True, exist_ok=True)
    if args.cache_only:
        if verbose:
            print(f"cache-only: wrote {sum(len(v) for v in cells.values())} "
                  "replicate records; summary skipped", flush=True)
        return summary
    _write_json(args.output / "power_analysis_summary.json", summary)
    (args.output / "power_analysis_report.txt").write_text(report, encoding="utf-8")
    if verbose:
        print(report)
    return summary


if __name__ == "__main__":
    main()