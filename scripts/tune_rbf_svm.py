"""Tune RBF-SVM C and gamma with nested, subject-separated cross-validation.

Outer loop: leave-one-subject-out, giving an unbiased score of the whole
tune-then-fit procedure. Inner loop: GroupKFold over the outer fold's training
subjects only, so the held-out subject never influences the chosen parameters.
A final search over every subject gives the recommended parameters.

Example: uv run --locked python -B scripts/tune_rbf_svm.py --preprocess bandpass --n-subjects 5
Faster:  ... --search random --n-iter 12 --search-subsample 0.25 --n-jobs -1
Outputs: report.json, tuning.json and provenance.json in a new results directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import platform

import internal._bootstrap  # Make src/ importable without an editable installation.
from internal import run_baseline as baseline_runner
from internal._bootstrap import ROOT

import numpy as np
from scipy.stats import loguniform
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import (
    GridSearchCV, GroupKFold, LeaveOneGroupOut, RandomizedSearchCV,
)

from sleepedf import SleepEDFTrack
from sleepedf.machine_learning import make_rbf_svm
from sleepedf.preprocessing import EXPERIMENT_CODES

KAPPA = make_scorer(cohen_kappa_score)


def gamma_value(value):
    """Accept sklearn's named gammas or a positive float."""
    if value in ("scale", "auto"):
        return value
    gamma = float(value)
    if gamma <= 0:
        raise argparse.ArgumentTypeError("gamma must be positive, 'scale' or 'auto'")
    return gamma


def positive_float(value):
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def fraction(value):
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("fraction must be in (0, 1]")
    return number


def build_parser():
    """Reuse the baseline's preprocessing, cohort and output arguments."""
    parser = baseline_runner.build_parser()
    parser.description = __doc__
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--search", choices=("grid", "random"), default="grid",
                        help="Exhaustive grid over --C x --gamma, or log-uniform random "
                             "sampling within their ranges (default: grid).")
    parser.add_argument("--C", type=positive_float, nargs="+", default=[0.1, 1, 10, 100],
                        help="C values for the grid; min/max bound random search.")
    parser.add_argument("--gamma", type=gamma_value, nargs="+",
                        default=["scale", 0.001, 0.01, 0.1, 1.0],
                        help="gamma values for the grid; numeric min/max bound random search.")
    parser.add_argument("--n-iter", type=int, default=15,
                        help="Candidates sampled per search when --search random (default: 15).")
    parser.add_argument("--inner-splits", type=int, default=3,
                        help="GroupKFold splits over training subjects (default: 3).")
    parser.add_argument("--search-subsample", type=fraction, default=1.0,
                        help="Fraction of training epochs (stratified by subject) used in the "
                             "search only; the chosen model is refitted on all of them.")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel candidate fits; -1 uses every core (default: 1).")
    parser.add_argument("--skip-outer", action="store_true",
                        help="Only run the final all-subject search (no unbiased score).")
    return parser


def subsample(groups, frac, rng):
    """Indices keeping ``frac`` of each subject's epochs, so no subject is dropped."""
    if frac >= 1:
        return np.arange(len(groups))
    keep = []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        keep.append(rng.choice(idx, size=max(1, round(frac * len(idx))), replace=False))
    return np.sort(np.concatenate(keep))


def make_search(args, n_subjects, seed):
    estimator = make_rbf_svm(seed=seed)
    cv = GroupKFold(n_splits=min(args.inner_splits, n_subjects))
    common = dict(scoring=KAPPA, cv=cv, n_jobs=args.n_jobs, refit=False,
                  error_score="raise")
    if args.search == "grid":
        grid = {"clf__C": args.C, "clf__gamma": args.gamma}
        return GridSearchCV(estimator, grid, **common)
    numeric_gamma = [g for g in args.gamma if not isinstance(g, str)]
    if len(numeric_gamma) < 2 or len(args.C) < 2:
        raise ValueError("--search random needs at least two numeric --C and --gamma bounds.")
    dists = {"clf__C": loguniform(min(args.C), max(args.C)),
             "clf__gamma": loguniform(min(numeric_gamma), max(numeric_gamma))}
    return RandomizedSearchCV(estimator, dists, n_iter=args.n_iter, random_state=seed,
                              **common)


def tune(args, X, y, groups, seed):
    """Search on (optionally subsampled) data; return best params and a summary."""
    rng = np.random.default_rng(seed)
    take = subsample(groups, args.search_subsample, rng)
    n_subjects = len(np.unique(groups))
    if n_subjects < 2:
        raise ValueError("Inner search needs at least two training subjects.")
    search = make_search(args, n_subjects, seed)
    search.fit(X[take], y[take], groups=groups[take])
    res = search.cv_results_
    order = np.argsort(res["rank_test_score"])
    candidates = [{"C": res["param_clf__C"][i], "gamma": res["param_clf__gamma"][i],
                   "mean_kappa": res["mean_test_score"][i],
                   "std_kappa": res["std_test_score"][i],
                   "mean_fit_seconds": res["mean_fit_time"][i]} for i in order]
    best = {"C": search.best_params_["clf__C"], "gamma": search.best_params_["clf__gamma"]}
    return best, {"best": best, "best_inner_kappa": search.best_score_,
                  "n_search_epochs": len(take), "candidates": candidates}


def nested_loso(args, track, X, y, groups):
    """Outer LOSO: tune on training subjects, refit, predict the held-out one."""
    yt, yp, gg, ff, folds = [], [], [], [], []
    splits = list(LeaveOneGroupOut().split(X, y, groups))
    for k, (tr, te) in enumerate(splits):
        held_out = groups[te][0]
        if set(groups[tr]) & set(groups[te]):
            raise AssertionError("Subject leak between outer train and test folds.")
        best, summary = tune(args, X[tr], y[tr], groups[tr], args.seed + k)
        clf = make_rbf_svm(seed=args.seed, **best).fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        kappa = cohen_kappa_score(y[te], pred)
        print(f"  fold {k + 1}/{len(splits)} subject={held_out}: C={best['C']:.4g}, "
              f"gamma={best['gamma']}, inner kappa={summary['best_inner_kappa']:.3f}, "
              f"outer kappa={kappa:.3f}", flush=True)
        yp.extend(pred); yt.extend(y[te]); gg.extend(groups[te].tolist()); ff.extend([k] * len(te))
        folds.append({"fold": k, "held_out_subject": held_out, "outer_kappa": kappa,
                      **summary})
    rep = track._make_report(yt, yp, groups=gg, folds=ff, n_groups=len(splits))
    return rep, folds


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.inner_splits < 2:
        parser.error("--inner-splits must be at least 2")

    cache = args.cache_dir.resolve()
    psgs = sorted(cache.rglob("*-PSG.edf"))
    if not psgs:
        parser.error(f"No PSG EDF files found under {cache}")
    try:
        psgs = baseline_runner.select_psgs(psgs, subjects=args.subjects,
                                           n_subjects=args.n_subjects, nights=args.nights)
        input_files = baseline_runner.pair_hypnograms(psgs)
    except ValueError as error:
        parser.error(str(error))
    if args.search == "random" and (len(args.C) < 2 or
                                    sum(not isinstance(g, str) for g in args.gamma) < 2):
        parser.error("--search random needs at least two numeric --C and --gamma bounds")

    started = datetime.now(timezone.utc)
    output = (args.output_dir or ROOT / "results" /
              f"real__{EXPERIMENT_CODES[args.preprocess]}_rbf_svm_tuned_"
              f"{started.strftime('%Y%m%dT%H%M%S%fZ')}").resolve()
    if output.exists():
        parser.error(f"Output directory already exists: {output}")

    cfg = {"seed": args.seed, "preprocess": args.preprocess, "spectral_method": "welch",
           "select": "none"}
    track = SleepEDFTrack()
    recs, subjects = baseline_runner.load_recordings(track, cache, psgs)
    X, y, groups = baseline_runner.extract_features(track, recs, cfg, subjects)
    groups = np.asarray(groups)

    rep = folds = None
    if not args.skip_outer:
        print(f"Nested LOSO: {len(subjects)} outer folds, {args.search} search, "
              f"{args.inner_splits} inner GroupKFold splits", flush=True)
        rep, folds = nested_loso(args, track, X, y, groups)

    print("Final search over all subjects...", flush=True)
    best, final = tune(args, X, y, groups, args.seed)
    print(f"Recommended: C={best['C']:.4g}, gamma={best['gamma']}, "
          f"inner kappa={final['best_inner_kappa']:.3f}", flush=True)

    search_settings = {k: getattr(args, k) for k in (
        "search", "C", "gamma", "n_iter", "inner_splits", "search_subsample", "seed")}
    tuning = {"scoring": "cohens_kappa", "settings": search_settings,
              "recommended": best, "final_search": final, "outer_folds": folds}
    provenance = {
        "data_kind": "real Sleep-EDF, not synthetic", "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(), "cache_dir": str(cache),
        "files": [{"path": str(p.relative_to(cache)), "bytes": p.stat().st_size,
                   "sha256": baseline_runner.sha256(p)} for p in input_files],
        "config": cfg,
        "selection": {"subjects": args.subjects, "n_subjects": args.n_subjects,
                      "nights": args.nights},
        "classifier": {"factory": "sleepedf.machine_learning.make_rbf_svm",
                       "tuning": "nested LOSO (outer) / GroupKFold (inner)",
                       **search_settings},
        "feature_shape": X.shape, "feature_names": track.feature_names(),
        "subjects": subjects,
        "class_counts": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("numpy", "scipy", "scikit-learn", "mne")},
        "source_head": baseline_runner.git_output("rev-parse", "HEAD").strip(),
        "working_tree_status": baseline_runner.git_output("status", "--short"),
        "source_sha256": baseline_runner.source_hashes(),
    }
    output.mkdir(parents=True, exist_ok=False)
    baseline_runner.write_json(output / "tuning.json", tuning)
    baseline_runner.write_json(output / "provenance.json", provenance)
    if rep is not None:
        baseline_runner.write_json(output / "report.json", rep)
        print(rep["summary"])
        for key in ("accuracy", "cohens_kappa", "macro_f1", "balanced_accuracy"):
            print(f"{key}: {rep[key]}")
    print(f"Saved results to {output}")
    return output


if __name__ == "__main__":
    main()
