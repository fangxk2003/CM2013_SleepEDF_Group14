"""Evaluate a preprocessing/model combination on local Sleep-EDF recordings.

Example: .venv/bin/python run_experiment.py --preprocess bandpass --model rbf_svm --n-subjects 3
Outputs: report.json and provenance.json in a new results directory.
"""
from __future__ import annotations

import argparse
import math

import _bootstrap  # Make src/ importable without an editable installation.
import run_baseline as baseline_runner

from sklearn.base import BaseEstimator

from sleepedf.machine_learning import MODEL_NAMES, make_model


def seed_value(value):
    seed = int(value)
    if not 0 <= seed <= 2**32 - 1:
        raise argparse.ArgumentTypeError("seed must be between 0 and 4294967295")
    return seed


def build_parser():
    """Reuse the baseline's preprocessing, cohort and output arguments."""
    parser = baseline_runner.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--model", choices=MODEL_NAMES, default="random_forest",
        help="Classifier to evaluate (default: random_forest).",
    )
    parser.add_argument(
        "--seed", type=seed_value, default=0,
        help="Random seed shared by the model and evaluation (default: 0).",
    )
    return parser


def estimator_settings(value):
    """Describe effective estimator parameters without non-JSON objects."""
    if isinstance(value, BaseEstimator):
        return {
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "parameters": {
                key: estimator_settings(item)
                for key, item in value.get_params(deep=False).items()
            },
        }
    if isinstance(value, dict):
        return {str(key): estimator_settings(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [estimator_settings(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"non_finite_float": str(value)}
    return baseline_runner.json_ready(value)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        clf = make_model(args.model, seed=args.seed)
    except ImportError as error:
        parser.error(str(error))
    classifier_info = {
        "factory": "sleepedf.machine_learning.make_model",
        "model": args.model,
        "parameters": {"seed": args.seed},
        "estimator_params": estimator_settings(clf),
    }
    print(f"Model={args.model}, preprocessing={args.preprocess}, seed={args.seed}",
          flush=True)
    return baseline_runner.run(
        args, parser, clf=clf, classifier_info=classifier_info, seed=args.seed,
        output_tag=args.model,
        extra_packages=("xgboost",) if args.model == "xgboost" else (),
    )


if __name__ == "__main__":
    main()
