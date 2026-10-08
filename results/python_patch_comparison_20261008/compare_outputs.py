"""Compare captured intermediates, predictions, controls and historical runs."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np


TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1]
RUNS = TASK / "runs"


def read_json(path):
    return json.loads(path.read_text())


def compare_reports(left, right):
    if len(left["y_pred"]) != len(right["y_pred"]):
        raise ValueError("Different evaluated epoch counts")
    changed = [i for i, (a, b) in enumerate(zip(left["y_pred"], right["y_pred"])) if a != b]
    return {
        "epochs": len(left["y_pred"]), "changed_predictions": len(changed),
        "full_report_identical": left == right,
        "changed_prediction_percent": 100 * len(changed) / len(left["y_pred"]),
        "changed_by_subject": dict(Counter(left["y_group"][i] for i in changed)),
        "identical": {key: left[key] == right[key] for key in (
            "y_true", "y_pred", "y_group", "y_fold", "labels", "confusion",
            "per_group", "per_fold", "spread", "summary",
            "accuracy", "cohens_kappa", "macro_f1", "balanced_accuracy",
        )},
        "metrics": {key: {"left": left[key], "right": right[key]}
                    for key in ("accuracy", "cohens_kappa", "macro_f1", "balanced_accuracy")},
    }


def compare_captures(left_name, right_name):
    left, right = RUNS / left_name, RUNS / right_name
    result = {"left": left_name, "right": right_name}
    result["reports"] = compare_reports(read_json(left / "report.json"),
                                        read_json(right / "report.json"))
    result["arrays"] = {}
    for name in ("features", "labels", "groups"):
        a = np.load(left / f"{name}.npy", allow_pickle=False)
        b = np.load(right / f"{name}.npy", allow_pickle=False)
        compatible = a.shape == b.shape and a.dtype == b.dtype
        result["arrays"][name] = {
            "shape": list(a.shape), "dtype": str(a.dtype),
            "same_shape_and_dtype": compatible,
            "bitwise_identical": compatible and a.tobytes() == b.tobytes(),
            "equal_values": bool(np.array_equal(a, b)),
        }
        if name == "features" and compatible:
            result["arrays"][name].update({
                "max_absolute_difference": float(np.max(np.abs(a - b))),
                "different_values": int(np.count_nonzero(a != b)),
            })
    result["identical_forest_signatures"] = (
        read_json(left / "forest_signatures.json") == read_json(right / "forest_signatures.json"))
    a, b = read_json(left / "environment.json"), read_json(right / "environment.json")
    result["environment_controls"] = {key: a[key] == b[key] for key in (
        "packages", "modules", "site_packages", "thread_settings", "threadpools",
        "numpy_config", "machine", "platform", "isolated", "no_site",
    )}
    # The Homebrew anchor intentionally uses its original interpreter build.
    # The two newly compiled versions must also match compiler/build settings.
    if left_name.startswith("compiled_") and right_name.startswith("compiled_"):
        def normalise_build(value):
            value = value.replace("install-3.11.14", "install-VERSION")
            return value.replace("install-3.11.17", "install-VERSION")

        def effective_flags(value):
            # CPython 3.11.17 repeats the trailing -O2; repeating it has no effect.
            flags = value.split()
            while len(flags) > 1 and flags[-1] == flags[-2]:
                flags.pop()
            return flags

        result["interpreter_build_controls"] = {
            "same_compiler": a["compiler"] == b["compiler"],
            "same_configure_options_except_prefix": (
                normalise_build(a["config_args"]) == normalise_build(b["config_args"])),
            "same_effective_compiler_flags": effective_flags(a["cflags"]) == effective_flags(b["cflags"]),
        }
    a, b = read_json(left / "provenance.json"), read_json(right / "provenance.json")
    result["experiment_controls"] = {key: a[key] == b[key] for key in (
        "files", "config", "selection", "classifier", "cropping", "feature_names",
        "subjects", "recordings", "class_counts", "packages", "source_head", "source_sha256",
    )}
    result["all_identical"] = (
        all(item["bitwise_identical"] for item in result["arrays"].values())
        and result["identical_forest_signatures"]
        and result["reports"]["full_report_identical"]
        and all(result["reports"]["identical"].values())
        and all(result["environment_controls"].values())
        and all(result["experiment_controls"].values())
        and all(result.get("interpreter_build_controls", {}).values())
    )
    return result


def main():
    pairs = {
        "interpreter_patch": ("compiled_3.11.14_repeat1", "compiled_3.11.17_repeat1"),
        "repeatability_314": ("compiled_3.11.14_repeat1", "compiled_3.11.14_repeat2"),
        "repeatability_317": ("compiled_3.11.17_repeat1", "compiled_3.11.17_repeat2"),
        "homebrew_vs_compiled": ("existing_3.11.14", "compiled_3.11.14_repeat1"),
    }
    results = {key: compare_captures(*pair) for key, pair in pairs.items()}
    historical_08 = read_json(ROOT / "results/real__P1_20261008T115727461008Z/report.json")
    historical_10 = read_json(ROOT / "results/real__P1_20261008T120846779566Z/report.json")
    results["historical_08_vs_10"] = compare_reports(historical_08, historical_10)
    results["historical_08_vs_current"] = {
        name: compare_reports(historical_08, read_json(RUNS / name / "report.json"))
        for name in ("existing_3.11.14", "compiled_3.11.14_repeat1", "compiled_3.11.17_repeat1")
    }
    (TASK / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    for key in pairs:
        item = results[key]
        print(f"{key}: all_identical={item['all_identical']}, "
              f"changed_predictions={item['reports']['changed_predictions']}, "
              f"max_feature_difference={item['arrays']['features']['max_absolute_difference']}")
    for name, item in results["historical_08_vs_current"].items():
        print(f"historical run08 vs {name}: changed_predictions={item['changed_predictions']}")
    print(f"Saved {TASK / 'comparison.json'}")


if __name__ == "__main__":
    main()
