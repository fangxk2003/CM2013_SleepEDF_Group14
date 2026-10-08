"""Offline CLI checks for model dispatch and reproducible cohort comparison."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.datasets import make_classification


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "experiment_runner_under_test", ROOT / "scripts" / "run_experiment.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

from sleepedf import SleepEDFTrack
from sleepedf.machine_learning import make_random_forest


class ExperimentCommandLineTests(unittest.TestCase):
    def test_defaults_and_inherited_selection_arguments(self):
        args = runner.build_parser().parse_args([])
        self.assertEqual(args.model, "random_forest")
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.preprocess, "none")
        self.assertIsNone(args.subjects)
        self.assertIsNone(args.n_subjects)
        self.assertIsNone(args.nights)
        args = runner.build_parser().parse_args([
            "--model", "rbf_svm", "--seed", "37", "--subjects", "0", "2",
            "--nights", "2", "--preprocess", "bandpass",
        ])
        self.assertEqual((args.model, args.seed), ("rbf_svm", 37))
        self.assertEqual(args.subjects, [0, 2])
        self.assertEqual(args.nights, [2])
        self.assertEqual(args.preprocess, "bandpass")

    def test_invalid_arguments_fail_before_running(self):
        cases = [
            ["--model", "typo"], ["--seed", "abc"], ["--unknown"],
            ["--seed", "-1"], ["--seed", str(2**32)],
            ["--preprocess", "typo"], ["--nights", "3"],
            ["--subjects", "0", "1", "--n-subjects", "2"],
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), \
                    patch.object(runner, "make_model") as make_model, \
                    patch.object(runner.baseline_runner, "run") as run, \
                    redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    runner.main(arguments)
                self.assertEqual(raised.exception.code, 2)
                make_model.assert_not_called()
                run.assert_not_called()

    def test_each_factory_receives_seed_and_routes_to_shared_runner(self):
        classifier = make_random_forest(seed=37, n_estimators=3)
        for name in ("random_forest", "logistic_regression", "rbf_svm", "xgboost"):
            with self.subTest(model=name), \
                    patch.object(runner, "make_model", return_value=classifier) as make_model, \
                    patch.object(runner.baseline_runner, "run") as run, \
                    redirect_stdout(io.StringIO()):
                runner.main(["--model", name, "--seed", "37", "--n-subjects", "2"])
                make_model.assert_called_once_with(name, seed=37)
                run.assert_called_once()
                arguments = run.call_args.args[0]
                self.assertEqual(arguments.n_subjects, 2)
                kwargs = run.call_args.kwargs
                self.assertIs(kwargs["clf"], classifier)
                self.assertEqual(kwargs["seed"], 37)
                self.assertEqual(kwargs["output_tag"], name)
                self.assertEqual(kwargs["extra_packages"],
                                 ("xgboost",) if name == "xgboost" else ())
                info = kwargs["classifier_info"]
                self.assertEqual(info["factory"], "sleepedf.machine_learning.make_model")
                self.assertEqual(info["model"], name)
                self.assertEqual(info["parameters"]["seed"], 37)

    def test_optional_dependency_failure_is_an_actionable_cli_error(self):
        error = ImportError("XGBoost is optional; install the xgboost extra.")
        stderr = io.StringIO()
        with patch.object(runner, "make_model", side_effect=error), \
                patch.object(runner.baseline_runner, "run") as run, \
                redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                runner.main(["--model", "xgboost"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn(str(error), stderr.getvalue())
        run.assert_not_called()


class ExperimentIntegrationTests(unittest.TestCase):
    def test_model_runs_save_provenance_and_compare_the_same_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            cache = root / "cache"
            cache.mkdir()
            selected = [cache / f"SC4{subject:02d}{night}E0-PSG.edf"
                        for subject in (0, 1) for night in (1, 2)]
            for path in selected:
                path.write_bytes(b"offline PSG placeholder")
                path.with_name(path.name[:6] + "EC-Hypnogram.edf").write_bytes(
                    b"offline hypnogram placeholder")
            excluded = cache / "SC4051E0-PSG.edf"
            excluded.write_bytes(b"excluded PSG without paired hypnogram")
            X, target = make_classification(
                n_samples=24, n_features=11, n_informative=5, n_redundant=2,
                n_classes=3, n_clusters_per_class=1, random_state=8,
            )
            y = np.array(["W", "N1", "REM"])[target]
            groups = np.repeat([path.name[:5] for path in selected], 6)
            records = [SimpleNamespace(
                group=path.name[:5], fs=100,
                meta={"record": path.name.removesuffix("-PSG.edf")},
                labels=y[index * 6:(index + 1) * 6],
                epochs={channel: np.zeros((6, 3000)) for channel in ("eeg", "eog", "emg")},
            ) for index, path in enumerate(selected)]
            provenance = []
            reports = []
            actual_evaluate = runner.baseline_runner.evaluate_loso
            for name in ("logistic_regression", "rbf_svm"):
                track = SleepEDFTrack()
                with self.subTest(model=name), \
                        patch.object(runner.baseline_runner, "ROOT", root), \
                        patch.object(runner.baseline_runner, "SleepEDFTrack", return_value=track), \
                        patch.object(track, "load", return_value=records) as load, \
                        patch.object(track, "build_dataset", return_value=(X, y, groups)) as build, \
                        patch.object(runner.baseline_runner, "evaluate_loso", wraps=actual_evaluate) as evaluate, \
                        patch.object(runner.baseline_runner, "version", return_value="test"), \
                        patch.object(runner.baseline_runner.subprocess, "check_output", return_value="test\n"), \
                        redirect_stdout(io.StringIO()):
                    arguments = [
                        "--cache-dir", str(cache),
                        "--n-subjects", "2", "--nights", "1", "2",
                        "--model", name, "--seed", "37", "--preprocess", "wavelet",
                    ]
                    output = runner.main(arguments)
                    self.assertEqual(output.parent, root / "results")
                    self.assertRegex(output.name, rf"^real__P2_{name}_\d{{8}}T\d{{12}}Z$")
                    load.assert_called_once_with(str(cache), psg_paths=selected)
                    self.assertIs(build.call_args.args[0], records)
                    cfg = build.call_args.args[1]
                    self.assertEqual(cfg["seed"], 37)
                    self.assertEqual(cfg["preprocess"], "wavelet")
                    self.assertEqual(cfg["loso_max_groups"], 2)
                    self.assertIs(evaluate.call_args.args[1], X)
                    self.assertIs(evaluate.call_args.args[2], y)
                    self.assertIs(evaluate.call_args.args[3], groups)
                    classifier = evaluate.call_args.kwargs["clf"]
                    self.assertEqual(classifier.named_steps["clf"].random_state, 37)
                    self.assertFalse(hasattr(classifier, "classes_"))

                    saved = json.loads((output / "provenance.json").read_text())
                    self.assertEqual(saved["classifier"]["model"], name)
                    self.assertEqual(saved["classifier"]["parameters"]["seed"], 37)
                    self.assertEqual(saved["config"]["preprocess"], "wavelet")
                    self.assertEqual(saved["subjects"], ["SC400", "SC401"])
                    self.assertEqual(len(saved["files"]), 8)
                    self.assertNotIn(excluded.name, [entry["path"] for entry in saved["files"]])
                    report = json.loads((output / "report.json").read_text())
                    self.assertEqual(len(report["y_pred"]), len(y))
                    self.assertEqual(len(report["per_fold"]), 2)
                    provenance.append(saved)
                    reports.append(report)

                    original_report = (output / "report.json").read_bytes()
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                        runner.main(arguments + ["--output-dir", str(output)])
                    self.assertEqual(raised.exception.code, 2)
                    self.assertEqual((output / "report.json").read_bytes(), original_report)
                    load.assert_called_once()

            for key in ("files", "recordings", "selection", "subjects", "class_counts"):
                self.assertEqual(provenance[0][key], provenance[1][key])
            self.assertEqual(reports[0]["y_true"], reports[1]["y_true"])
            self.assertEqual(reports[0]["y_group"], reports[1]["y_group"])


if __name__ == "__main__":
    unittest.main()
