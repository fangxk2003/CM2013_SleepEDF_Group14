"""Offline checks for reading, comparing and exporting saved evaluations."""
from contextlib import redirect_stderr, redirect_stdout
import csv
import html
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "visualisation_results_under_test", ROOT / "scripts" / "visualisation_results.py")
visualisation = importlib.util.module_from_spec(spec)
# dataclasses resolves annotations through the module registry.
sys.modules[spec.name] = visualisation
spec.loader.exec_module(visualisation)


def tiny_report():
    """Six epochs with one unsupported stage and two classification errors."""
    return {
        "accuracy": 4 / 6,
        "cohens_kappa": 0.25,
        "macro_f1": 0.625,
        "balanced_accuracy": 0.625,
        "labels": ["W", "N1", "REM"],
        "confusion": [[3, 1, 0], [1, 1, 0], [0, 0, 0]],
        "y_true": ["W", "W", "W", "W", "N1", "N1"],
        "y_pred": ["W", "W", "W", "N1", "W", "N1"],
        "split_unit": "subject",
        "primary_metric": "cohens_kappa",
        "y_group": ["SC400", "SC400", "SC400", "SC401", "SC401", "SC401"],
        "y_fold": [0, 0, 0, 1, 1, 1],
        "n_groups": 2,
        "per_group": [
            {"group": "SC400", "n": 3, "accuracy": 1.0,
             "macro_f1": 1.0, "balanced_accuracy": 1.0, "cohens_kappa": None},
            {"group": "SC401", "n": 3, "accuracy": 1 / 3,
             "macro_f1": 0.25, "balanced_accuracy": 0.25, "cohens_kappa": -0.5},
        ],
        "per_fold": [
            {"fold": 0, "n": 3, "accuracy": 1.0,
             "macro_f1": 1.0, "balanced_accuracy": 1.0, "cohens_kappa": None},
            {"fold": 1, "n": 3, "accuracy": 1 / 3,
             "macro_f1": 0.25, "balanced_accuracy": 0.25, "cohens_kappa": -0.5},
        ],
    }


def tiny_provenance():
    return {
        "data_kind": "offline fixture",
        "started_utc": "2026-10-08T12:00:00+00:00",
        "finished_utc": "2026-10-08T12:00:02+00:00",
        "files": [
            {"path": "sleep-data/SC4001E0-PSG.edf", "bytes": 100, "sha256": "a" * 64},
            {"path": "sleep-data/SC4011E0-PSG.edf", "bytes": 100, "sha256": "b" * 64},
        ],
        "config": {"seed": 0, "preprocess": "none", "spectral_method": "welch",
                   "select": "none", "loso_max_groups": 2},
        "selection": {"subjects": [0, 1], "n_subjects": None, "nights": [1]},
        "classifier": {"factory": "offline.random_forest", "seed": 0,
                       "n_estimators": 10, "imbalance": "balanced"},
        "cropping": "none",
        "epoch_seconds": 30,
        "fs": 100,
        "feature_shape": [6, 2],
        "feature_names": ["eeg_delta", "emg_rms"],
        "subjects": ["SC400", "SC401"],
        "recordings": [
            {"record": "SC4001E0", "subject": "SC400", "epochs": 3},
            {"record": "SC4011E0", "subject": "SC401", "epochs": 3},
        ],
        "class_counts": {"W": 4, "N1": 2, "REM": 0},
        "source_head": "fixture",
        "python": "3.11",
        "packages": {"numpy": "fixture"},
    }


def write_pair(directory, *, report=None, provenance=None, encoding="utf-8"):
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "report.json"
    provenance_path = directory / "provenance.json"
    report_path.write_text(json.dumps(tiny_report() if report is None else report),
                           encoding=encoding)
    provenance_path.write_text(
        json.dumps(tiny_provenance() if provenance is None else provenance),
        encoding=encoding)
    return report_path, provenance_path


class StageScoreTests(unittest.TestCase):
    def test_scores_match_hand_calculation_and_preserve_undefined_stage(self):
        scores = visualisation.stage_scores(np.array([[3, 1, 0], [2, 0, 0], [0, 0, 0]]))
        np.testing.assert_array_equal(scores["support"], [4, 2, 0])
        np.testing.assert_array_equal(scores["predicted"], [5, 1, 0])
        np.testing.assert_allclose(scores["precision"], [3 / 5, 0, np.nan], equal_nan=True)
        np.testing.assert_allclose(scores["recall"], [3 / 4, 0, np.nan], equal_nan=True)
        np.testing.assert_allclose(scores["f1"], [2 / 3, 0, np.nan], equal_nan=True)

    def test_zero_predictions_and_zero_support_have_distinct_meanings(self):
        # Class 0 is observed but never predicted; class 1 is predicted but absent.
        scores = visualisation.stage_scores(np.array([[0, 2], [0, 0]]))
        np.testing.assert_allclose(scores["precision"], [np.nan, 0], equal_nan=True)
        np.testing.assert_allclose(scores["recall"], [0, np.nan], equal_nan=True)
        np.testing.assert_array_equal(scores["f1"], [0, 0])


class SavedRunTests(unittest.TestCase):
    def test_loads_bom_json_and_defaults_to_sibling_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            report_path, provenance_path = write_pair(
                Path(temporary) / "iteration", encoding="utf-8-sig")
            run = visualisation.load_run(report_path)
            self.assertEqual(list(run.labels), ["W", "N1", "REM"])
            np.testing.assert_array_equal(run.confusion, tiny_report()["confusion"])
            self.assertEqual(run.n_epochs, 6)
            self.assertEqual(run.report["y_true"], tiny_report()["y_true"])
            self.assertEqual(run.provenance["files"], tiny_provenance()["files"])
            self.assertEqual(visualisation.load_run(
                report_path, provenance_path).n_epochs, 6)

    def test_invalid_confusion_matrices_are_rejected(self):
        invalid_matrices = [
            [3, 1, 0],
            [[3, 1], [1, 1]],
            [[3, 1, 0], [1, 1], [0, 0, 0]],
            [[3, 1, 0], [1, -1, 0], [0, 0, 0]],
            [[3, 1, 0], [1, 0.5, 0], [0, 0, 0]],
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, confusion in enumerate(invalid_matrices):
                with self.subTest(confusion=confusion):
                    report = tiny_report()
                    report["confusion"] = confusion
                    report_path, provenance_path = write_pair(root / str(index), report=report)
                    with self.assertRaises(ValueError):
                        visualisation.load_run(report_path, provenance_path)

    def test_prediction_length_and_confusion_disagreement_are_rejected(self):
        shorter_prediction = tiny_report()
        shorter_prediction["y_pred"].pop()
        disagreement = tiny_report()
        disagreement["y_pred"][0] = "N1"
        shorter_truth = tiny_report()
        shorter_truth["y_true"].pop()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, report in enumerate(
                    [shorter_prediction, disagreement, shorter_truth]):
                with self.subTest(case=index):
                    report_path, provenance_path = write_pair(root / str(index), report=report)
                    with self.assertRaises(ValueError):
                        visualisation.load_run(report_path, provenance_path)

    def test_provenance_population_disagreement_is_rejected(self):
        wrong_epochs = tiny_provenance()
        wrong_epochs["feature_shape"][0] = 7
        wrong_classes = tiny_provenance()
        wrong_classes["class_counts"]["W"] = 5
        wrong_subjects = tiny_provenance()
        wrong_subjects["subjects"] = ["SC400", "SC402"]
        with tempfile.TemporaryDirectory() as temporary:
            for index, provenance in enumerate([wrong_epochs, wrong_classes, wrong_subjects]):
                with self.subTest(case=index):
                    paths = write_pair(Path(temporary) / str(index), provenance=provenance)
                    with self.assertRaises(ValueError):
                        visualisation.load_run(*paths)

    def test_sample_spread_requires_two_finite_group_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = visualisation.load_run(*write_pair(Path(temporary) / "iteration"))
            np.testing.assert_allclose(run.spread("cohens_kappa"), [-0.5, np.nan],
                                       equal_nan=True)
            np.testing.assert_allclose(run.spread("accuracy"), [2 / 3, np.sqrt(2) / 3])
            run.report["spread_rows"] = [{"group": "SC400", "cohens_kappa": 1.0}]
            np.testing.assert_allclose(run.spread("cohens_kappa"), [1.0, np.nan],
                                       equal_nan=True)

    def test_cohorts_follow_evaluated_data_and_ignore_pipeline_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original_paths = write_pair(root / "original")
            windows_provenance = tiny_provenance()
            for entry in windows_provenance["files"]:
                entry["path"] = entry["path"].replace("/", "\\")
            windows_provenance["config"]["preprocess"] = "denoise_clipping"
            windows_provenance["classifier"]["factory"] = "offline.other_model"
            windows_provenance["source_head"] = "changed-code"
            windows_paths = write_pair(root / "windows", provenance=windows_provenance)

            changed_labels = tiny_report()
            changed_labels["y_true"][0] = "N1"
            changed_labels["confusion"][0][0] = 2
            changed_labels["confusion"][1][0] = 2
            label_provenance = tiny_provenance()
            label_provenance["class_counts"] = {"W": 3, "N1": 3, "REM": 0}
            label_paths = write_pair(root / "changed_labels", report=changed_labels,
                                     provenance=label_provenance)

            changed_count = tiny_report()
            changed_count["y_true"].append("W")
            changed_count["y_pred"].append("W")
            changed_count["y_group"].append("SC401")
            changed_count["y_fold"].append(1)
            changed_count["confusion"][0][0] = 4
            changed_count["per_fold"][1]["n"] = 4
            changed_count["per_group"][1]["n"] = 4
            count_provenance = tiny_provenance()
            count_provenance["feature_shape"][0] = 7
            count_provenance["recordings"][1]["epochs"] = 4
            count_provenance["class_counts"]["W"] = 5
            count_paths = write_pair(root / "changed_count", report=changed_count,
                                     provenance=count_provenance)
            runs = [visualisation.load_run(*paths) for paths in
                    [original_paths, windows_paths, label_paths, count_paths]]
            visualisation.assign_cohorts(runs)
            self.assertEqual([run.cohort for run in runs], [1, 1, 2, 3])


class DiscoveryTests(unittest.TestCase):
    def test_recursively_discovers_and_deduplicates_explicit_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = write_pair(root / "nested" / "first")
            second = write_pair(root / "second")
            pairs = visualisation.discover_run_pairs(
                [root, first[0].parent, first[0], root])
            self.assertEqual(set(pairs), {first, second})
            self.assertEqual(len(pairs), 2)

    def test_missing_provenance_and_empty_roots_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                visualisation.discover_run_pairs([root])
            report_path, provenance_path = write_pair(root / "incomplete")
            provenance_path.unlink()
            for path in (root, report_path.parent, report_path):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    visualisation.discover_run_pairs([path])


class ExportTests(unittest.TestCase):
    def test_cli_writes_image_csv_and_escaped_html(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_name = 'P0_<iteration & "one">'
            report_path, provenance_path = write_pair(root / run_name)
            output = root / "visualisations"
            arguments = ["--report", str(report_path), "--provenance",
                         str(provenance_path), "--output-dir", str(output), "--dpi", "40"]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(visualisation.main(arguments), 0)
            images = list(output.glob("run_01_*.png"))
            self.assertEqual(len(images), 1)
            self.assertGreater(images[0].stat().st_size, 100)
            self.assertFalse((output / "comparison.png").exists())
            with (output / "summary.csv").open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertTrue(rows)
            self.assertTrue(any(run_name in value for row in rows
                                for value in row.values() if isinstance(value, str)))
            page = (output / "index.html").read_text(encoding="utf-8")
            self.assertIn(html.escape(run_name), page)
            self.assertNotIn(run_name, page)

    def test_multiple_iterations_export_in_time_order_and_note_matching_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            earlier = write_pair(root / "z_earlier")
            later_provenance = tiny_provenance()
            later_provenance["started_utc"] = "2026-10-08T12:01:00+00:00"
            later_provenance["finished_utc"] = "2026-10-08T12:01:02+00:00"
            later = write_pair(root / "a_later", provenance=later_provenance)
            output = root / "visualisations"
            arguments = [str(later[0].parent), str(earlier[0].parent),
                         "--output-dir", str(output), "--dpi", "40"]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(visualisation.main(arguments), 0)
            self.assertGreater((output / "comparison.png").stat().st_size, 100)
            self.assertEqual(len(list(output.glob("run_*.png"))), 2)
            with (output / "summary.csv").open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row["run"] for row in rows], ["z_earlier", "a_later"])
            self.assertEqual([row["cohort"] for row in rows], ["C1", "C1"])
            self.assertEqual([row["cohens_kappa_sd"] for row in rows], ["", ""])
            page = (output / "index.html").read_text(encoding="utf-8")
            self.assertIn("predictions are identical to the previous iteration", page)


if __name__ == "__main__":
    unittest.main()
