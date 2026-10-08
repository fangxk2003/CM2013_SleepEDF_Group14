"""Regression checks for the modular split, using offline synthetic data."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleepedf import SleepEDFTrack
from sleepedf.feature_extraction import BaselineFeatureExtractor, FEATURE_NAMES
from sleepedf.machine_learning import default_baseline, evaluate_loso
from sleepedf.preprocessing import SubjectRecordingNormalisation, make_preprocessor
from sleepedf.reference import report as reference_report
from sleepedf.reference.sleep_edf import SleepEDFTrack as ReferenceSleepEDFTrack


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = SleepEDFTrack()
        cls.reference = ReferenceSleepEDFTrack()
        cls.recordings = cls.track.smoke(n_subjects=3, n_epochs=20, seed=0)

    def test_features_match_supplied_baseline(self):
        for actual, expected in zip(self.track.build_dataset(self.recordings),
                                    self.reference.build_dataset(self.recordings)):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(self.track.feature_names(), self.reference.feature_names())
        self.assertEqual(len(FEATURE_NAMES), 11)

    def test_loso_predictions_match_supplied_baseline(self):
        X, y, groups = self.track.build_dataset(self.recordings)
        actual = evaluate_loso(self.track, X, y, groups)
        expected = self.reference.evaluate(X, y, groups)
        np.testing.assert_array_equal(actual["y_true"], expected["y_true"])
        np.testing.assert_array_equal(actual["y_pred"], expected["y_pred"])
        self.assertEqual(len(actual["per_fold"]), 3)
        self.assertEqual(np.asarray(actual["confusion"]).sum(), len(y))
        panel = self.track.report(actual, show=False)
        for metric in ("cohens_kappa", "macro_f1", "balanced_accuracy", "accuracy"):
            self.assertEqual(round(panel["pooled"][metric], 3), actual[metric])
        self.assertEqual(panel["confusion_md"], reference_report.confusion_table(
            actual["y_true"], actual["y_pred"], labels=actual["labels"]))
        self.assertEqual(panel["spread_unit"], "subject")
        model = default_baseline()
        self.assertEqual(model.named_steps["clf"].n_estimators, 200)
        self.assertEqual(model.named_steps["clf"].class_weight, "balanced")
        self.assertEqual(model.named_steps["clf"].random_state, 0)

    def test_p0_preserves_recording(self):
        rec = self.recordings[0]
        self.assertIs(self.track.preprocess(rec), rec)

    def test_planned_experiments_cannot_silently_run(self):
        for name in ("p4", "normalisation"):
            with self.subTest(name=name), self.assertRaises(NotImplementedError):
                self.track.build_dataset(self.recordings, {"preprocess": name})
        for name in ("p4", "normalisation"):
            normaliser = make_preprocessor(name)
            self.assertIsInstance(normaliser, SubjectRecordingNormalisation)
            with self.assertRaises(NotImplementedError):
                normaliser.transform(self.recordings[0])
        with self.assertRaises(ValueError):
            make_preprocessor("typo")

    def test_empty_recording_retains_feature_width(self):
        rec = deepcopy(self.recordings[0])
        rec.epochs = {k: v[:0] for k, v in rec.epochs.items()}
        rec.labels = rec.labels[:0]
        X, y, group = BaselineFeatureExtractor().transform(rec)
        self.assertEqual(X.shape, (0, 11))
        self.assertEqual(len(y), 0)
        self.assertEqual(group, rec.group)


if __name__ == "__main__":
    unittest.main()
