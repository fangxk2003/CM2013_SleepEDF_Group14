"""Offline checks for model comparison, fold-local scaling and class weights."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleepedf import SleepEDFTrack
from sleepedf.machine_learning import (
    default_baseline, evaluate_loso, make_model, make_random_forest,
    make_logistic_regression, make_rbf_svm, make_xgboost,
)
from sleepedf.machine_learning.xgboost import ClassWeightedXGBClassifier


class RecordingScaler(StandardScaler):
    """Capture each cloned scaler's fitted mean, without retaining fold models."""
    fitted_means = []

    def fit(self, X, y=None, sample_weight=None):
        super().fit(X, y, sample_weight=sample_weight)
        type(self).fitted_means.append(self.mean_.copy())
        return self


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    """An XGBoost-like estimator to inspect wrapper behavior without XGBoost."""

    def __init__(self, objective=None, num_class=None):
        self.objective = objective
        self.num_class = num_class

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.n_features_in_ = np.asarray(X).shape[1]
        self.y_ = np.asarray(y).copy()
        self.sample_weight_ = (None if sample_weight is None
                              else np.asarray(sample_weight).copy())
        return self

    def predict(self, X):
        return np.arange(len(X)) % len(self.classes_)

    def predict_proba(self, X):
        return np.eye(len(self.classes_))[self.predict(X)]


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = SleepEDFTrack()
        cls.X, target = make_classification(
            n_samples=90, n_features=11, n_informative=5, n_redundant=2,
            n_classes=5, n_clusters_per_class=1, random_state=8,
        )
        cls.y = np.asarray(["W", "N1", "N2", "N3", "REM"])[target]
        cls.groups = np.tile(np.arange(3), 30)

    def check_loso(self, model, X=None):
        X = self.X if X is None else X
        report = evaluate_loso(self.track, X, self.y, self.groups, clf=model)
        self.assertEqual(report["n_groups"], 3)
        self.assertEqual(len(report["per_fold"]), 3)
        self.assertEqual(np.asarray(report["confusion"]).sum(), len(self.y))
        np.testing.assert_array_equal(report["y_true"],
                                      self.y[np.argsort(self.groups, kind="stable")])
        self.assertTrue(set(report["y_pred"]).issubset(set(self.y)))
        self.assertTrue(np.isfinite(report["macro_f1"]))
        self.assertFalse(hasattr(model, "classes_"))
        return report

    def test_random_forest_preserves_baseline(self):
        model = make_random_forest(seed=7, n_estimators=8, imbalance="balanced")
        baseline = default_baseline(seed=7, n_estimators=8, imbalance="balanced")
        clone(model).fit(self.X, self.y)
        model.fit(self.X, self.y)
        baseline.fit(self.X, self.y)
        np.testing.assert_array_equal(model.predict(self.X), baseline.predict(self.X))
        self.assertEqual(model.named_steps["clf"].random_state, 7)
        self.assertEqual(model.named_steps["clf"].class_weight, "balanced")
        self.check_loso(make_random_forest(n_estimators=8))

    def test_linear_and_svm_scalers_fit_only_training_subjects(self):
        X = self.X + 100 * self.groups[:, None]
        for factory in (make_logistic_regression, make_rbf_svm):
            with self.subTest(model=factory.__name__):
                RecordingScaler.fitted_means = []
                model = factory().set_params(scale=RecordingScaler())
                self.check_loso(model, X)
                self.assertEqual(len(RecordingScaler.fitted_means), 3)
                for group, mean in enumerate(RecordingScaler.fitted_means):
                    np.testing.assert_allclose(mean, X[self.groups != group].mean(axis=0))
                self.assertFalse(hasattr(model.named_steps["scale"], "mean_"))
                self.assertFalse(hasattr(model.named_steps["clf"], "classes_"))

    def test_scaling_models_are_cloneable_and_accept_tuning_parameters(self):
        lr = make_logistic_regression(seed=5, C=0.25, class_weight=None)
        svm = make_rbf_svm(seed=5, C=2, gamma=0.1, probability=True)
        self.assertEqual(lr.named_steps["clf"].C, 0.25)
        self.assertIsNone(lr.named_steps["clf"].class_weight)
        self.assertEqual(svm.named_steps["clf"].kernel, "rbf")
        self.assertEqual(svm.named_steps["clf"].class_weight, "balanced")
        for model in (lr, svm):
            with self.subTest(model=type(model.named_steps["clf"]).__name__):
                copied = clone(model).set_params(clf__C=0.5)
                copied.fit(self.X, self.y)
                self.assertEqual(copied.named_steps["clf"].C, 0.5)
                probabilities = copied.predict_proba(self.X[:4])
                self.assertEqual(probabilities.shape, (4, 5))
                np.testing.assert_allclose(probabilities.sum(axis=1), 1)
                self.assertFalse(hasattr(model, "classes_"))

    def test_registry_normalizes_names_and_rejects_unknown_models(self):
        for name in ("random_forest", "Random-Forest", " random forest "):
            with self.subTest(name=name):
                model = make_model(name, n_estimators=8)
                self.assertEqual(model.named_steps["clf"].n_estimators, 8)
        for name in ("logistic_regression", "LOGISTIC REGRESSION", "rbf-svm"):
            with self.subTest(name=name):
                self.assertEqual(make_model(name, C=0.5).named_steps["clf"].C, 0.5)
        with self.assertRaises(ValueError):
            make_model("typo")

    def test_xgboost_wrapper_is_compatible_with_loso_without_dependency(self):
        model = ClassWeightedXGBClassifier(RecordingClassifier())
        self.check_loso(model)
        self.assertFalse(hasattr(model.estimator, "classes_"))

    @unittest.skipUnless(importlib.util.find_spec("xgboost") is not None,
                         "optional xgboost dependency is not installed")
    def test_real_xgboost_supports_string_labels_and_loso(self):
        model = make_model("XGBoost", n_estimators=5, max_depth=2, seed=5)
        self.check_loso(model)
        copied = clone(model).set_params(estimator__max_depth=1)
        copied.fit(self.X, self.y)
        self.assertEqual(copied.estimator_.max_depth, 1)
        np.testing.assert_array_equal(copied.classes_, np.unique(self.y))
        probabilities = copied.predict_proba(self.X[:4])
        self.assertEqual(probabilities.shape, (4, 5))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=1e-6)
        self.assertFalse(hasattr(model.estimator, "classes_"))

    @unittest.skipUnless(importlib.util.find_spec("xgboost") is not None,
                         "optional xgboost dependency is not installed")
    def test_real_xgboost_handles_binary_stage_subset(self):
        selected = np.isin(self.y, ["N1", "W"])
        X, y = self.X[selected], self.y[selected]
        model = make_xgboost(n_estimators=5, max_depth=2).fit(X, y)
        np.testing.assert_array_equal(model.classes_, ["N1", "W"])
        probabilities = model.predict_proba(X)
        self.assertEqual(probabilities.shape, (len(y), 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=1e-6)
        np.testing.assert_array_equal(model.predict(X),
                                      model.classes_[probabilities.argmax(axis=1)])


class XGBoostWrapperTests(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(24, dtype=float).reshape(8, 3)
        self.y = np.array(["W", "W", "W", "W", "W", "N1", "N1", "REM"])
        self.prototype = RecordingClassifier()
        self.model = ClassWeightedXGBClassifier(self.prototype, class_weight="balanced")

    def test_label_encoding_and_combined_training_weights(self):
        supplied_weights = np.linspace(1, 2, len(self.y))
        model = self.model.fit(self.X, self.y, sample_weight=supplied_weights)
        self.assertIs(model, self.model)
        self.assertIsNot(model.estimator_, self.prototype)
        self.assertFalse(hasattr(self.prototype, "y_"))
        np.testing.assert_array_equal(model.classes_, np.unique(self.y))
        np.testing.assert_array_equal(model.estimator_.y_,
                                      model.label_encoder_.transform(self.y))
        np.testing.assert_allclose(model.estimator_.sample_weight_,
                                   supplied_weights * compute_sample_weight("balanced", self.y))
        self.assertEqual(model.estimator_.objective, "multi:softprob")
        self.assertEqual(model.estimator_.num_class, 3)
        self.assertEqual(model.n_features_in_, 3)
        np.testing.assert_array_equal(model.predict(self.X[:3]), model.classes_)
        np.testing.assert_array_equal(model.predict_proba(self.X[:3]), np.eye(3))

    def test_refit_recomputes_weights_and_classes(self):
        self.model.fit(self.X, self.y)
        first_estimator = self.model.estimator_
        y = np.array(["N1", "REM", "REM", "REM"])
        self.model.fit(self.X[:4], y)
        self.assertIsNot(self.model.estimator_, first_estimator)
        np.testing.assert_array_equal(self.model.classes_, ["N1", "REM"])
        np.testing.assert_allclose(self.model.estimator_.sample_weight_,
                                   compute_sample_weight("balanced", y))
        self.assertEqual(self.model.estimator_.num_class, 2)
        self.assertEqual(first_estimator.num_class, 3)

    def test_prediction_requires_fit_and_training_feature_width(self):
        for name in ("predict", "predict_proba"):
            with self.subTest(method=name), self.assertRaises(NotFittedError):
                getattr(self.model, name)(self.X)
        self.model.fit(self.X, self.y)
        for name in ("predict", "predict_proba"):
            with self.subTest(method=name), self.assertRaises(ValueError):
                getattr(self.model, name)(self.X[:, :2])

    def test_custom_class_weights_and_unweighted_fit(self):
        for class_weight in (None, {"N1": 5, "REM": 2, "W": 1}):
            with self.subTest(class_weight=class_weight):
                model = ClassWeightedXGBClassifier(
                    RecordingClassifier(), class_weight=class_weight,
                ).fit(self.X, self.y, sample_weight=np.arange(1, 9))
                expected = np.arange(1, 9) * compute_sample_weight(class_weight, self.y)
                np.testing.assert_allclose(model.estimator_.sample_weight_, expected)

    def test_clone_nested_parameters_and_single_class_rejection(self):
        self.model.fit(self.X, self.y)
        copied = clone(self.model).set_params(estimator__objective="placeholder")
        self.assertEqual(copied.estimator.objective, "placeholder")
        self.assertFalse(hasattr(copied, "estimator_"))
        self.assertFalse(hasattr(copied.estimator, "classes_"))
        with self.assertRaises(ValueError):
            copied.fit(self.X, np.full(len(self.y), "W"))


if __name__ == "__main__":
    unittest.main()
