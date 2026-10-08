"""Optional XGBoost with training-fold label encoding and class weighting."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data


class ClassWeightedXGBClassifier(ClassifierMixin, BaseEstimator):
    """Adapt XGBoost to sleep-stage labels and fold-local class balancing.

    ``estimator`` is an unfitted XGBClassifier. Each fit clones it, encodes the
    observed labels to contiguous integers, and computes class weights from
    those training labels only. Predictions and ``classes_`` use the original
    labels; probability columns follow ``classes_``. Tune the underlying model
    with parameters such as ``estimator__max_depth``.
    """

    def __init__(self, estimator, class_weight="balanced"):
        self.estimator = estimator
        self.class_weight = class_weight

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def fit(self, X, y, sample_weight=None):
        """Fit using only the supplied training rows and labels."""
        X, y = validate_data(self, X, y, ensure_all_finite="allow-nan")
        check_classification_targets(y)
        encoder = LabelEncoder().fit(y)
        if len(encoder.classes_) < 2:
            raise ValueError("XGBoost requires at least two training classes.")

        weights = compute_sample_weight(self.class_weight, y)
        if sample_weight is not None:
            supplied = np.asarray(sample_weight, dtype=float)
            if supplied.shape != (len(y),):
                raise ValueError("sample_weight must have one value per training row.")
            if not np.isfinite(supplied).all() or (supplied < 0).any():
                raise ValueError("sample_weight must be finite and non-negative.")
            weights = weights * supplied

        estimator = clone(self.estimator).set_params(
            objective="multi:softprob", num_class=len(encoder.classes_))
        estimator.fit(X, encoder.transform(y), sample_weight=weights)
        self.estimator_ = estimator
        self.label_encoder_ = encoder
        self.classes_ = encoder.classes_
        return self

    def predict(self, X):
        """Predict original sleep-stage labels without refitting."""
        check_is_fitted(self, ["estimator_", "label_encoder_"])
        X = validate_data(self, X, reset=False, ensure_all_finite="allow-nan")
        # XGBClassifier.predict() treats two observed classes as binary even
        # with multi:softprob, returning a thresholded two-column matrix.
        # The probability columns retain the encoded class order in both cases.
        encoded = np.asarray(self.estimator_.predict_proba(X)).argmax(axis=1)
        return self.label_encoder_.inverse_transform(encoded)

    def predict_proba(self, X):
        """Return probabilities in the order of the original ``classes_``."""
        check_is_fitted(self, ["estimator_", "label_encoder_"])
        X = validate_data(self, X, reset=False, ensure_all_finite="allow-nan")
        return self.estimator_.predict_proba(X)

    @property
    def feature_importances_(self):
        check_is_fitted(self, "estimator_")
        return self.estimator_.feature_importances_


def make_xgboost(*, seed=0, n_estimators=200, max_depth=3,
                 learning_rate=0.05, min_child_weight=1, subsample=0.8,
                 colsample_bytree=1.0, reg_lambda=1.0,
                 class_weight="balanced", n_jobs=1):
    """Return a fresh, regularized boosted-tree classifier.

    XGBoost is imported only when this factory is called. Tree inputs do not
    need feature scaling. Balancing is computed during fit, and can be disabled
    with ``class_weight=None``. Early stopping is deliberately not configured:
    any validation set used for tuning must contain training subjects only.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost is optional. Install it with `uv sync --extra xgboost` "
            "or `pip install -e '.[xgboost]'`.") from exc

    estimator = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, min_child_weight=min_child_weight,
        subsample=subsample, colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda, tree_method="hist", objective="multi:softprob",
        eval_metric="mlogloss", random_state=seed, n_jobs=n_jobs, verbosity=0)
    return ClassWeightedXGBClassifier(estimator, class_weight=class_weight)
