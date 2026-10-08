"""Select a fresh estimator for a controlled model comparison."""
from .baseline import make_random_forest
from .logistic_regression import make_logistic_regression
from .svm import make_rbf_svm
from .xgboost import make_xgboost


MODEL_NAMES = ("random_forest", "logistic_regression", "rbf_svm", "xgboost")


def make_model(name="random_forest", **kwargs):
    """Build an unfitted model; pass factory parameters through ``kwargs``.

    Pass the result to ``evaluate_loso(..., clf=model)`` so every model uses
    the same subject-separated folds. This does not alter the track's default.
    """
    factories = dict(zip(MODEL_NAMES, (
        make_random_forest, make_logistic_regression, make_rbf_svm, make_xgboost)))
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in factories:
        raise ValueError(
            f"Unknown model {name!r}; choose from {', '.join(MODEL_NAMES)}.")
    return factories[key](**kwargs)
