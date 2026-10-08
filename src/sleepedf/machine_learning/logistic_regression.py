"""Regularized linear classification of per-epoch features."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic_regression(*, seed=0, C=1.0, max_iter=2000,
                             class_weight="balanced", solver="lbfgs",
                             tol=1e-4):
    """Return a fresh scaler and L2-regularized logistic regression.

    Both scaling and class weights are learned from the training fold only.
    Tune ``clf__C`` through subject-aware cross-validation; smaller C means
    stronger regularization. ``class_weight=None`` disables balancing.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=C, max_iter=max_iter, class_weight=class_weight,
            solver=solver, tol=tol, random_state=seed)),
    ])
