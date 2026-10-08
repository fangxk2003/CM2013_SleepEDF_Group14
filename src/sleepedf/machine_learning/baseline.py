"""Model factory; preserve the supplied scaler + random forest baseline."""
from ..reference.adapter import default_baseline as _supplied_baseline


def default_baseline(seed=0, n_estimators=200, imbalance="balanced",
                     threshold=0.5, smote_k=5):
    """Return a fresh estimator, fitted separately inside each LOSO fold."""
    return _supplied_baseline(
        seed=seed, n_estimators=n_estimators, imbalance=imbalance,
        threshold=threshold, smote_k=smote_k)


def make_random_forest(*, seed=0, n_estimators=200, imbalance="balanced",
                       threshold=0.5, smote_k=5):
    """Construct the supplied random forest, including its imbalance options.

    The default remains the recorded 200-tree baseline. For tree tuning with
    the default pipeline, use e.g. ``set_params(clf__max_depth=5)``.
    """
    return default_baseline(
        seed=seed, n_estimators=n_estimators, imbalance=imbalance,
        threshold=threshold, smote_k=smote_k)
