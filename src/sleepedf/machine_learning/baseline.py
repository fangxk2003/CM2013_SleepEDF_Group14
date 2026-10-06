"""Model factory; preserve the supplied scaler + random forest baseline."""
from ..reference.adapter import default_baseline as _supplied_baseline


def default_baseline(seed=0, n_estimators=200, imbalance="balanced",
                     threshold=0.5, smote_k=5):
    """Return a fresh estimator, fitted separately inside each LOSO fold."""
    return _supplied_baseline(
        seed=seed, n_estimators=n_estimators, imbalance=imbalance,
        threshold=threshold, smote_k=smote_k)
