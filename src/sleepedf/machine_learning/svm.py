"""RBF support-vector classification of per-epoch features."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_rbf_svm(*, seed=0, C=1.0, gamma="scale", class_weight="balanced",
                 cache_size=256, probability=False):
    """Return a fresh scaler and RBF SVM, with fold-local class weights.

    Tune ``clf__C`` and ``clf__gamma`` on training subjects. Probability
    estimation is disabled by default because LOSO evaluation needs only
    predictions. SVC's optional internal probability calibration does not
    respect subject groups; it is unsuitable for reporting group-validated
    calibration. ``class_weight=None`` disables balancing.
    """
    # sklearn 1.9 deprecates even an explicit False; its omitted default
    # already disables probability estimation.
    probability_options = {} if probability is False else {"probability": probability}
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(
            kernel="rbf", C=C, gamma=gamma, class_weight=class_weight,
            cache_size=cache_size, **probability_options,
            random_state=seed)),
    ])
