"""Leave-one-subject-out evaluation using the supplied evaluation harness."""


def evaluate_loso(track, X, y, groups, clf=None, cfg=None):
    """Evaluate all subjects; scaling/selection are fitted within each fold."""
    config = dict(cfg or {})
    config["loso_max_groups"] = len(set(groups))
    return track.evaluate(X, y, groups, clf=clf, cfg=config)
