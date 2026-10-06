"""Connect the three stages to the supplied loading/evaluation harness."""
from .feature_extraction import BaselineFeatureExtractor
from .machine_learning import default_baseline
from .preprocessing import make_preprocessor
from .reference.adapter import SPECTRAL_CFG_KEYS
from .reference.sleep_edf import SleepEDFTrack as ReferenceSleepEDFTrack


class SleepEDFTrack(ReferenceSleepEDFTrack):
    """P0 + eleven features + random forest by default.

    Loading, synthetic data, subject splits and reporting reuse the reference
    scaffold. Override a stage method to add a new feature set or classifier.
    """

    SUPPORTED_CFG_KEYS = SPECTRAL_CFG_KEYS | {"preprocess"}

    def preprocess(self, rec, cfg=None):
        cfg = self._cfg(cfg)
        return make_preprocessor(cfg["preprocess"]).transform(rec)

    def extract_features(self, rec, cfg=None):
        return BaselineFeatureExtractor().transform(rec, self._cfg(cfg))

    def feature_names(self, cfg=None):
        return BaselineFeatureExtractor().feature_names()

    def baseline(self, cfg=None):
        cfg = self._cfg(cfg)
        return default_baseline(
            seed=int(cfg["seed"]), imbalance=cfg["imbalance"],
            threshold=float(cfg["threshold"]), smote_k=int(cfg["smote_k"]))
