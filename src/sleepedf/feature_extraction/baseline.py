"""The supplied 11 numbers per epoch, in their original order."""
import numpy as np

from bsp import sleep_pipeline as sp
from ..reference.adapter import Recording, spectral_bandpower

FEATURE_NAMES = (
    "eeg_delta", "eeg_theta", "eeg_alpha", "eeg_sigma", "eeg_beta",
    "eeg_hjorth_activity", "eeg_hjorth_mobility", "eeg_hjorth_complexity",
    "eeg_spec_entropy", "eog_movement", "emg_rms",
)


class BaselineFeatureExtractor:
    """Stateless extraction; reuse the supplied numerical feature functions."""

    def feature_names(self):
        return list(FEATURE_NAMES)

    def transform(self, recording: Recording, cfg=None):
        cfg = cfg or {}
        method = str(cfg.get("spectral_method", "welch") or "welch").lower()
        rows = [sp.epoch_features(e, o, m, recording.fs) for e, o, m in zip(
            recording.epochs["eeg"], recording.epochs["eog"], recording.epochs["emg"])]
        if method not in ("welch", "default"): # recalculate spectral bandpower for the first five features
            for row, eeg in zip(rows, recording.epochs["eeg"]):
                for name, band in sp._BANDS.items():
                    row[f"eeg_{name}"] = spectral_bandpower(
                        eeg, recording.fs, band, method=method,
                        ar_order=int(cfg.get("ar_order", 16)),
                        bandwidth=float(cfg.get("mt_bandwidth", 4.0)))
        X = np.asarray([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=float).reshape(-1, len(FEATURE_NAMES))
        return X, np.asarray(recording.labels), recording.group
