"""Download the original three-subject, first-night Sleep-EDF subset."""
import argparse
from pathlib import Path

from mne.datasets.sleep_physionet.age import fetch_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "sleep_edf_data")
    args = parser.parse_args()
    data_dir = args.cache_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    for psg, hypnogram in fetch_data(subjects=[0, 1, 2], recording=[1], path=str(data_dir)):
        print("Signals:", psg)
        print("Labels: ", hypnogram)


if __name__ == "__main__":
    main()
