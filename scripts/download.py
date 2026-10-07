"""Download selected Sleep-EDF Sleep-Cassette subjects and nights."""
import argparse
from pathlib import Path

from mne.datasets.sleep_physionet.age import fetch_data


def subject_id(value):
    subject = int(value)
    if not 0 <= subject <= 82:
        raise argparse.ArgumentTypeError("subject IDs must be between 0 and 82")
    return subject


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "sleep_edf_data")
    parser.add_argument("--subjects", type=subject_id, nargs="+", default=[0, 1, 2],
                        help="Sleep-Cassette subject IDs (default: 0 1 2)")
    parser.add_argument("--nights", type=int, nargs="+", choices=[1, 2], default=[1],
                        help="night indices (default: 1)")
    parser.add_argument("--base-url", help="optional Sleep-Cassette download URL")
    args = parser.parse_args()
    data_dir = args.cache_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    subjects = list(dict.fromkeys(args.subjects))
    nights = list(dict.fromkeys(args.nights))
    print("Requested subjects:", subjects)
    print("Requested nights:", nights)
    fetch_options = {}
    if args.base_url is not None:
        fetch_options["base_url"] = args.base_url
    recordings = fetch_data(subjects=subjects, recording=nights, path=str(data_dir),
                            on_missing="warn", **fetch_options)
    for psg, hypnogram in recordings:
        print("Signals:", psg)
        print("Labels: ", hypnogram)
    subject_count = len({Path(psg).name[:5] for psg, _ in recordings})
    print(f"Available: {subject_count} subjects, {len(recordings)} PSG/hypnogram pairs")


if __name__ == "__main__":
    main()
