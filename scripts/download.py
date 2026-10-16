"""Download selected Sleep-EDF Sleep-Cassette subjects and nights.

Run from the repository root:

    python scripts/download.py                          # subjects 0 1 2, nights 1 2
    python scripts/download.py --subjects 0 1 2 3 4     # choose subjects
    python scripts/download.py --subjects 5 --nights 1  # one night only
    python scripts/download.py --cache-dir /path/to/data
    download all 78 subjects: python scripts/download.py --subjects $(seq 0 82)

Files go to <repo>/sleep_edf_data by default. Already-downloaded files are
reused, and missing recordings are reported as warnings instead of errors.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mne.datasets.sleep_physionet.age import fetch_data

from internal.cli import subject_id


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "sleep_edf_data")
    parser.add_argument("--subjects", type=subject_id, nargs="+", default=[0, 1, 2],
                        help="Sleep-Cassette subject IDs (default: 0 1 2)")
    parser.add_argument("--nights", type=int, nargs="+", choices=[1, 2], default=[1, 2],
                        help="night indices (default: 1 2)")
    parser.add_argument("--base-url", help="optional Sleep-Cassette download URL")
    parser.add_argument("--workers", type=int, default=4,
                        help="subjects to download in parallel (default: 4)")
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
    # One fetch_data call per subject, run in threads, so several connections download at once.
    def fetch_subject(subject):
        return fetch_data(subjects=[subject], recording=nights, path=str(data_dir),
                          on_missing="warn", **fetch_options)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(fetch_subject, subjects))
    recordings = [pair for result in results for pair in result]
    for psg, hypnogram in recordings:
        print("Signals:", psg)
        print("Labels: ", hypnogram)
    subject_count = len({Path(psg).name[:5] for psg, _ in recordings})
    print(f"Available: {subject_count} subjects, {len(recordings)} PSG/hypnogram pairs")


if __name__ == "__main__":
    main()
