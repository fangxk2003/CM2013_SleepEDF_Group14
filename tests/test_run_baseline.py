"""Offline checks for selecting recordings before baseline loading."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "baseline_runner_under_test", ROOT / "scripts" / "run_baseline.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def psg(subject, night, directory=Path("cache")):
    return directory / f"SC4{subject:02d}{night}E0-PSG.edf"


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.paths = [psg(subject, night)
                      for subject in (0, 1, 2, 5) for night in (1, 2)]

    def test_default_preserves_all_recordings_in_sorted_order(self):
        paths = self.paths + [Path("cache/ST7011J0-PSG.edf")]
        self.assertEqual(runner.select_psgs(list(reversed(paths))), sorted(paths))

    def test_first_three_subjects_include_both_nights(self):
        selected = runner.select_psgs(list(reversed(self.paths)), n_subjects=3)
        self.assertEqual(selected, self.paths[:6])

    def test_explicit_subjects_and_nights(self):
        selected = runner.select_psgs(self.paths, subjects=[5, 0], nights=[2])
        self.assertEqual(selected, [psg(0, 2), psg(5, 2)])

    def test_nights_filter_excludes_non_sleep_cassette_recordings(self):
        paths = self.paths + [Path("cache/ST7011J0-PSG.edf")]
        selected = runner.select_psgs(paths, nights=[1])
        self.assertEqual(selected, self.paths[::2])

    def test_missing_explicit_subject_fails(self):
        with self.assertRaises(ValueError):
            runner.select_psgs(self.paths, subjects=[0, 9])

    def test_requested_count_exceeding_available_subjects_fails(self):
        with self.assertRaises(ValueError):
            runner.select_psgs(self.paths, n_subjects=5)

    def test_missing_requested_night_fails(self):
        paths = [path for path in self.paths if path != psg(1, 2)]
        with self.assertRaises(ValueError):
            runner.select_psgs(paths, subjects=[0, 1], nights=[1, 2])

    def test_first_subjects_are_chosen_before_night_filter(self):
        paths = [path for path in self.paths if path != psg(0, 2)]
        with self.assertRaises(ValueError):
            runner.select_psgs(paths, n_subjects=2, nights=[2])

    def test_default_nights_allow_subject_with_one_available_night(self):
        paths = [path for path in self.paths if path != psg(0, 2)]
        self.assertEqual(runner.select_psgs(paths, subjects=[0, 1]), paths[:3])

    def test_filtered_loso_requires_at_least_two_subjects(self):
        with self.assertRaises(ValueError):
            runner.select_psgs(self.paths, subjects=[0])


class CommandLineTests(unittest.TestCase):
    def test_defaults_do_not_filter_recordings(self):
        args = runner.build_parser().parse_args([])
        self.assertIsNone(args.subjects)
        self.assertIsNone(args.n_subjects)
        self.assertIsNone(args.nights)
        self.assertEqual(args.preprocess, "none")

    def test_subject_and_night_arguments(self):
        args = runner.build_parser().parse_args(
            ["--subjects", "0", "1", "2", "--nights", "1", "2"])
        self.assertEqual(args.subjects, [0, 1, 2])
        self.assertEqual(args.nights, [1, 2])
        args = runner.build_parser().parse_args(["--n-subjects", "3"])
        self.assertEqual(args.n_subjects, 3)

    def test_invalid_arguments_are_rejected(self):
        cases = [
            ["--subjects", "-1"],
            ["--subjects", "83"],
            ["--n-subjects", "1"],
            ["--n-subjects", "0"],
            ["--nights", "3"],
            ["--subjects", "0", "1", "--n-subjects", "2"],
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    runner.build_parser().parse_args(arguments)
                self.assertEqual(raised.exception.code, 2)


class LoaderTests(unittest.TestCase):
    def test_explicit_paths_read_only_selected_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory).resolve()
            selected = [psg(0, 1, cache), psg(0, 2, cache), psg(2, 2, cache)]
            excluded = psg(5, 1, cache)
            for path in selected + [excluded]:
                path.write_bytes(b"offline PSG placeholder")
                path.with_name(path.name[:6] + "EC-Hypnogram.edf").write_bytes(
                    b"offline hypnogram placeholder")

            channels = ["EEG Fpz-Cz", "EOG horizontal", "EMG submental"]
            channel_data = {name: np.arange(3000, dtype=float) + index
                            for index, name in enumerate(channels)}
            raw = Mock()
            raw.info = {"sfreq": 100.0}
            raw.ch_names = channels
            raw.get_data.side_effect = lambda *, picks: channel_data[picks][None, :]
            with patch("mne.io.read_raw_edf", return_value=raw) as read_psg, \
                    patch("mne.read_annotations") as read_hyp, \
                    patch("mne.events_from_annotations", return_value=(
                        np.array([[0, 0, 0]]), {"Sleep stage W": 0})):
                records = runner.SleepEDFTrack().load(
                    str(cache), psg_paths=list(reversed(selected)))

            self.assertEqual([Path(call.args[0]) for call in read_psg.call_args_list], selected)
            self.assertEqual([Path(call.args[0]) for call in read_hyp.call_args_list],
                             [path.with_name(path.name[:6] + "EC-Hypnogram.edf")
                              for path in selected])
            self.assertEqual([record.group for record in records], ["SC400", "SC400", "SC402"])
            self.assertEqual([record.meta["record"] for record in records],
                             [path.name.removesuffix("-PSG.edf") for path in selected])
            for record in records:
                self.assertEqual(record.fs, 100)
                np.testing.assert_array_equal(record.labels, ["W"])
                self.assertEqual(set(record.epochs), {"eeg", "eog", "emg"})
                for key, channel in zip(("eeg", "eog", "emg"), channels):
                    self.assertEqual(record.epochs[key].shape, (1, 3000))
                    np.testing.assert_array_equal(record.epochs[key][0], channel_data[channel])
            raw.resample.assert_not_called()

    def test_empty_explicit_paths_do_not_read_any_edfs(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            path = psg(0, 1, cache)
            path.write_bytes(b"offline PSG placeholder")
            path.with_name(path.name[:6] + "EC-Hypnogram.edf").write_bytes(
                b"offline hypnogram placeholder")
            with patch("mne.io.read_raw_edf") as read_psg, \
                    patch("mne.read_annotations") as read_hyp, \
                    patch("mne.events_from_annotations") as events:
                records = runner.SleepEDFTrack().load(str(cache), psg_paths=[])
            self.assertEqual(records, [])
            read_psg.assert_not_called()
            read_hyp.assert_not_called()
            events.assert_not_called()


class RunnerIntegrationTests(unittest.TestCase):
    def test_main_loads_and_reports_only_selected_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            cache = root / "cache"
            cache.mkdir()
            selected = [psg(subject, night, cache)
                        for subject in (0, 1, 2) for night in (1, 2)]
            for path in selected:
                path.write_bytes(b"offline PSG placeholder")
                # Scorer suffix can differ from the corresponding PSG suffix.
                hyp = path.with_name(path.name[:6] + "EC-Hypnogram.edf")
                hyp.write_bytes(b"offline hypnogram placeholder")
            # An excluded file intentionally lacks a hypnogram: filtering must
            # happen before both pairing validation and signal loading.
            excluded = psg(5, 1, cache)
            excluded.write_bytes(b"excluded PSG placeholder")
            output = root / "results"

            records = [SimpleNamespace(
                group=path.name[:5], fs=100,
                meta={"record": path.name.removesuffix("-PSG.edf")},
                labels=np.array([0, 1]),
                epochs={channel: np.zeros((2, 3000))
                        for channel in ("eeg", "eog", "emg")},
            ) for path in selected]
            labels = np.tile([0, 1], len(selected))
            groups = np.repeat([record.group for record in records], 2)
            track = Mock()

            def load_selected(cache_dir, *, psg_paths):
                self.assertEqual(Path(cache_dir), cache)
                self.assertEqual([Path(path) for path in psg_paths], selected)
                self.assertNotIn(excluded, psg_paths)
                return records

            track.load.side_effect = load_selected
            track.build_dataset.return_value = (np.zeros((len(labels), 11)), labels, groups)
            track.feature_names.return_value = [f"feature_{index}" for index in range(11)]
            report = {
                "summary": "offline test", "y_pred": labels, "labels": [0, 1],
                "confusion": np.array([[6, 0], [0, 6]]),
                "per_fold": [{}, {}, {}], "accuracy": 1.0, "cohens_kappa": 1.0,
                "macro_f1": 1.0, "balanced_accuracy": 1.0,
            }
            arguments = [
                "run_baseline.py", "--cache-dir", str(cache),
                "--output-dir", str(output), "--n-subjects", "3", "--nights", "1", "2",
            ]
            with patch.object(sys, "argv", arguments), \
                    patch.object(runner, "SleepEDFTrack", return_value=track), \
                    patch.object(runner, "default_baseline"), \
                    patch.object(runner, "evaluate_loso", return_value=report) as evaluate, \
                    patch.object(runner, "version", return_value="test"), \
                    patch.object(runner.subprocess, "check_output", return_value="test\n"), \
                    redirect_stdout(io.StringIO()):
                runner.main()

            track.load.assert_called_once()
            self.assertEqual(evaluate.call_args.kwargs["cfg"]["loso_max_groups"], 3)
            provenance = json.loads((output / "provenance.json").read_text())
            self.assertEqual(provenance["subjects"], ["SC400", "SC401", "SC402"])
            self.assertEqual(provenance["selection"],
                             {"subjects": None, "n_subjects": 3, "nights": [1, 2]})
            self.assertEqual([record["record"] for record in provenance["recordings"]],
                             [path.name.removesuffix("-PSG.edf") for path in selected])
            expected_files = {path.name for path in cache.glob("*.edf") if path != excluded}
            self.assertEqual({entry["path"] for entry in provenance["files"]}, expected_files)
            self.assertEqual(len(provenance["files"]), 12)
            saved_report = json.loads((output / "report.json").read_text())
            self.assertEqual(saved_report["y_pred"], labels.tolist())


if __name__ == "__main__":
    unittest.main()
