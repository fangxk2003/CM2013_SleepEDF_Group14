"""Quick visualisation of Sleep-EDF recordings.

Run without arguments to visualise an offline synthetic recording:

	python test.py

To visualise downloaded Sleep-EDF data instead:

	python test.py --cache-dir sleep_edf_data --record 0 --epoch 10
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from sleep_edf import SleepEDFTrack


def plot_recording(record, epoch_index: int = 0) -> None:
	"""Plot one epoch and the label sequence for a recording."""
	n_epochs = len(record.labels)
	if n_epochs == 0:
		raise ValueError("The selected recording contains no labelled epochs.")
	if not 0 <= epoch_index < n_epochs:
		raise IndexError(f"epoch must be between 0 and {n_epochs - 1}")

	signals = {
		"EEG Fpz-Cz": record.epochs["eeg"][epoch_index],
		"EOG horizontal": record.epochs["eog"][epoch_index],
		"EMG submental": record.epochs["emg"][epoch_index],
	}
	time = np.arange(len(next(iter(signals.values())))) / record.fs

	figure, axes = plt.subplots(4, 1, figsize=(12, 9), constrained_layout=True)
	for axis, (name, signal) in zip(axes[:3], signals.items()):
		axis.plot(time, signal, linewidth=0.7)
		axis.set_ylabel(name)
		axis.grid(alpha=0.25)
	axes[2].set_xlabel("Time (s)")

	stage_names = ["W", "N1", "N2", "N3", "REM"]
	stage_to_number = {stage: number for number, stage in enumerate(stage_names)}
	stage_numbers = [stage_to_number.get(label, np.nan) for label in record.labels]
	axes[3].step(np.arange(n_epochs), stage_numbers, where="mid", color="tab:orange")
	axes[3].axvline(epoch_index, color="black", linestyle="--", linewidth=1)
	axes[3].set_xlabel("Epoch")
	axes[3].set_ylabel("Stage")
	axes[3].set_title("Sleep-stage sequence")
	axes[3].set_yticks(range(len(stage_names)), stage_names)
	axes[3].grid(alpha=0.25)

	record_name = record.meta.get("record", record.group)
	figure.suptitle(
		f"{record_name} | epoch {epoch_index} | label: {record.labels[epoch_index]}"
	)
	plt.show()


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--cache-dir",
		help="Directory containing downloaded Sleep-EDF files; synthetic data is used otherwise.",
	)
	parser.add_argument("--record", type=int, default=0, help="Recording index to plot.")
	parser.add_argument("--epoch", type=int, default=0, help="Epoch index to plot.")
	args = parser.parse_args()

	track = SleepEDFTrack()
	if args.cache_dir:
		recordings = track.load(args.cache_dir)
	else:
		recordings = track.smoke(n_subjects=1, n_epochs=80, seed=0)

	if not recordings:
		raise RuntimeError("No recordings were found.")
	if not 0 <= args.record < len(recordings):
		raise IndexError(f"record must be between 0 and {len(recordings) - 1}")
	plot_recording(recordings[args.record], args.epoch)


if __name__ == "__main__":
	main()
