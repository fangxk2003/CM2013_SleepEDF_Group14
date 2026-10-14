# Signal diagnostics

These scripts inspect cached recordings and explain the P3/P5 decisions in
[the results log](../../RESULTS.md). They do not download data or train models.
The candidate thresholds and selected events are exploratory evidence; they are
not additional validated preprocessing methods.

Run commands from the repository root after `uv sync --locked`. Every script
reads `sleep_edf_data/`. The P5 plotting scripts save figures to
`results/p5_diagnostics/` (event morphology uses its `morphology/` subfolder).

| Script | Purpose |
| --- | --- |
| `diagnose_p3.py` | Inspect low-variability windows, dropout candidates, and P3 alignment checks. |
| `diagnose_p5.py` | Rank high-amplitude EEG epochs, plot candidates, and inspect repeated extrema. |
| `check_p5_flags.py` | Verify that clipping flags preserve signals, labels, and alignment. |
| `diagnose_p5_other_noise.py` | Print abrupt-change and baseline-variation candidates. |
| `diagnose_p5_eeg_eog_events.py` | Inspect candidate EEG/EOG events and group nearby detections. |
| `plot_p5_noise_examples.py` | Plot the selected SC4012E0 epochs used in the investigation. |
| `plot_p5_event_morphology.py` | Plot selected events at several time scales. |

For example:

```bash
uv run --locked python -B scripts/diagnostics/diagnose_p3.py
uv run --locked python -B scripts/diagnostics/diagnose_p5.py
uv run --locked python -B scripts/diagnostics/check_p5_flags.py
uv run --locked python -B scripts/diagnostics/diagnose_p5_other_noise.py
uv run --locked python -B scripts/diagnostics/diagnose_p5_eeg_eog_events.py
uv run --locked python -B scripts/diagnostics/plot_p5_noise_examples.py
uv run --locked python -B scripts/diagnostics/plot_p5_event_morphology.py
```

For routine experiments, use `scripts/run_experiment.py`; for offline checks,
use `scripts/smoke_test.py` and the `tests/` suite. These diagnostics require the
real cache and some plots use fixed recording IDs and epoch numbers.
