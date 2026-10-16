
"""Quantitative evaluation of sleep-spindle candidate preservation.

CM2013 - Sleep-EDF - Group 14

Compare P1 (Butterworth) and P2 (wavelet denoising)
on the same spindle candidates detected from Raw EEG.

This evaluates signal preservation, not detector accuracy.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    ROOT / "results" / "spindle_candidates"
    / "spindle_candidates.csv"
)

OUTPUT_DIR = (
    ROOT / "results" / "spindle_evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PIPELINES = ["P1", "P2"]

COLORS = {
    "P1": "tab:blue",
    "P2": "tab:green",
}


# ============================================================
# 2. LOAD AND VALIDATE RESULTS
# ============================================================

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Candidate CSV not found: {INPUT_CSV}"
    )

df = pd.read_csv(INPUT_CSV)

required_columns = [
    "epoch",
    "candidate_id",
    "Raw_sigma_power_uv2",
    "Raw_sigma_specificity_ratio",
    "Raw_mean_envelope_uv",
]

for pipeline in PIPELINES:
    required_columns += [
        f"{pipeline}_sigma_power_uv2",
        f"{pipeline}_sigma_specificity_ratio",
        f"{pipeline}_mean_envelope_uv",
    ]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns in CSV: {missing}"
    )

# Restrict analysis to the two inspected epochs.
df = df[df["epoch"].isin([1025, 1637])].copy()

df = df.sort_values(
    ["epoch", "candidate_id"]
).reset_index(drop=True)

if df.empty:
    raise ValueError("No candidates found.")

if df.duplicated(["epoch", "candidate_id"]).any():
    raise ValueError("Duplicate candidate identifiers found.")

df["event_label"] = (
    df["epoch"].astype(str)
    + "-C"
    + df["candidate_id"].astype(str)
)

print("=" * 65)
print("CM2013 - SPINDLE PRESERVATION EVALUATION")
print("=" * 65)

print(f"Number of candidates: {len(df)}")
print(f"Epochs: {sorted(df['epoch'].unique())}")


# ============================================================
# 3. COMPUTE PRESERVATION METRICS
# ============================================================

def safe_ratio(numerator, denominator):
    """Return NaN for invalid or zero denominators."""

    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    result = np.full(
        numerator.shape, np.nan, dtype=float
    )

    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > 0)
    )

    result[valid] = (
        numerator[valid] / denominator[valid]
    )

    return result


for pipeline in PIPELINES:

    # A. Absolute sigma power retention
    power_ratio = safe_ratio(
        df[f"{pipeline}_sigma_power_uv2"],
        df["Raw_sigma_power_uv2"],
    )

    df[f"{pipeline}_power_retention_pct"] = (
        100 * power_ratio
    )

    # B. Sigma power change in dB
    df[f"{pipeline}_power_change_db"] = np.where(
        power_ratio > 0,
        10 * np.log10(
            np.maximum(power_ratio, 1e-300)
        ),
        np.nan,
    )

    # C. Change in sigma specificity ratio
    ssr_ratio = safe_ratio(
        df[f"{pipeline}_sigma_specificity_ratio"],
        df["Raw_sigma_specificity_ratio"],
    )

    df[f"{pipeline}_ssr_change_pct"] = (
        100 * (ssr_ratio - 1)
    )

    # D. Mean Hilbert envelope retention
    envelope_ratio = safe_ratio(
        df[f"{pipeline}_mean_envelope_uv"],
        df["Raw_mean_envelope_uv"],
    )

    df[f"{pipeline}_envelope_retention_pct"] = (
        100 * envelope_ratio
    )


# ============================================================
# 4. EVENT-LEVEL RESULTS
# ============================================================

columns = [
    "event_label",
    "P1_power_retention_pct",
    "P2_power_retention_pct",
    "P1_power_change_db",
    "P2_power_change_db",
    "P1_ssr_change_pct",
    "P2_ssr_change_pct",
    "P1_envelope_retention_pct",
    "P2_envelope_retention_pct",
]

print("\nEVENT-LEVEL RESULTS")
print("-" * 65)

print(
    df[columns].round(2).to_string(index=False)
)

df.to_csv(
    OUTPUT_DIR / "preservation_metrics.csv",
    index=False,
)


# ============================================================
# 5. SUMMARY STATISTICS
# ============================================================

metrics = [
    "power_retention_pct",
    "power_change_db",
    "ssr_change_pct",
    "envelope_retention_pct",
]

summary_rows = []

for pipeline in PIPELINES:
    for metric in metrics:

        values = df[f"{pipeline}_{metric}"].dropna()

        summary_rows.append({
            "pipeline": pipeline,
            "metric": metric,
            "n": len(values),
            "mean": values.mean(),
            "std": values.std(ddof=1),
            "median": values.median(),
            "min": values.min(),
            "max": values.max(),
        })

summary = pd.DataFrame(summary_rows)

print("\nSUMMARY STATISTICS")
print("-" * 65)

print(summary.round(2).to_string(index=False))

summary.to_csv(
    OUTPUT_DIR / "preservation_summary.csv",
    index=False,
)


# ============================================================
# 6. PLOT: SIGMA POWER RETENTION
# ============================================================

x = np.arange(len(df))
width = 0.36

fig, ax = plt.subplots(figsize=(12, 5))

for i, pipeline in enumerate(PIPELINES):

    offset = (i - 0.5) * width

    ax.bar(
        x + offset,
        df[f"{pipeline}_power_retention_pct"],
        width,
        label=pipeline,
        color=COLORS[pipeline],
    )

ax.axhline(
    100,
    color="black",
    linestyle="--",
    label="Raw reference",
)

ax.set_xticks(x)
ax.set_xticklabels(df["event_label"])

ax.set_xlabel("Spindle candidate")
ax.set_ylabel("Sigma power retention (%)")

ax.set_title(
    "Sigma-band power preservation across candidates"
)

ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "sigma_power_comparison.png",
    dpi=200,
)

plt.show()


# ============================================================
# 7. PLOT: SSR CHANGE
# ============================================================

fig, ax = plt.subplots(figsize=(12, 5))

for i, pipeline in enumerate(PIPELINES):

    offset = (i - 0.5) * width

    ax.bar(
        x + offset,
        df[f"{pipeline}_ssr_change_pct"],
        width,
        label=pipeline,
        color=COLORS[pipeline],
    )

ax.axhline(
    0,
    color="black",
    linestyle="--",
)

ax.set_xticks(x)
ax.set_xticklabels(df["event_label"])

ax.set_xlabel("Spindle candidate")
ax.set_ylabel("SSR change relative to Raw (%)")

ax.set_title(
    "Change in sigma spectral specificity"
)

ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "ssr_change_comparison.png",
    dpi=200,
)

plt.show()


# ============================================================
# 8. PLOT: HILBERT ENVELOPE RETENTION
# ============================================================

fig, ax = plt.subplots(figsize=(12, 5))

for i, pipeline in enumerate(PIPELINES):

    offset = (i - 0.5) * width

    ax.bar(
        x + offset,
        df[f"{pipeline}_envelope_retention_pct"],
        width,
        label=pipeline,
        color=COLORS[pipeline],
    )

ax.axhline(
    100,
    color="black",
    linestyle="--",
    label="Raw reference",
)

ax.set_xticks(x)
ax.set_xticklabels(df["event_label"])

ax.set_xlabel("Spindle candidate")
ax.set_ylabel("Mean Hilbert envelope retention (%)")

ax.set_title(
    "Spindle-band amplitude preservation"
)

ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "envelope_comparison.png",
    dpi=200,
)

plt.show()


print("\nEvaluation completed.")
print(f"Results saved in: {OUTPUT_DIR}")
