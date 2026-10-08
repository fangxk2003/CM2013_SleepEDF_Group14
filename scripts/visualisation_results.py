"""Visualise saved evaluation iterations without reloading EDFs or training models.

From the repository root::

    python scripts/visualisation_results.py
    python scripts/visualisation_results.py results/real__P0_<timestamp>
    python scripts/visualisation_results.py --report results/<run>/report.json \\
        --provenance results/<run>/provenance.json --output-dir results/plots

Writes a dashboard for each run, an iteration comparison, summary.csv and an
index.html with settings/provenance. Figures use a headless backend by default;
add --show to also open them. Only report.json and provenance.json are read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import re
import textwrap

import matplotlib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS = {
    "cohens_kappa": "Cohen's kappa",
    "macro_f1": "Macro F1",
    "balanced_accuracy": "Balanced accuracy",
    "accuracy": "Accuracy",
}


def number(value) -> float:
    """JSON null and non-finite values represent unavailable metrics."""
    return float(value) if value is not None else float("nan")


def format_score(value) -> str:
    value = number(value)
    return f"{value:.3f}" if np.isfinite(value) else "N/A"


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


@dataclass
class ResultRun:
    report_path: Path
    provenance_path: Path
    report: dict
    provenance: dict
    labels: list[str]
    confusion: np.ndarray
    cohort: int = 0

    @property
    def name(self) -> str:
        return self.report_path.parent.name

    @property
    def n_epochs(self) -> int:
        return int(self.confusion.sum())

    @property
    def preprocess(self) -> str:
        return str(self.provenance.get("config", {}).get("preprocess", "unknown"))

    @property
    def rows(self) -> list[dict]:
        return (self.report.get("spread_rows") or self.report.get("per_group")
                or self.report.get("per_fold") or [])

    @property
    def unit(self) -> str:
        return str(self.report.get("spread_unit", self.report.get("split_unit", "group")))

    @property
    def duration(self) -> float:
        start = parse_time(self.provenance.get("started_utc"))
        end = parse_time(self.provenance.get("finished_utc"))
        return max(0.0, end - start) if start is not None and end is not None else float("nan")

    def spread(self, metric: str) -> tuple[float, float]:
        values = np.array([number(row.get(metric)) for row in self.rows])
        values = values[np.isfinite(values)]
        if not len(values):
            return float("nan"), float("nan")
        return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def parse_time(value) -> float | None:
    if not value:
        return None
    try:
        # Timestamps without a zone are treated as UTC, independent of the host.
        from datetime import timezone
        time = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return time.replace(tzinfo=timezone.utc).timestamp() if time.tzinfo is None else time.timestamp()
    except ValueError:
        return None


def load_run(report_path: Path, provenance_path: Path | None = None) -> ResultRun:
    report_path = Path(report_path).resolve()
    provenance_path = Path(provenance_path or report_path.with_name("provenance.json")).resolve()
    report, provenance = read_json(report_path), read_json(provenance_path)
    labels = report.get("labels")
    if (not isinstance(labels, list) or not labels or
            not all(isinstance(label, str) for label in labels) or len(set(labels)) != len(labels)):
        raise ValueError(f"{report_path}: labels must be unique, non-empty strings.")
    try:
        confusion = np.asarray(report.get("confusion"), dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{report_path}: invalid confusion counts.") from error
    if (confusion.shape != (len(labels), len(labels)) or not np.isfinite(confusion).all()
            or (confusion < 0).any() or (confusion != np.floor(confusion)).any()
            or confusion.sum() == 0):
        raise ValueError(f"{report_path}: confusion must be a non-empty square matrix of non-negative counts in labels order.")
    n = int(confusion.sum())
    shape = provenance.get("feature_shape")
    if shape is not None and (not isinstance(shape, list) or not shape or shape[0] != n):
        raise ValueError(f"{provenance_path}: feature_shape epoch count disagrees with report.json ({n}).")
    class_counts = provenance.get("class_counts")
    if class_counts is not None:
        expected = dict(zip(labels, confusion.sum(axis=1).tolist()))
        if (not isinstance(class_counts, dict) or
                any(class_counts.get(label, 0) != count for label, count in expected.items()) or
                any(count != 0 for label, count in class_counts.items() if label not in expected)):
            raise ValueError(f"{provenance_path}: class_counts disagrees with report.json.")
    for key in ("y_true", "y_pred", "y_group", "y_fold"):
        if key in report and (not isinstance(report[key], list) or len(report[key]) != n):
            raise ValueError(f"{report_path}: {key} must contain {n} aligned entries.")
    if ("y_true" in report) != ("y_pred" in report):
        raise ValueError(f"{report_path}: y_true and y_pred must be supplied together.")
    if "y_true" in report:
        positions = {label: index for index, label in enumerate(labels)}
        actual = np.zeros_like(confusion)
        try:
            for truth, prediction in zip(report["y_true"], report["y_pred"]):
                actual[positions[truth], positions[prediction]] += 1
        except (KeyError, TypeError) as error:
            raise ValueError(f"{report_path}: predictions contain a label absent from labels.") from error
        if not np.array_equal(actual, confusion):
            raise ValueError(f"{report_path}: confusion disagrees with y_true/y_pred.")
    for array, rows, key in (("y_group", "per_group", "group"), ("y_fold", "per_fold", "fold")):
        if array in report and report.get(rows):
            counts = Counter(map(str, report[array]))
            recorded = {str(row[key]): row.get("n") for row in report[rows]}
            if recorded != counts or len(recorded) != len(report[rows]):
                raise ValueError(f"{report_path}: {rows} support disagrees with {array}.")
    if (report.get("split_unit") == "subject" and "y_group" in report
            and "subjects" in provenance
            and set(map(str, provenance["subjects"])) != set(map(str, report["y_group"]))):
        raise ValueError(f"{provenance_path}: subjects disagrees with report.json y_group.")
    return ResultRun(report_path, provenance_path, report, provenance, labels, confusion)


def discover_run_pairs(inputs: list[Path]) -> list[tuple[Path, Path]]:
    pairs = {}
    for source in inputs:
        source = Path(source).resolve()
        if source.is_file():
            if source.name not in ("report.json", "provenance.json"):
                raise ValueError(f"Expected a results directory, report.json or provenance.json: {source}")
            reports = [source.with_name("report.json")]
        elif source.is_dir():
            reports = [source / "report.json"] if (source / "report.json").is_file() else sorted(source.rglob("report.json"))
        else:
            raise ValueError(f"Input does not exist: {source}")
        if not reports:
            raise ValueError(f"No report.json files found in {source}.")
        for report in reports:
            provenance = report.with_name("provenance.json")
            if not report.is_file() or not provenance.is_file():
                raise ValueError(f"Incomplete result pair in {report.parent}: report.json and provenance.json are both required.")
            pairs[report] = (report, provenance)
    if not pairs:
        raise ValueError("No result pairs found.")
    return list(pairs.values())


def portable_path(value: str) -> str:
    return value.replace("\\", "/")


def cohort_signature(run: ResultRun) -> str:
    """Separate input/evaluation populations; model and environment remain visible separately."""
    provenance, report = run.provenance, run.report
    identity = {
        "data_kind": provenance.get("data_kind"),
        "files": sorted((portable_path(item["path"]).split("/")[-1], item.get("sha256"))
                        for item in provenance.get("files", [])),
        "recordings": sorted((item.get("record"), item.get("subject"), item.get("epochs"))
                             for item in provenance.get("recordings", [])),
        "subjects": sorted(provenance.get("subjects", [])),
        "split_unit": report.get("split_unit"),
        "epoch_seconds": provenance.get("epoch_seconds"),
        "fs": provenance.get("fs"), "cropping": provenance.get("cropping"),
        "n": run.n_epochs,
        "support": dict(zip(run.labels, run.confusion.sum(axis=1).tolist())),
        "y_true": report.get("y_true"), "y_group": report.get("y_group"),
        "y_fold": report.get("y_fold"),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def assign_cohorts(runs: list[ResultRun]) -> None:
    cohorts = {}
    for run in runs:
        signature = cohort_signature(run)
        run.cohort = cohorts.setdefault(signature, len(cohorts) + 1)


def stage_scores(confusion: np.ndarray) -> dict[str, np.ndarray]:
    confusion = np.asarray(confusion, dtype=float)
    support, predicted = confusion.sum(axis=1), confusion.sum(axis=0)
    correct = np.diag(confusion)

    def divide(numerator, denominator):
        return np.divide(numerator, denominator, out=np.full_like(correct, np.nan), where=denominator != 0)

    return {"precision": divide(correct, predicted), "recall": divide(correct, support),
            "f1": divide(2 * correct, support + predicted), "support": support, "predicted": predicted}


def configuration(run: ResultRun) -> dict:
    fields = ("config", "classifier", "selection", "cropping", "epoch_seconds", "fs",
              "feature_names", "python", "packages", "source_head")
    return {key: run.provenance.get(key) for key in fields}


def changes(previous: ResultRun | None, run: ResultRun) -> list[str]:
    if previous is None:
        return ["First iteration in this selection."]
    before, after = configuration(previous), configuration(run)
    result = []
    for key in after:
        if before[key] != after[key]:
            if isinstance(before[key], dict) and isinstance(after[key], dict):
                for field in sorted(before[key].keys() | after[key].keys()):
                    old, new = before[key].get(field), after[key].get(field)
                    if old != new:
                        result.append(f"{key}.{field}: {old} → {new}")
            else:
                result.append(f"{key}: {before[key]} → {after[key]}")
    before_hashes = {portable_path(key): value for key, value in previous.provenance.get("source_sha256", {}).items()}
    after_hashes = {portable_path(key): value for key, value in run.provenance.get("source_sha256", {}).items()}
    modified = sorted(key for key in before_hashes.keys() | after_hashes.keys()
                      if before_hashes.get(key) != after_hashes.get(key))
    if modified:
        result.append("Source hashes changed: " + ", ".join(modified))
    if previous.cohort != run.cohort:
        result.append(f"Evaluation cohort: C{previous.cohort} → C{run.cohort} (inspect input files, groups and evaluated epochs).")
    elif previous.report.get("y_pred") is not None and previous.report["y_pred"] == run.report.get("y_pred"):
        result.append("Out-of-fold predictions are identical to the previous iteration.")
    return result or ["No recorded configuration/source changes from the previous iteration."]


def plot_confusion(axis, run: ResultRun):
    support = run.confusion.sum(axis=1)
    normalised = np.divide(run.confusion, support[:, None], out=np.zeros_like(run.confusion), where=support[:, None] != 0)
    axis.imshow(normalised, vmin=0, vmax=1, cmap="Blues")
    for row in range(len(run.labels)):
        for column in range(len(run.labels)):
            fraction = normalised[row, column]
            label = f"{fraction:.0%}\n{int(run.confusion[row, column]):,}" if support[row] else "N/A\n0"
            axis.text(column, row, label, ha="center", va="center", fontsize=8,
                      color="white" if fraction > 0.55 else "#17263c")
    axis.set_xticks(range(len(run.labels)), run.labels)
    axis.set_yticks(range(len(run.labels)), [f"{label} (n={int(n):,})" for label, n in zip(run.labels, support)])
    axis.set_xlabel("Predicted stage")
    axis.set_ylabel("True stage")
    axis.set_title("Confusion: row percentage and epoch count")


def metric_axis(axis, values, title):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    low = min(-0.05, float(finite.min()) - 0.05) if len(finite) else -0.05
    high = max(1.05, float(finite.max()) + 0.05) if len(finite) else 1.05
    axis.set_ylim(low, high)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)


def plot_dashboard(run: ResultRun, iteration: int):
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(3, 2, figsize=(15, 13), layout="constrained")
    figure.suptitle(f"Iteration {iteration:02d} | {run.preprocess} | cohort C{run.cohort}\n{run.name}", fontsize=15)
    plot_confusion(axes[0, 0], run)
    scores = stage_scores(run.confusion)
    x = np.arange(len(run.labels))
    axis = axes[0, 1]
    for index, metric in enumerate(("precision", "recall", "f1")):
        position = x + (index - 1) * 0.25
        axis.bar(position, scores[metric], width=0.25, label=metric.capitalize())
        for xpos, value in zip(position, scores[metric]):
            axis.text(xpos, value + 0.02 if np.isfinite(value) else 0.02,
                      f"{value:.2f}" if np.isfinite(value) else "N/A", ha="center", fontsize=8, rotation=90)
    axis.set_xticks(x, run.labels)
    axis.set_ylim(0, 1.22)
    axis.set_title("Per-stage scores (N/A = undefined)")
    axis.legend(loc="upper right", ncols=3, fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    axis = axes[1, 0]
    for offset, key, label in ((-0.18, "support", "True"), (0.18, "predicted", "Predicted")):
        percentages = scores[key] / run.n_epochs * 100
        axis.bar(x + offset, percentages, width=0.36, label=label)
        for xpos, value in zip(x + offset, percentages):
            axis.text(xpos, value + 0.7, f"{value:.1f}%", ha="center", fontsize=8)
    axis.set_xticks(x, run.labels)
    axis.set_ylim(0, min(115, max(scores["support"].max(), scores["predicted"].max()) / run.n_epochs * 100 + 12))
    axis.set_ylabel("Evaluated epochs (%)")
    axis.set_title(f"Stage distribution | n={run.n_epochs:,}")
    axis.legend()
    axis = axes[1, 1]
    pooled = [number(run.report.get(metric)) for metric in METRICS]
    means, deviations = zip(*(run.spread(metric) for metric in METRICS))
    mx = np.arange(len(METRICS))
    axis.bar(mx - 0.18, pooled, 0.36, label="Pooled", color="#277da8")
    axis.bar(mx + 0.18, means, 0.36, label=f"{run.unit.capitalize()} mean ± SD", color="#f4a261")
    for xpos, mean, sd in zip(mx + 0.18, means, deviations):
        if np.isfinite(mean) and np.isfinite(sd):
            axis.errorbar(xpos, mean, yerr=sd, fmt="none", color="#333333", capsize=4)
    for positions, values in ((mx - 0.18, pooled), (mx + 0.18, means)):
        for xpos, value in zip(positions, values):
            axis.text(xpos, 0.03, format_score(value), ha="center", fontsize=8, rotation=90)
    axis.set_xticks(mx, METRICS.values(), rotation=15)
    bounds = pooled + [mean - sd for mean, sd in zip(means, deviations)] + [mean + sd for mean, sd in zip(means, deviations)]
    metric_axis(axis, bounds, "Pooled scores and spread (SD is not a confidence interval)")
    axis.legend(fontsize=8)
    axis = axes[2, 0]
    if run.rows:
        values = np.array([[number(row.get(metric)) for metric in METRICS] for row in run.rows])
        image = axis.imshow(np.ma.masked_invalid(values), vmin=-1 if np.any(values < 0) else 0, vmax=1, cmap="YlGnBu", aspect="auto")
        for row, row_values in enumerate(values):
            for column, value in enumerate(row_values):
                axis.text(column, row, format_score(value), ha="center", va="center", fontsize=7 if len(values) > 12 else 9,
                          color="white" if value > 0.65 else "#17263c")
        axis.set_xticks(range(len(METRICS)), METRICS.values(), rotation=15)
        axis.set_yticks(range(len(run.rows)), [f"{row.get('group', row.get('fold', '?'))} (n={row.get('n', '?')})" for row in run.rows], fontsize=7 if len(values) > 12 else 9)
        figure.colorbar(image, ax=axis, shrink=0.7)
    else:
        axis.text(0.5, 0.5, "No per-group/fold scores saved", ha="center", transform=axis.transAxes)
        axis.set_axis_off()
    axis.set_title(f"Variation across {run.unit}s")
    axis = axes[2, 1]
    axis.set_axis_off()
    metadata = [f"Split unit: {run.report.get('split_unit', 'unknown')}; spread unit: {run.unit}",
                f"Primary metric: {run.report.get('primary_metric', 'cohens_kappa')}",
                f"Started UTC: {run.provenance.get('started_utc', 'unknown')}",
                f"Runtime: {run.duration:.1f} s" if np.isfinite(run.duration) else "Runtime: N/A",
                f"Python: {run.provenance.get('python', 'unknown')}",
                f"Source revision: {run.provenance.get('source_head', 'unknown')}",
                "Settings: " + json.dumps(run.provenance.get("config", {}), sort_keys=True),
                "Classifier: " + json.dumps(run.provenance.get("classifier", {}), sort_keys=True),
                "Features: " + str(run.provenance.get("feature_shape", "unknown"))]
    recording_n = sum(item.get("epochs", 0) for item in run.provenance.get("recordings", []))
    if recording_n and recording_n != run.n_epochs:
        metadata.append(f"Input recording epochs: {recording_n:,}; evaluated: {run.n_epochs:,}. Epoch removals are not mapped to recordings in these inputs.")
    metadata.extend(str(note) for note in run.report.get("notes", []))
    axis.text(0, 1, "\n\n".join(textwrap.fill(line, 80) for line in metadata),
              transform=axis.transAxes, va="top", fontsize=9)
    return figure


def plot_comparison(runs: list[ResultRun]):
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(max(12, len(runs) * 0.85), 9), layout="constrained")
    figure.suptitle("Iteration comparison | ordered by start time (UTC)\nCompare within a cohort; configuration, source and environment can also change", fontsize=14)
    x = np.arange(len(runs))
    colours = [plt.get_cmap("tab10")((run.cohort - 1) % 10) for run in runs]
    for axis, (metric, title) in zip(axes.flat, METRICS.items()):
        bounds = []
        for index, run in enumerate(runs):
            pooled = number(run.report.get(metric))
            mean, sd = run.spread(metric)
            bounds.extend((pooled, mean - sd, mean + sd))
            if np.isfinite(pooled):
                axis.scatter(index - 0.1, pooled, marker="o", color=colours[index], s=40)
                axis.annotate(format_score(pooled), (index - 0.1, pooled), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
            if np.isfinite(mean):
                axis.errorbar(index + 0.1, mean, yerr=sd if np.isfinite(sd) else None,
                              fmt="s", color=colours[index], capsize=3, markersize=4)
        axis.set_xticks(x, [f"{index:02d}: {run.preprocess}\nC{run.cohort} | n={run.n_epochs:,}" for index, run in enumerate(runs, 1)], rotation=55, ha="right", fontsize=8)
        axis.set_xlim(-0.7, len(runs) - 0.3)
        metric_axis(axis, bounds, title)
        axis.plot([], [], "o", color="#333333", label="Pooled")
        axis.plot([], [], "s", color="#333333", label="Group/fold mean ± sample SD")
        axis.legend(fontsize=8, loc="lower right")
    return figure


def summary_row(iteration: int, run: ResultRun) -> dict:
    row = {"iteration": iteration, "run": run.name, "cohort": f"C{run.cohort}",
           "started_utc": run.provenance.get("started_utc", ""), "preprocess": run.preprocess,
           "n_epochs": run.n_epochs, "split_unit": run.report.get("split_unit", ""),
           "spread_unit": run.unit, "runtime_seconds": run.duration if np.isfinite(run.duration) else "",
           "python": run.provenance.get("python", ""), "source_head": run.provenance.get("source_head", "")}
    for metric in METRICS:
        mean, sd = run.spread(metric)
        for key, value in ((metric, number(run.report.get(metric))), (f"{metric}_mean", mean), (f"{metric}_sd", sd)):
            row[key] = value if np.isfinite(value) else ""
    row.update({"report_path": str(run.report_path), "provenance_path": str(run.provenance_path),
                "config": json.dumps(run.provenance.get("config", {}), sort_keys=True),
                "classifier": json.dumps(run.provenance.get("classifier", {}), sort_keys=True),
                "packages": json.dumps(run.provenance.get("packages", {}), sort_keys=True)})
    return row


def write_index(runs: list[ResultRun], output: Path, images: list[str], comparison: str | None):
    escape = html.escape

    def media(filename):
        if filename.endswith((".png", ".svg")):
            return f'<a href="{escape(filename)}"><img src="{escape(filename)}" alt="Result visualization"></a>'
        return f'<a href="{escape(filename)}">Open PDF figure</a>'

    parts = ["<!doctype html><html lang='en'><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>Sleep-EDF iteration results</title><style>",
             "body{font:15px system-ui,sans-serif;color:#17263c;background:#f4f7fb;margin:2em auto;padding:0 1em;max-width:1500px}"
             "section{background:white;border-radius:12px;padding:1.5em;margin:1.5em 0}"
             "img{width:100%;height:auto}table{border-collapse:collapse;width:100%}th,td{padding:.7em;text-align:left;border-bottom:1px solid #d8e0ea}"
             "th{background:#e8eef6}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7fb;padding:1em}"
             ".scroll{overflow:auto}a{color:#17658c}li{margin:.4em 0;overflow-wrap:anywhere}",
             "</style><body><h1>Sleep-EDF iteration results</h1>",
             "<p>Iterations are ordered by start time (UTC). Pooled scores weight epochs; group means weight groups equally. "
             "Error bars show sample standard deviation across the saved spread rows, not confidence intervals. Undefined values are N/A; "
             "SD requires at least two finite group scores.</p>",
             "<p>Cohorts distinguish input files, recordings, evaluated true labels, groups/folds and epoch counts. "
             "A matching cohort does not isolate a preprocessing effect: inspect source hashes, settings and package versions. "
             "These JSON inputs do not establish recording timelines or feature importance.</p>",
             '<p><a href="summary.csv">Download summary CSV</a></p><section><h2>All iterations</h2><div class="scroll"><table>',
             "<tr><th>Iteration / preprocessing</th><th>Cohort / epochs</th><th>Pooled kappa</th><th>Group kappa mean ± SD</th><th>Macro F1</th><th>Balanced accuracy</th><th>Run / start UTC</th></tr>"]
    for index, run in enumerate(runs, 1):
        mean, sd = run.spread("cohens_kappa")
        parts.append(f'<tr><td><a href="#run-{index}">{index:02d}: {escape(run.preprocess)}</a></td>'
                     f'<td>C{run.cohort} / {run.n_epochs:,}</td><td>{format_score(run.report.get("cohens_kappa"))}</td>'
                     f'<td>{format_score(mean)} ± {format_score(sd)} ({escape(run.unit)})</td>'
                     f'<td>{format_score(run.report.get("macro_f1"))}</td><td>{format_score(run.report.get("balanced_accuracy"))}</td>'
                     f'<td>{escape(run.name)}<br>{escape(str(run.provenance.get("started_utc", "unknown")))}</td></tr>')
    parts.append("</table></div></section>")
    if comparison:
        parts.append("<section><h2>Comparison across iterations</h2>" + media(comparison) + "</section>")
    for index, (run, filename) in enumerate(zip(runs, images), 1):
        parts.append(f'<section id="run-{index}"><h2>{index:02d}: {escape(run.preprocess)} · C{run.cohort}</h2><p>{escape(run.name)}</p>')
        parts.append(media(filename))
        parts.append("<h3>Changes from the previous selected iteration</h3><ul>")
        parts.extend(f"<li>{escape(change)}</li>" for change in changes(runs[index - 2] if index > 1 else None, run))
        matches = [f"{earlier_index:02d}: {earlier.preprocess}" for earlier_index, earlier in enumerate(runs[:index - 1], 1)
                   if earlier.cohort == run.cohort and run.report.get("y_pred") is not None
                   and earlier.report.get("y_pred") == run.report["y_pred"]]
        if matches:
            parts.append("<li>Out-of-fold predictions also match earlier iterations: " + escape(", ".join(matches)) + ".</li>")
        parts.append("</ul><details><summary>Saved configuration and full provenance</summary><pre>" +
                     escape(json.dumps(run.provenance, indent=2, ensure_ascii=False)) + "</pre></details></section>")
    parts.append("</body></html>")
    (output / "index.html").write_text("\n".join(parts), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path, help="Results roots, run folders or JSON paths to compare (default: repository results/).")
    parser.add_argument("--results-dir", type=Path, help="Alternative results root; cannot be combined with positional paths.")
    parser.add_argument("--report", type=Path, help="Explicit single report.json input.")
    parser.add_argument("--provenance", type=Path, help="Explicit single provenance.json input; defaults to the report's sibling.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "visualisations", help="Output directory (default: results/visualisations).")
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png", help="Figure format (default: png).")
    parser.add_argument("--dpi", type=int, default=150, help="Raster resolution (default: 150).")
    parser.add_argument("--show", action="store_true", help="Also display the generated figures interactively.")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error("--dpi must be positive.")
    if args.paths and args.results_dir:
        parser.error("Use positional paths or --results-dir, not both.")
    if args.report and (args.paths or args.results_dir):
        parser.error("--report cannot be combined with positional paths or --results-dir.")
    if args.provenance and not args.report:
        parser.error("--provenance requires --report.")
    try:
        pairs = [(args.report, args.provenance)] if args.report else discover_run_pairs(args.paths or [args.results_dir or ROOT / "results"])
        runs = [load_run(report, provenance) for report, provenance in pairs]
        runs.sort(key=lambda run: (parse_time(run.provenance.get("started_utc")) is None,
                                  parse_time(run.provenance.get("started_utc")) or 0, run.name))
        assign_cohorts(runs)
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    if not args.show:
        matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    images = []
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        for index, run in enumerate(runs, 1):
            slug = re.sub(r"[^A-Za-z0-9_.-]", "_", run.name)
            filename = f"run_{index:02d}_{slug}.{args.format}"
            figure = plot_dashboard(run, index)
            figure.savefig(output / filename, dpi=args.dpi)
            images.append(filename)
            if not args.show:
                plt.close(figure)
            print(f"{index:02d} | C{run.cohort} | {run.preprocess} | n={run.n_epochs:,} | kappa={format_score(run.report.get('cohens_kappa'))} | {filename}")
        comparison = None
        if len(runs) > 1:
            comparison = f"comparison.{args.format}"
            figure = plot_comparison(runs)
            figure.savefig(output / comparison, dpi=args.dpi)
            if not args.show:
                plt.close(figure)
    rows = [summary_row(index, run) for index, run in enumerate(runs, 1)]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_index(runs, output, images, comparison)
    print(f"\nSaved {len(runs)} iteration(s), {len(set(run.cohort for run in runs))} cohort(s). Open {output / 'index.html'}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
