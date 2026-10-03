# Results log — `Sleep EDF`, `Group 14`

> **Copy this file into YOUR team's project repository** (not this scaffold repo) as
> `RESULTS.md`, and add one row per iteration as you go — not the night before the deadline.
> It exists because Chapter 16 §16.3 defines an iteration as *done* only when it (1) runs end
> to end to a result, (2) reports the primary metric **with its spread**, (3) is committed with
> a note saying what changed and why, and (4) is at least as good as the previous iteration —
> **or** says in writing why the change was kept anyway. This table is where (2), (3) and (4)
> land; `git log` is where (1) becomes checkable. It also makes writing the report an assembly
> job rather than an archaeology project.
>
> **This file is graded.** `CAPSTONE_REPORT_RUBRIC.md` **Criterion 9 — Iteration & revision
> history (3 pts)** takes `RESULTS.md` as its evidence, and it asks for one specific thing that
> a forward-only log cannot show: **at least one earlier decision you went back and changed
> because of a downstream result**, with the symptom that sent you back named. A row that
> *lowered* the headline metric and was kept for a stated reason is full-marks evidence, not a
> weakness. Rows here are also direct evidence for "Reproducibility & engineering" and "Report
> quality & defence".

**Track:** `sleep_edf` ·
**Split unit:** `subject` · **Primary metric:** Cohen's kappa ·
**Evaluation mode:** new-subject (LOSO for this five-subject smoke run)

## Iteration log

Paste the metric straight from the harness — `rep["summary"]` prints the required shape, e.g.
`mean cohens_kappa 0.61 (sd 0.12, range 0.34-0.73 across 8 subjects)`. A pooled number with no
spread is half a result.

| # | Date | What changed & why (one line) | Primary metric **with spread** | Better than previous? | If not — why it was kept | Commit |
|---|---|---|---|---|---|---|
| 1 (smoke only) | 2026-09-30 | Supplied baseline on synthetic medium-difficulty data | mean kappa 0.682 (sd 0.318, range 0.155-0.933 across 5 subjects) | Baseline | Functional check only | Source HEAD `0a48f51`; result log not yet committed |
| 1b (real baseline) | 2026-10-02 | Supplied baseline run end-to-end on real Sleep-EDF data to establish the reference performance before DSP/feature changes | mean cohens_kappa 0.720 (sd 0.119, range 0.589-0.822 across 3 subjects) | Baseline on real data | - | `69d392f` |
| 3 |  |  |  | yes / no |  |  |
| 4 |  |  |  | yes / no |  |  |

*"Better" means better under the same honest harness — same split unit, same evaluation mode,
same seed. A change that lowers the metric can still be the right call (simpler, faster, more
robust across subjects, removes a leak). Say so in the column instead of quietly reverting it:
"kept — κ fell 0.02 but the worst-subject κ rose from 0.18 to 0.31" is a stronger result than a
silent higher mean.*

## Synthetic smoke-test evidence - 2026-09-30

PASS. These are synthetic-data results, not performance on real Sleep-EDF.

### Configuration

- Data: `track.smoke(n_subjects=5, n_epochs=80, seed=0, difficulty="medium")`.
- 400 labelled 30-second epochs, 100 Hz, 11 supplied features per epoch.
- Preprocessing: `none`; spectral estimator: `welch`; feature selection: `none`.
- Classifier: `adapter.default_baseline(seed=0, n_estimators=200, imbalance="balanced")`.
- Pipeline: 200-tree random forest (balanced class weights) -> StandardScaler
- Validation: five LOSO folds; 320 training and 80 test epochs per fold.
- Environment: Python 3.13.3; NumPy 2.5.3; SciPy 1.18.1; scikit-learn 1.9.1; Matplotlib 3.11.2.

### Results

| Pooled metric | Harness value |
|---|---|
| Accuracy | 0.753 |
| Cohen's kappa | 0.677 |
| Macro-F1 | 0.730 |
| Balanced accuracy | 0.747 |

Harness summary: `mean cohens_kappa 0.682 (sd 0.318, range 0.155-0.933 across 5 subjects)`.

| Held-out subject | Epochs | Accuracy | Kappa | Macro-F1 |
|---|---|---|---|---|
| S01 | 80 | 0.9125 | 0.867862 | 0.894755 |
| S02 | 80 | 0.3125 | 0.155470 | 0.354552 |
| S03 | 80 | 0.7125 | 0.615706 | 0.487004 |
| S04 | 80 | 0.9500 | 0.933040 | 0.951562 |
| S05 | 80 | 0.8750 | 0.837662 | 0.862940 |

Confusion matrix: rows are true labels, columns are predicted labels.

| True \ predicted | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| W | 28 | 1 | 0 | 0 | 13 |
| N1 | 7 | 25 | 7 | 0 | 2 |
| N2 | 0 | 38 | 103 | 12 | 2 |
| N3 | 0 | 1 | 12 | 71 | 0 |
| REM | 1 | 1 | 2 | 0 | 74 |

Observed labels: W=42, N1=41, N2=155, N3=84, REM=78. The largest confusion is N2 predicted as N1 (38 epochs). S02 is much weaker than the other subjects, the pooled score alone hides this.

### Code

```bash
import numpy as np
from adapter import default_baseline
from sleep_edf import SleepEDFTrack

track = SleepEDFTrack()
recs = track.smoke(n_subjects=5, n_epochs=80, seed=0, difficulty="medium")
X, y, groups = track.build_dataset(recs)
assert X.shape == (400, 11) and np.isfinite(X).all()
clf = default_baseline(seed=0, n_estimators=200, imbalance="balanced")
rep = track.evaluate(X, y, groups, clf=clf)
assert len(rep["y_pred"]) == len(y) == 400
assert rep["n_groups"] == len(rep["per_fold"]) == 5
assert np.asarray(rep["confusion"]).sum() == 400
assert all(np.isfinite(rep[k]) for k in (
    "accuracy", "cohens_kappa", "macro_f1", "balanced_accuracy"))
repeat = track.run_smoke()
assert np.array_equal(rep["y_true"], repeat["y_true"])
assert np.array_equal(rep["y_pred"], repeat["y_pred"])
print(rep["summary"])
for key in ("accuracy", "cohens_kappa", "macro_f1", "balanced_accuracy",
            "labels", "confusion", "per_group", "spread"):
    print(key, rep[key])
print("PASS: shape, finite values, prediction count, five folds, confusion total, repeatability")
```

## ⚠️ Before you fill in many rows — the garden of forking paths

The harness guarantees that **no single run** leaks a subject. It cannot guarantee anything
about the *sequence* of runs. Every row in the table above is scored on the **same**
group-aware folds, and those are the same folds that produce the number you will report. Try
twelve configurations against one held-out set and report the best, and the winner was chosen
partly because it happens to suit *these particular subjects*. Its margin over the runner-up is
optimistically biased, and nothing in the report will look wrong. This is a **selection effect**,
not a leak — which is exactly why the leakage guard cannot catch it, and why it has to be
handled by how you *plan* the work rather than by the code.

It bites hardest here because the cohorts are small: a dozen comparisons on five or six subjects
is a great deal of selection pressure on very little data.

Two defensible ways to handle it. **Say in the report which one you used** — Criterion 7 grades
the quality of the comparison, and this is part of that quality:

| approach | what you do | what it costs |
|---|---|---|
| **an explicit development set** | hold out a few groups *up front* as a DEV set; compare every design option on DEV only; run the chosen pipeline **once** on the untouched evaluation folds and report that | fewer groups serving each purpose — painful on these cohorts — and a noisier DEV estimate |
| **a comparison budget** | decide in advance how many configurations you will compare (a handful, not a sweep), write them into this file **before** you run them, and report the count | you may miss a better option, and you have to resist re-opening the budget once you have seen the numbers — which is the whole discipline |

Either way, two habits that cost nothing:

- **Record the count.** "We compared 6 configurations" belongs in the report. A team that
  compares six and says so is doing better science than one that compares twenty and shows a
  longer table.
- **Read a small margin as no margin.** If the difference between two options is smaller than
  the spread across your subjects/folds, it is not a result. Say "no measurable difference; we
  kept the simpler one" — that sentence earns marks, and claiming the 0.01 does not.

**Related, and equally invisible: *where* your preprocessing is fitted.** The notebook builds
features for every recording **before** `evaluate()` cuts the folds. That is safe for everything
the scaffold ships, because all of it is **stateless** — a fixed filter, or a threshold estimated
from one epoch's own samples. It stops being safe the moment you add a stage that **learns**:
CSP, PCA/ICA, a cohort-wide z-score, a learned artifact template. Fit one of those over all the
recordings and every held-out subject has already shaped the transform that built the training
features — and `assert_no_subject_leak` will still pass, because it checks group ids and cannot
see what a filter was learned from. A learned stage must be fitted **inside** the fold: as a step
in the classifier pipeline (cloned and refit per fold) or via `select_features()`. If you added
one, record here which of the two you used.

## Decision log — the choices behind the numbers

Rows above say *what happened*; this says *what you chose and why*, which is what §16.4 asks you
to make traceable and what the report's defence is built from. There is no single correct
pipeline here — the scaffold deliberately ships options, not answers. One line per decision;
add rows as the pipeline grows, and note the alternative you rejected.

| Pipeline module | Option chosen | Alternative(s) considered | Why this one (one sentence) | Iteration | Revised later? |
|---|---|---|---|---|---|
| 1. Data loading | *e.g.* 8 subjects, both nights | more subjects, one night each | subject-level split needs both nights inside one group | 1 | — |
| 2. Preprocessing |  |  |  |  |  |
| 3. Feature extraction |  |  |  |  |  |
| 4. Feature selection | *e.g.* `select="none"` | ANOVA `SelectKBest`, tree importances | 14 features vs. ~1 800 epochs — pruning risked more than it saved | 1 | *e.g.* **yes, iter 4** — `select_k=20` was a no-op (harness said so); switched to `k=6` |
| 5. Classification, incl. `imbalance` | *e.g.* `imbalance="balanced"` | `"none"`, `"resample"`, `"threshold"` | *(if you kept the default, say you looked and why — a silent default earns nothing)* |  |  |
| 6. Inference |  |  |  |  |  |
| 7. Reporting |  |  |  |  |  |

## Revisions — the ones that went **backwards** (Criterion 9's actual evidence)

Adding iterations forward is a to-do list. What this section wants is the place a number
downstream sent you back **up** the pipeline. One row is enough; two is a good project.
The notebook's "Decision points on this track" section has a symptom → stage table to
diagnose from.

| # | The downstream result that triggered it | Which earlier decision it indicted | What you changed | What happened to the metric |
|---|---|---|---|---|
| 1 | *e.g.* worst-subject κ 0.14 vs. mean 0.61 | stage 2 — no per-recording normalisation | z-scored band powers within each recording | mean κ 0.61 → 0.59, **worst subject 0.14 → 0.38** — kept |
| 2 |  |  |  |  |

## Who did what

One line per person per iteration — the honest-disclosure requirement of §16.7, and the evidence
`INDIVIDUAL_ASSESSMENT.md` asks for. The seven modules do **not** have to map one-to-one onto
people: one person may own several modules, two people may share one, and ownership may rotate
between iterations. Record what actually happened.

| Iteration | Who | Modules / tasks owned | Reviewed by |
|---|---|---|---|
| 1 | Lili | Ran and verified the supplied baseline on real Sleep-EDF data; inspected LOSO metrics, subject-wise spread and confusion matrix |  |
| 2 |  |  |  |

## Final numbers (fill in once, at the end)

| | Value | Under what split |
|---|---|---|
| Primary metric, development (mean + spread) |  |  |
| Secondary metrics (macro-F1 / balanced accuracy / per-class recall) |  |  |
| Held-out set — **scored once, never tuned against** |  |  |
| Supplied baseline, same harness |  |  |
| Yardstick from the dataset card (human ceiling / benchmark / chance) |  |  |
| **Configurations compared** before settling (a number) |  | on DEV folds / on the evaluation folds — say which |
| How option comparison was kept separate from final reporting |  | *e.g.* "separate DEV groups" / "budget of 6, fixed in advance" |

*Grading: `CAPSTONE_REPORT_RUBRIC.md` (team, 30 pts) and `INDIVIDUAL_ASSESSMENT.md`
(individual, 10 pts).*
