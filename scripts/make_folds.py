#!/usr/bin/env python3
"""
Subject-wise 10-fold CV splits for Sleep-EDF Expanded, Sleep Cassette (SC, 78 subjects).

Every subject (both nights) lands in exactly one test fold. Inside each fold,
a set of validation subjects is carved out of the training subjects for early
stopping / model selection. Folds are optionally stratified by age group (and sex)
using SC-subjects metadata.

Usage (from the project folder)
-------------------------------
  uv run --locked python scripts/make_folds.py

  With no arguments it scans sleep_edf_data/physionet-sleep-data, reads age/sex
  from MNE's bundled age_records.csv, and writes folds.json in the project folder.

  --data-dir  folder with SC4ssN*-PSG.edf files (searched recursively).
  --meta      age/sex table: MNE's age_records.csv (default), SC-subjects.xls
              (needs `xlrd`) or a CSV export of it. Needs columns subject, age,
              sex (sex as F=1/M=2 or female/male). --no-meta disables stratification.
  --subjects  alternative to --data-dir: build folds from the official
              78-subject ID list, without needing the data on disk.

Generate the folds ONCE, commit folds.json, and load it in every experiment.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

# Sleep Cassette subject numbers present in Sleep-EDF Expanded (0..82 minus 39, 68, 69, 78, 79).
OFFICIAL_SC_SUBJECTS = [s for s in range(83) if s not in {39, 68, 69, 78, 79}]
FNAME_RE = re.compile(r"SC4(\d\d)(\d)", re.IGNORECASE)
AGE_BINS = [0, 40, 60, 80, 200]  # 25-39 / 40-59 / 60-79 / 80+


def scan_files(data_dir):
    """Map subject -> sorted list of PSG file names."""
    files = defaultdict(list)
    for p in sorted(Path(data_dir).rglob("*")):
        m = FNAME_RE.match(p.name)
        if m and p.name.upper().endswith("PSG.EDF"):
            files[int(m.group(1))].append(p.name)
    return dict(files)


def load_meta(path):
    """Return subject -> (age, sex) from SC-subjects.xls or a CSV export of it."""
    import pandas as pd

    path = Path(path)
    if path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(path)  # .xls requires `pip install xlrd`
    else:
        df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}

    def col(name):
        for k, v in cols.items():
            if k.startswith(name):
                return v
        sys.exit(f"metadata file is missing a '{name}' column (found: {list(df.columns)})")

    subj_c, age_c, sex_c = col("subject"), col("age"), col("sex")

    def sex_code(v):  # 1 = female, 2 = male
        s = str(v).strip().lower()
        return 1 if s in {"1", "f", "female"} else 2 if s in {"2", "m", "male"} else sys.exit(f"unknown sex value {v!r}")

    meta = {}
    for _, r in df.iterrows():
        meta[int(r[subj_c])] = (float(r[age_c]), sex_code(r[sex_c]))
    return meta


def find_mne_age_records():
    """Path to the age/sex table that ships with MNE, if MNE is installed."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("mne")
    except Exception:
        return None
    if spec is None or not spec.origin:
        return None
    p = Path(spec.origin).parent / "datasets" / "sleep_physionet" / "age_records.csv"
    return p if p.exists() else None


def strat_label(age, sex, use_sex):
    age_bin = int(np.digitize(age, AGE_BINS)) - 1
    return age_bin * 2 + (sex if use_sex else 0)


def make_folds(subjects, meta, n_folds, n_val, seed, use_sex):
    subjects = np.array(sorted(subjects))
    rng = np.random.default_rng(seed)

    if meta:
        missing = [s for s in subjects if s not in meta]
        if missing:
            sys.exit(f"no metadata for subjects {missing}")
        y = np.array([strat_label(*meta[s], use_sex) for s in subjects])
        # fall back to age-only if some age x sex cell is too small to spread over n_folds
        counts = np.bincount(y)
        if use_sex and counts[counts > 0].min() < n_folds:
            print("note: an age x sex group has fewer subjects than folds; stratifying by age only",
                  file=sys.stderr)
            y = np.array([strat_label(*meta[s], False) for s in subjects])
            use_sex = False
        make_folds.stratified_by = "age+sex" if use_sex else "age"
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(subjects, y, groups=subjects)
    else:
        y = None
        perm = rng.permutation(len(subjects))  # GroupKFold is deterministic; shuffle first
        split_iter = ((perm[tr], perm[te]) for tr, te in
                      GroupKFold(n_splits=n_folds).split(subjects[perm], groups=subjects[perm]))

    folds = []
    for k, (trval_idx, test_idx) in enumerate(split_iter):
        trval = subjects[trval_idx]
        # validation subjects: spread across strata when metadata is available
        if y is not None:
            order = rng.permutation(len(trval))
            by_stratum = defaultdict(list)
            for i in order:
                by_stratum[y[trval_idx[i]]].append(trval[i])
            val, strata = [], list(by_stratum.values())
            while len(val) < n_val:
                for lst in strata:
                    if lst and len(val) < n_val:
                        val.append(lst.pop())
        else:
            val = list(rng.choice(trval, size=n_val, replace=False))
        val = sorted(int(s) for s in val)
        train = sorted(int(s) for s in trval if s not in set(val))
        test = sorted(int(s) for s in subjects[test_idx])
        folds.append({"fold": k, "train": train, "val": val, "test": test})
    return folds


def check(folds, subjects):
    all_test = [s for f in folds for s in f["test"]]
    assert sorted(all_test) == sorted(subjects), "each subject must be tested exactly once"
    for f in folds:
        a, b, c = set(f["train"]), set(f["val"]), set(f["test"])
        assert not (a & b or a & c or b & c), f"overlap in fold {f['fold']}"
        assert a | b | c == set(subjects), f"fold {f['fold']} does not cover all subjects"


def main():
    project = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--data-dir", default=str(project / "sleep_edf_data" / "physionet-sleep-data"))
    src.add_argument("--subjects", action="store_true", help="use the official 78 SC subject IDs")
    meta_grp = ap.add_mutually_exclusive_group()
    meta_grp.add_argument("--meta", help="age/sex table (default: MNE's age_records.csv)")
    meta_grp.add_argument("--no-meta", action="store_true", help="no age/sex stratification")
    ap.add_argument("--out", default=str(project / "folds.json"))
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--n-val", type=int, default=None,
                    help="validation subjects per fold (default: ~10%% of subjects)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-sex", action="store_true", help="stratify by age only")
    args = ap.parse_args()

    if not args.subjects:
        files = scan_files(args.data_dir)
        if not files:
            sys.exit(f"no SC4ssN*-PSG.edf files found under {args.data_dir}")
        subjects = sorted(files)
        if len(subjects) != 78:
            print(f"warning: found {len(subjects)} subjects, expected 78", file=sys.stderr)
    else:
        files, subjects = None, OFFICIAL_SC_SUBJECTS

    if args.n_folds > len(subjects):
        sys.exit(f"--n-folds {args.n_folds} is larger than the {len(subjects)} subjects available")
    if args.n_val is None:
        args.n_val = max(1, round(0.1 * len(subjects)))

    meta_path = None if args.no_meta else (args.meta or find_mne_age_records())
    if meta_path is None and not args.no_meta:
        print("note: no age/sex table found (pass --meta); folds are not stratified", file=sys.stderr)
    meta = load_meta(meta_path) if meta_path else None
    folds = make_folds(subjects, meta, args.n_folds, args.n_val, args.seed, not args.no_sex)
    check(folds, subjects)

    if files:
        for f in folds:
            for part in ("train", "val", "test"):
                f[f"{part}_files"] = [fn for s in f[part] for fn in files[s]]

    out = {
        "dataset": "Sleep-EDF Expanded, Sleep Cassette",
        "protocol": f"{args.n_folds}-fold subject-wise CV, {args.n_val} val subjects per fold",
        "seed": args.seed,
        "metadata": str(meta_path) if meta_path else None,
        "stratified_by": getattr(make_folds, "stratified_by", None) if meta else None,
        "n_subjects": len(subjects),
        "folds": folds,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"wrote {args.out}: {len(subjects)} subjects, {args.n_folds} folds")
    for f in folds:
        line = f"  fold {f['fold']}: train {len(f['train']):2d}  val {len(f['val'])}  test {len(f['test'])}"
        if meta:
            ages = [meta[s][0] for s in f["test"]]
            line += f"  test age {np.mean(ages):5.1f} (range {min(ages):.0f}-{max(ages):.0f})"
        print(line)


if __name__ == "__main__":
    main()


# --- loading in a training script --------------------------------------------
# import json
# folds = json.load(open("folds.json"))["folds"]
# f = folds[k]
# train_files, val_files, test_files = f["train_files"], f["val_files"], f["test_files"]
