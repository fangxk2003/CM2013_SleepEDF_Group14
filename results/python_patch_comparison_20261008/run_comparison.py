"""Repeat P1 under two interpreters while sharing the exact package files."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-314", required=True, type=Path)
    parser.add_argument("--python-317", required=True, type=Path)
    parser.add_argument("--resume", action="store_true", help="Reuse completed captures.")
    args = parser.parse_args()
    task = Path(__file__).resolve().parent
    root = task.parents[1]
    site = root / ".venv/lib/python3.11/site-packages"
    env = os.environ.copy()
    for name in (
        "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    env["MPLCONFIGDIR"] = str(task / "runtime-cache/matplotlib")
    env["JOBLIB_TEMP_FOLDER"] = "/private/tmp/sleepedf-python-patch-20261008/joblib"
    plan = [
        ("existing_3.11.14", root / ".venv/bin/python", "3.11.14"),
        ("compiled_3.11.14_repeat1", args.python_314, "3.11.14"),
        ("compiled_3.11.17_repeat1", args.python_317, "3.11.17"),
        ("compiled_3.11.14_repeat2", args.python_314, "3.11.14"),
        ("compiled_3.11.17_repeat2", args.python_317, "3.11.17"),
    ]
    (task / "logs").mkdir(exist_ok=True)
    (task / "run_plan.json").write_text(json.dumps({
        "plan": [{"name": name, "executable": str(executable), "python": version}
                 for name, executable, version in plan],
        "shared_site_packages": str(site),
        "thread_settings": {name: value for name, value in env.items()
                            if name.endswith("_NUM_THREADS") or name == "VECLIB_MAXIMUM_THREADS"},
    }, indent=2) + "\n")
    for name, executable, version in plan:
        output = task / "runs" / name
        if args.resume and (output / "environment.json").is_file():
            print(f"Using completed capture: {name}", flush=True)
            continue
        if output.exists():
            raise RuntimeError(f"Capture exists but cannot be reused: {output}")
        cmd = [
            str(executable), "-I", "-S", "-B", str(task / "capture_run.py"),
            "--project-root", str(root), "--site-packages", str(site),
            "--output-dir", str(output), "--expected-python", version,
        ]
        print(f"Running {name}...", flush=True)
        log_path = task / "logs" / f"{name}.log"
        with log_path.open("w") as log:
            completed = subprocess.run(cmd, cwd=root, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"Run failed ({completed.returncode}); inspect {log_path}")
        report = json.loads((output / "report.json").read_text())
        print(f"Completed {name}: {report['summary']}", flush=True)


if __name__ == "__main__":
    main()
