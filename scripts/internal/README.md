# Internal support

The everyday commands live directly in `scripts/`. This folder holds shared
support code: `_bootstrap.py` resolves the project source directory, and
`run_baseline.py` provides the loading, evaluation, and provenance functions used
by `scripts/run_experiment.py`.

Use `scripts/run_experiment.py` for new experiments. The original baseline runner
remains directly callable here for reproducing historical commands:

```bash
uv run --locked python -B scripts/internal/run_baseline.py --n-subjects 3 --nights 1 2
```

Its arguments, cached-data location, and output naming are preserved. It uses
cached EDFs and creates a new result directory; it does not download data.
