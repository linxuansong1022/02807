# Music cold-start recommendation

This repository prepares Music4All-Onion listening events and track metadata for a study of recommending tracks with no training interactions. It includes data download, preprocessing, and optional checks of candidate time windows.

## Run preprocessing

This project uses `uv` to manage its Python environment and dependencies. Install `uv` before running the commands below. `uv sync --locked` uses Python 3.14 and downloads it if needed. [Official `uv` installation guide](https://docs.astral.sh/uv/getting-started/installation/)

Use macOS or Linux with `curl`. Allow at least 8 GB of free disk space for the roughly 2.27 GB download, Parquet files, and temporary data. Run the commands from the repository root.

```bash
uv sync --locked
uv run python scripts/download_data.py
uv run python scripts/prepare_data.py
```

The downloader fetches the listening log, tag dictionary, tag and genre TF-IDF files, and dataset README. It verifies the published MD5 checksums and records them in `data/raw/download_manifest.json`. It reuses files that pass verification. [Music4All-Onion v2 on Zenodo](https://zenodo.org/records/15394646)

Preprocessing streams the listening log into Parquet, creates an analysis log starting in 2005, normalizes tags, and joins track metadata. It preserves repeated plays and keeps timestamps in the source's timezone-naive format.

## Check candidate time windows

```bash
uv run python scripts/inspect_splits.py
```

Run this after preprocessing to count eligible users and new tracks for candidate dates. It writes feasibility statistics and a data handoff without creating train, validation, or test splits.

## Open the notebook

After preprocessing and checking candidate time windows, launch JupyterLab with the `notebook` dependency group and open the [data preparation notebook](notebooks/01_data_preparation.ipynb).

```bash
uv run --locked --group notebook jupyter lab
```

## Generated outputs

| Path | Contents |
| --- | --- |
| `data/raw/` | Verified source files and download manifest |
| `data/processed/events_source.parquet` | All source events |
| `data/processed/events.parquet` | Analysis events with string user and track IDs |
| `data/processed/tracks.parquet` | Tags, weights, genres, and each track's earliest play in the full analysis log |
| `data/processed/` | Additional feature, first-play, and descriptive statistics tables |
| `data/samples/dev/`, `data/samples/standard/` | Nested samples of up to 2,000 and 20,000 users with complete analysis histories |
| `reports/` | Source manifest, `data_summary.json`, `processing_complete.txt`, and figures. The optional check adds `split_feasibility.csv`, its settings, and `data_handoff.md`. |

Generated data and reports are Git-ignored. An existing complete source conversion can be reused on reruns. To write a fresh copy without overwriting default outputs, pass `--output-dir data/reproduced` to `prepare_data.py`. The optional split check reads the default output paths.

Track `first_seen` comes from the full analysis log, so it stays consistent across samples. Compute training activity from events before the chosen cutoff. The provided full-log statistics describe the whole dataset.

## Test

```bash
uv run python -m unittest discover -s tests -v
```

The tests use synthetic data and run without the full download. `pyproject.toml` and `uv.lock` define the environment, and `.python-version` selects Python 3.14.
