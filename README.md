# Music cold-start recommendation

This repository prepares Music4All-Onion listening events and track metadata for a study of recommending tracks with no training interactions. It includes data download, preprocessing, and optional checks of candidate time windows.

## Folder structure

```text
.
├── .gitignore                       # Local files excluded from Git
├── .python-version                  # Python version selected by uv
├── README.md                        # Setup and workflow instructions
├── pyproject.toml                   # Project dependencies
├── uv.lock                          # Locked dependency versions
├── docs/
│   └── project.pdf                  # Course project specification
├── notebooks/
│   ├── 01_data_preparation.ipynb    # Prepared data inspection and figures
│   └── 02_data_splitting.ipynb      # Candidate cutoffs and split design
├── scripts/
│   ├── download_data.py             # Download and verify source files
│   ├── prepare_data.py              # Prepare analysis tables and user samples
│   ├── inspect_splits.py            # Summarize candidate time windows
│   ├── build_splits.py              # Generate the fixed temporal split
│   ├── verify_data.py               # Check datasets against the reference
│   └── evaluate.py                  # Score ranked user recommendations
├── tests/
│   └── test_prepare_data.py         # Verify preprocessing on synthetic data
├── validation/
│   └── data_reference.json          # Expected schemas, counts, and fingerprints
├── data/                            # Local datasets
│   ├── raw/
│   ├── processed/
│   ├── samples/
│   └── splits/
└── reports/                         # Local summaries and figures
```

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

## Verify the data pipeline

After preprocessing and [building the fixed split](#build-the-fixed-temporal-split), check the analysis tables, both user samples, and the five split files against the [committed reference](validation/data_reference.json).

```bash
uv run --locked python scripts/verify_data.py --check
```

The read-only command prints `PASS`, `FAIL`, or `MISSING` per file and exits with a failure status for missing files or mismatches. It compares Parquet row counts, schemas, and order-independent fingerprints, and checks the split configuration, training cutoff, and ground-truth timing. Use the locked environment because DuckDB's hash values can change across versions. [DuckDB hash documentation](https://duckdb.org/docs/2.0/sql/functions/utility#hashvalue)

After an intentional data change, explicitly regenerate the reference from a complete local build and review the resulting Git diff.

```bash
uv run --locked python scripts/verify_data.py --write-reference
```

## Check candidate time windows

```bash
uv run python scripts/inspect_splits.py
```

Run this after preprocessing to count eligible users and new tracks for candidate dates. It writes feasibility statistics and a data handoff without creating train, validation, or test splits.

## Build the fixed temporal split

After preprocessing the full dataset, run the builder from the repository root.

```bash
uv run --locked python scripts/build_splits.py
```

Read the following local files from `data/splits/` after the command succeeds. Join on the original string IDs and use `data/processed/tracks.parquet` for content features.

| File | Use |
| --- | --- |
| `eligible_users.parquet` | Candidate user IDs for training and evaluation |
| `train_interactions.parquet` | Play counts per eligible user and track before 2018-01-01, including tracks without tags |
| `track_splits.parquet` | Train, validation, and test track IDs with global `first_seen`, including evaluation tracks without eligible listeners |
| `ground_truth.parquet` | Distinct eligible listeners during each evaluation track's first 30 days. Use these rows as evaluation labels. |
| `split_config.json` | Arrival windows, eligibility rules, and output counts |

## Evaluate recommendations

Provide a Parquet file for the selected split with the original string `track_id` and `user_id` columns and an integer `rank` column. Each track in that split needs 1,000 distinct eligible users at ranks 1–1,000, where rank 1 is best. Additional ranks must also have unique users and ranks per track. Optional scores are ignored.

```bash
uv run --locked python scripts/evaluate.py --predictions path/to/predictions.parquet --split validation
```

Use `--split test` for test predictions. The evaluator reports macro Recall and binary NDCG at 100, 500, and 1,000. Recall uses all observed listeners for each track as its denominator. Tracks with zero observed listeners are counted separately, and each remaining track contributes equally to the average.

Python callers can import `evaluate` from `scripts.evaluate` and pass the prediction path and split.

## Open the notebooks

After preprocessing, launch JupyterLab with the `notebook` dependency group.

```bash
uv run --locked --group notebook jupyter lab
```

- Inspect the prepared data, samples, and figures in the [data preparation notebook](notebooks/01_data_preparation.ipynb).
- Compare training cutoffs and evaluation windows in the [temporal split notebook](notebooks/02_data_splitting.ipynb).

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
