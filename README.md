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
│   ├── 02_data_splitting.ipynb      # Candidate cutoffs and split design
│   └── 03_retrieval.ipynb           # Candidate-pool validation and retrieval benchmark
├── scripts/
│   ├── download_data.py             # Download and verify source files
│   ├── prepare_data.py              # Prepare analysis tables and user samples
│   ├── inspect_splits.py            # Summarize candidate time windows
│   ├── build_splits.py              # Generate the fixed temporal split
│   ├── verify_data.py               # Check datasets against the reference
│   ├── evaluate.py                  # Score ranked user recommendations
│   ├── evaluate_retrieval.py        # Tune tags and evaluate listener rankings
│   ├── retrieval_data.py            # Read content and split tables
│   ├── retrieval.py                 # Exact Jaccard and MinHash LSH
│   ├── benchmark_retrieval.py       # Measure neighbor recall, speed, and memory
│   └── plot_lsh_s_curve.py          # Plot candidate probability and scaling
├── tests/
│   ├── test_prepare_data.py         # Verify preprocessing on a small fixture
│   └── test_retrieval.py            # Check retrieval on the prepared dataset
├── validation/
│   ├── data_reference.json          # Expected schemas, counts, and fingerprints
│   └── retrieval_snapshot.json      # Fixed results for the retrieval notebook
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

## Validate the candidate pool and benchmark retrieval

The [retrieval notebook](notebooks/03_retrieval.ipynb) reads one fixed [validation snapshot](validation/retrieval_snapshot.json). It includes all 15 tag trials, all 72 retrieval benchmarks, listener metrics and fallback groups. The notebook checks the snapshot digest and can run without the local data or reports.

To repeat the experiment after verifying the prepared data and fixed split, run:

```bash
uv run --locked python scripts/evaluate_retrieval.py
```

Each completed experiment is published in a new `reports/retrieval/runs/<run-id>/` directory. The manifest identifies its scope and contains SHA-256 digests of its inputs and artifacts. A tuning-only run (`--stage validation`) has no final snapshot. The default and `--stage benchmark` run tuning and benchmarking together. The standalone benchmarking script uses a separate output directory. Fresh runs measure timing and peak process memory on the current machine, so these values and the fastest qualifying configuration can vary. The notebook keeps its fixed snapshot until the snapshot file and its pinned SHA-256 are deliberately updated together.

The tag grid uses five minimum weights and three frequency caps. Fit tag prevalence on all eligible, originally tagged training songs, including songs later left empty by filtering. Drop empty songs from the index. Select by validation listener Recall@100 from exact retrieval, breaking ties with NDCG@100.

Retrieve up to 50 songs with positive Jaccard similarity, ordered by descending similarity and ascending track ID. Empty queries return no neighbors. For each eligible listener, sum similarity times `log1p(play_count)` over neighbors played at least twice before the cutoff. Fill ranks through 1,000 with unused eligible users in descending training-play order, breaking ties by user ID. Downstream systems can access the scores in eligible-user ID order via `ListenerPool.scores()`.

Neighbor Recall@50 uses each query's exact positive top list as its denominator and reports queries with no positive neighbors separately. Missing neighbor positions are NaN in `neighbor_jaccard.npy`. Band selection uses candidate probability at the 10th percentile of positive neighbor scores and measured macro neighbor recall, both with example targets of 90%. The 88,092-song stress cohort excludes empty songs after filtering and is used for retrieval only.

`fallback_by_track.parquet` records raw and kept tag counts, positive neighbor counts for the method and exact search, uncapped candidate users, fill slots, listener hits and recall. `fallback_groups.csv` separates songs with emptied tags, fewer than 50 positive neighbors, 50 neighbors but fewer than 1,000 candidate users, and sufficient neighbors and candidates. Songs with fewer than 50 neighbors can still have enough candidates. The table reports both group size and the number actually requiring fill.

For downstream clustering, align `pool.scores(neighbors)` with `pool.user_ids`. In this experiment the content pool is the first `min(content_count, 1000)` users from `ranked, content_count = pool.rank(neighbors)`. Any remaining ranks are activity fills with zero content score. The fixed results evaluate the candidate pool on validation. The full-system comparison requires D's clustering and fusion.

## Open the notebooks

After preprocessing, launch JupyterLab with the `notebook` dependency group.

```bash
uv run --locked --group notebook jupyter lab
```

- Inspect the prepared data, samples, and figures in the [data preparation notebook](notebooks/01_data_preparation.ipynb).
- Compare training cutoffs and evaluation windows in the [temporal split notebook](notebooks/02_data_splitting.ipynb).
- Review candidate-pool validation and the retrieval figures in the [retrieval notebook](notebooks/03_retrieval.ipynb).

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

The preprocessing test runs without the full download. The retrieval tests use the prepared Music4All-Onion data and B's fixed split. `pyproject.toml` and `uv.lock` define the environment, and `.python-version` selects Python 3.14.
