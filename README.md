# Music cold-start recommendation

This project ranks potential listeners for cold-start tracks using Music4All-Onion listening histories and track tags. The pipeline prepares and verifies the data, builds a fixed temporal split, and uses a shared evaluator to compare random, most-active, and tag-profile baselines with ALS + Ridge.

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
│   ├── 02_data_splitting.ipynb      # Temporal split and sparse feature diagnostics
│   ├── 03_baselines.ipynb           # Baseline costs, coverage, and validation results
│   └── 04_als_ridge.ipynb           # ALS factors, Ridge diagnostics, and validation results
├── scripts/
│   ├── data/
│   │   ├── download_data.py         # Download and verify source files
│   │   ├── prepare_data.py          # Prepare analysis tables and user samples
│   │   ├── inspect_splits.py        # Summarize candidate time windows
│   │   ├── build_splits.py          # Generate the fixed temporal split
│   │   └── verify_data.py           # Check datasets against the reference
│   ├── common/
│   │   ├── evaluate.py              # Score ranked user recommendations
│   │   ├── features.py              # Build aligned sparse interactions and tag features
│   │   └── scoring.py               # Rank users deterministically and export predictions
│   └── models/
│       ├── als.py                   # Fit CPU implicit ALS and export ID-keyed factors
│       ├── baselines.py             # Generate shared baseline recommendations
│       └── ridge.py                 # Fit the collaborative mapping and rank cold-start tracks
├── tests/
│   └── test_prepare_data.py         # Verify preprocessing on synthetic data
├── validation/
│   └── data_reference.json          # Expected schemas, counts, and fingerprints
├── data/                            # Local datasets
│   ├── raw/
│   ├── processed/
│   ├── samples/
│   ├── splits/
│   ├── models/                      # Local model artifacts
│   └── predictions/                 # Local ranked recommendations
└── reports/                         # Local summaries and figures
```

## Run preprocessing

This project uses `uv` to manage its Python environment and dependencies. Install `uv` before running the commands below. `uv sync --locked` uses Python 3.14 and downloads it if needed. [Official `uv` installation guide](https://docs.astral.sh/uv/getting-started/installation/)

Use macOS or Linux with `curl`. Allow at least 8 GB of free disk space for the roughly 2.27 GB download, Parquet files, and temporary data. Run the commands from the repository root.

```bash
uv sync --locked
uv run --locked python -m scripts.data.download_data
uv run --locked python -m scripts.data.prepare_data
```

The downloader fetches the listening log, tag dictionary, tag and genre TF-IDF files, and dataset README. It verifies the published MD5 checksums and records them in `data/raw/download_manifest.json`. It reuses files that pass verification. [Music4All-Onion v2 on Zenodo](https://zenodo.org/records/15394646)

Preprocessing streams the listening log into Parquet, creates an analysis log starting in 2005, normalizes tags, and joins track metadata. It preserves repeated plays and keeps timestamps in the source's timezone-naive format.

## Verify the data pipeline

After preprocessing and [building the fixed split](#build-the-fixed-temporal-split), check the analysis tables, both user samples, and the five split files against the [committed reference](validation/data_reference.json).

```bash
uv run --locked python -m scripts.data.verify_data --check
```

The read-only command prints `PASS`, `FAIL`, or `MISSING` per file and exits with a failure status for missing files or mismatches. It compares Parquet row counts, schemas, and order-independent fingerprints, and checks the split configuration, training cutoff, and ground-truth timing. Use the locked environment because DuckDB's hash values can change across versions. [DuckDB hash documentation](https://duckdb.org/docs/2.0/sql/functions/utility#hashvalue)

After an intentional data change, explicitly regenerate the reference from a complete local build and review the resulting Git diff.

```bash
uv run --locked python -m scripts.data.verify_data --write-reference
```

## Check candidate time windows

```bash
uv run --locked python -m scripts.data.inspect_splits
```

Run this after preprocessing to count eligible users and new tracks for candidate dates. It writes feasibility statistics and a data handoff without creating train, validation, or test splits.

## Build the fixed temporal split

After preprocessing the full dataset, run the builder from the repository root.

```bash
uv run --locked python -m scripts.data.build_splits
```

Read the following local files from `data/splits/` after the command succeeds. Join on the original string IDs and use `data/processed/tracks.parquet` for content features.

| File | Use |
| --- | --- |
| `eligible_users.parquet` | Candidate user IDs for training and evaluation |
| `train_interactions.parquet` | Play counts per eligible user and track before 2018-01-01, including tracks without tags |
| `track_splits.parquet` | Train, validation, and test track IDs with global `first_seen`, including evaluation tracks without eligible listeners |
| `ground_truth.parquet` | Distinct eligible listeners during each evaluation track's first 30 days. Use these rows as evaluation labels. |
| `split_config.json` | Arrival windows, eligibility rules, and output counts |

## Build sparse model inputs

After verifying the data, import the shared loaders from the repository root. The available tag weightings are `binary`, `weighted`, and `tfidf`.

```python
from scripts.common.features import load_interactions, load_track_features

R, user_ids, train_ids = load_interactions()
X, track_ids, vocabulary = load_track_features("tfidf", training_ids=train_ids)
```

`R` contains raw aggregated play counts in float32 CSR format. Its rows are sorted eligible user IDs and its columns are sorted training track IDs. Each `X[split]` is a float32 CSR matrix with sorted track rows and the same sorted training tag columns. Untagged tracks and tracks with only unseen tags keep zero rows. Use these IDs to align the ALS track factors with the rows selected for Ridge.

See the Shared Sparse Features section for weighting definitions, matrix sizes, construction times, and tag coverage. [Temporal split notebook](notebooks/02_data_splitting.ipynb)

## Run the shared baselines

Build and verify the fixed split, then generate predictions from the repository root.

```bash
uv run --locked python -m scripts.models.baselines --split validation
uv run --locked python -m scripts.common.evaluate --split validation --predictions data/predictions/baseline_tag_profile_validation.parquet
```

The same evaluation command accepts the random and most-active prediction files. The generator writes `baseline_{random,most_active,tag_profile}_{split}.parquet` under the Git-ignored `data/predictions/` directory. Each track has 1,000 distinct eligible users at ranks 1–1,000. Most-active saves pre-cutoff play totals as scores, tag profile saves TF-IDF cosine similarities, and random uses seed 42 with a separate stream for each split. Activity and similarity ties use ascending user ID.

Use `--split test` to generate test predictions for the final evaluation phase. Use `--batch-size` to adjust the number of user profiles and scores held at once. The notebook runs the three baselines on validation and compares the metrics, coverage, and computational cost. [Baselines notebook](notebooks/03_baselines.ipynb)

## Train implicit ALS

Build and verify the fixed split, then run CPU training from the repository root. The defaults are 64 factors, regularization 0.1, confidence scale 10, 15 iterations, and seed 42. Use `--help` for parameter overrides.

```bash
uv run --locked python -m scripts.models.als
```

Training uses raw pre-cutoff play counts for every eligible user and training track. It stores confidence `1 + 10 × play_count` for observed pairs, with unit confidence for missing pairs. The library receives these weights with its own `alpha` set to 1.

The command writes three Git-ignored files under `data/models/`. Both Parquet files preserve the original string IDs in the order returned by the interaction loader. Join candidates to the user factors on `user_id`.

| File | Schema or contents |
| --- | --- |
| `als_user_factors.parquet` | `user_id` as string, `factors` as a fixed-size float32 list |
| `als_track_factors.parquet` | `track_id` as string, `factors` as a fixed-size float32 list |
| `als_config.json` | Training parameters, fit time, verified training split, and required SHA-256 hashes for both factor files |

Each list contains 64 values with the defaults. The notebook trains ALS only when all three artifacts are absent. An existing export must be complete and pass both SHA-256 checks before the notebook displays its fit time, storage, and factor norms. [ALS and Ridge notebook](notebooks/04_als_ridge.ipynb)

## Run ALS + Ridge

Use the ALS export and the verified split. Ridge verifies both ALS factor files against `factor_sha256` in `als_config.json`. Run the ALS command above to regenerate artifacts if either hash is missing or mismatched. Ridge fits the nonempty training rows with TF-IDF, regularization 1.0, and damped LSQR, then ranks the full eligible-user pool by raw dot product. Run these commands from the repository root.

```bash
uv run --locked python -m scripts.models.ridge --split validation
uv run --locked python -m scripts.common.evaluate --split validation --predictions data/predictions/ridge_validation.parquet
```

Use `--help` for solver, feature weighting, and batch-size options. The notebook compares Ridge to the three existing validation baseline files. Generate any missing baseline files with the baseline command above. For the final evaluation phase, `--split test` applies the saved mapping and writes the equivalent test artifacts.

| Git-ignored artifact | Schema or contents |
| --- | --- |
| `data/models/ridge_weights.npy` | Float32 matrix with feature columns as rows and ALS dimensions as columns |
| `data/models/ridge_vocabulary.json` | Ordered list of feature names matching the weight rows |
| `data/models/ridge_config.json` | Split, settings, per-dimension LSQR diagnostics, run timings, and input/output hashes |
| `data/models/ridge_validation_factors.parquet` | String `track_id` and a fixed-size float32 `factors` list |
| `data/predictions/ridge_validation.parquet` | String `track_id`, string `user_id`, integer `rank`, and float32 dot-product `score` |

To reconstruct B-path scores for candidate pairs, join `user_id` to `als_user_factors.parquet` and `track_id` to `ridge_validation_factors.parquet`, then take the dot product of the two stored vectors. This works for any eligible user. Compare the files' SHA-256 digests with `als_sha256` and `ridge_sha256` in `ridge_config.json` before combining them.

## Evaluate recommendations

Provide a Parquet file for the selected split with the original string `track_id` and `user_id` columns and an integer `rank` column. Each track in that split needs 1,000 distinct eligible users at ranks 1–1,000, where rank 1 is best. Additional ranks must also have unique users and ranks per track. Optional scores are ignored.

```bash
uv run --locked python -m scripts.common.evaluate --predictions path/to/predictions.parquet --split validation
```

Use `--split test` for test predictions. The evaluator reports macro Recall and binary NDCG at 100, 500, and 1,000. Recall uses all observed listeners for each track as its denominator. Tracks with zero observed listeners are counted separately, and each remaining track contributes equally to the average.

Python callers can import `evaluate` from `scripts.common.evaluate` and pass the prediction path and split.

## Open the notebooks

After preprocessing, launch JupyterLab with the `notebook` dependency group.

```bash
uv run --locked --group notebook jupyter lab
```

- Inspect the prepared data, samples, and figures in the [data preparation notebook](notebooks/01_data_preparation.ipynb).
- Compare training cutoffs and evaluation windows, then examine the matrices built from the fixed split. [Temporal split notebook](notebooks/02_data_splitting.ipynb)
- Compare the three shared baselines on validation. [Baselines notebook](notebooks/03_baselines.ipynb)
- Inspect ALS factors, fit Ridge, and compare the validation metrics. [ALS and Ridge notebook](notebooks/04_als_ridge.ipynb)

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

Generated data and reports are Git-ignored. An existing complete source conversion can be reused on reruns. To write a fresh copy without overwriting default outputs, pass `--output-dir data/reproduced` to `python -m scripts.data.prepare_data`. The optional split check reads the default output paths.

Track `first_seen` comes from the full analysis log, so it stays consistent across samples. Compute training activity from events before the chosen cutoff. The provided full-log statistics describe the whole dataset.

## Test

```bash
uv run --locked python -m unittest discover -s tests -v
```

The tests use synthetic data and run without the full download. `pyproject.toml` and `uv.lock` define the environment, and `.python-version` selects Python 3.14.
