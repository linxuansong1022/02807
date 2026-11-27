"""Load B's verified splits and prepare weighted Last.fm tag sets."""

from __future__ import annotations

from collections import Counter
import csv
import gzip
from pathlib import Path
from typing import Sequence

import duckdb
import numpy as np

if __package__:
    from . import retrieval
else:
    import retrieval


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_THRESHOLDS = (0, 10, 20, 34, 50)


def connect_data(data_dir: Path) -> duckdb.DuckDBPyConnection:
    """Open A's content and B's split tables without reading future interactions."""
    scratch = data_dir / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect(config={
        "threads": 4, "memory_limit": "2GB", "temp_directory": str(scratch),
    })
    try:
        db.read_parquet(str(data_dir / "processed" / "tracks.parquet")).create_view("tracks")
        db.execute("""CREATE VIEW tag_rows AS
            SELECT track_id, unnest(tags) AS tag, unnest(tag_weights) AS weight FROM tracks""")
        for name in ("eligible_users", "train_interactions", "track_splits"):
            db.read_parquet(str(data_dir / "splits" / f"{name}.parquet")).create_view(name)
    except Exception:
        db.close()
        raise
    return db


def training_track_ids(db: duckdb.DuckDBPyConnection) -> list[str]:
    """Use B's played training tracks with positive raw tags."""
    return [track for (track,) in db.execute("""
        SELECT s.track_id FROM track_splits s JOIN tracks t USING (track_id)
        WHERE s.split = 'train' AND t.has_tags
        ORDER BY s.track_id
    """).fetchall()]


def query_track_ids(db: duckdb.DuckDBPyConnection, split: str) -> list[str]:
    return [track for (track,) in db.execute("""
        SELECT track_id FROM track_splits WHERE split = ? ORDER BY track_id
    """, [split]).fetchall()]


def read_tag_sets(
    db: duckdb.DuckDBPyConnection, *, min_weight: float,
    track_ids: Sequence[str] | None = None, excluded_tags: frozenset[str] = frozenset(),
) -> dict[str, set[str]]:
    filter_ids = "" if track_ids is None else "WHERE track_id IN (SELECT unnest(?::VARCHAR[]))"
    params = [] if track_ids is None else [list(track_ids)]
    cursor = db.execute(f"SELECT track_id, tag, weight FROM tag_rows {filter_ids}", params)

    def rows():
        while batch := cursor.fetchmany(65_536):
            yield from batch

    return retrieval.tag_sets_from_rows(rows(), min_weight=min_weight, excluded_tags=excluded_tags)


def fit_tag_sets(
    db: duckdb.DuckDBPyConnection, training_ids: Sequence[str], *, min_weight: float,
    max_tag_fraction: float | None = None,
) -> tuple[dict[str, set[str]], frozenset[str]]:
    """Apply the weight rule, fit frequent-tag exclusions on training, then transform all songs."""
    tag_sets = read_tag_sets(db, min_weight=min_weight)
    return apply_frequency_cap(tag_sets, training_ids, max_tag_fraction)


def apply_frequency_cap(weighted_sets, training_ids, max_fraction):
    """Fit on the original eligible training cohort, before dropping any empty rows."""
    if max_fraction is None:
        return weighted_sets, frozenset()
    excluded = retrieval.frequent_tags({t: weighted_sets[t] for t in training_ids}, max_fraction)
    return {t: tags - excluded for t, tags in weighted_sets.items()}, excluded


def write_tag_statistics(
    db: duckdb.DuckDBPyConnection, training_ids: Sequence[str], output: Path,
) -> list[dict[str, float | int]]:
    """Record distributions and tag prevalence on the unfiltered training-song cohort."""
    if not training_ids:
        raise ValueError("no eligible training tracks")
    training = "SELECT unnest(?::VARCHAR[]) AS track_id"
    counts = ", ".join(
        f"count(DISTINCT tag) FILTER (WHERE weight >= {threshold})"
        for threshold in WEIGHT_THRESHOLDS
    )
    lengths = np.array(db.execute(f"""
        SELECT {counts} FROM ({training}) train LEFT JOIN tag_rows USING(track_id)
        GROUP BY train.track_id
    """, [list(training_ids)]).fetchall())
    summary = []
    with (output / "tag_count_distribution.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("min_weight", "tag_count", "track_count"))
        for threshold, column in zip(WEIGHT_THRESHOLDS, lengths.T):
            p10, median, p90 = np.quantile(column, (0.1, 0.5, 0.9))
            empty = int(np.count_nonzero(column == 0))
            summary.append({
                "min_weight": threshold, "empty_tracks": empty, "empty_fraction": empty / len(column),
                "p10": float(p10), "median": float(median), "p90": float(p90),
            })
            writer.writerows((threshold, count, freq) for count, freq in sorted(Counter(column).items()))

    frequencies = ", ".join(
        f"count(DISTINCT track_id) FILTER (WHERE weight >= {threshold}) AS n{threshold}"
        for threshold in WEIGHT_THRESHOLDS
    )
    cursor = db.execute(f"""SELECT tag, {frequencies}
        FROM ({training}) train JOIN tag_rows USING(track_id)
        GROUP BY tag ORDER BY n0 DESC, tag""", [list(training_ids)])
    with gzip.open(output / "training_tag_frequency.csv.gz", "wt", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["tag"] + [column for threshold in WEIGHT_THRESHOLDS
                                     for column in (f"count_at_{threshold}", f"fraction_at_{threshold}")])
        while batch := cursor.fetchmany(65_536):
            writer.writerows(
                [tag] + [value for count in counts for value in (count, count / len(lengths))]
                for tag, *counts in batch
            )
    return summary
