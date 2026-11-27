"""Tune C's candidate pool and compare retrieval methods with B's evaluator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import defaultdict
from datetime import datetime, timezone
from uuid import uuid4
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix

if __package__:
    from . import benchmark_retrieval, retrieval, retrieval_data
    from .common.evaluate import RANK_CUTOFFS, evaluate
else:
    import benchmark_retrieval
    import retrieval
    import retrieval_data
    from evaluate import RANK_CUTOFFS, evaluate


TOP_N = 50
MIN_PLAY_COUNT = 2
MAX_TAG_FRACTIONS = (None, 0.50, 0.25)
MAX_RANK = max(RANK_CUTOFFS)
VALIDATION = "validation"


class ListenerPool:
    """Score users from B's training interactions and fill short rankings by activity."""

    def __init__(self, db: duckdb.DuckDBPyConnection, track_ids: list[str]):
        self.track_rows = {track: row for row, track in enumerate(track_ids)}
        self.user_ids = db.sql("SELECT user_id FROM eligible_users ORDER BY user_id").fetchnumpy()["user_id"]
        if len(self.user_ids) < MAX_RANK:
            raise ValueError(f"B's evaluator requires at least {MAX_RANK} eligible users")
        db.register("pool_tracks", pa.table({
            "track_id": track_ids, "track_row": np.arange(len(track_ids), dtype=np.int32),
        }))
        db.register("pool_users", pa.table({
            "user_id": self.user_ids.tolist(), "user_row": np.arange(len(self.user_ids), dtype=np.int32),
        }))
        try:
            edges = db.execute("""
                SELECT t.track_row, u.user_row, ln(1 + i.play_count) AS weight
                FROM train_interactions i
                JOIN pool_tracks t USING (track_id)
                JOIN pool_users u USING (user_id)
                WHERE i.play_count >= ?
            """, [MIN_PLAY_COUNT]).fetchnumpy()
            self.weights = csr_matrix(
                (edges["weight"], (edges["track_row"], edges["user_row"])),
                shape=(len(track_ids), len(self.user_ids)),
            )
            self.popular = db.execute("""
                SELECT u.user_row
                FROM train_interactions i JOIN pool_users u USING (user_id)
                GROUP BY u.user_row
                ORDER BY sum(i.play_count) DESC, u.user_row
                LIMIT ?
            """, [MAX_RANK]).fetchnumpy()["user_row"]
        finally:
            db.unregister("pool_tracks")
            db.unregister("pool_users")

    def scores(self, neighbors: retrieval.Neighbors) -> np.ndarray:
        """Return candidate scores in eligible-user ID order for downstream models."""
        rows = [(self.track_rows[track], similarity) for track, similarity in neighbors if similarity > 0]
        if not rows:
            return np.zeros(len(self.user_ids))
        indices, similarities = zip(*rows)
        query = csr_matrix(
            (similarities, indices, [0, len(rows)]), shape=(1, self.weights.shape[0]),
        )
        return (query @ self.weights).toarray().ravel()

    def rank(self, neighbors: retrieval.Neighbors) -> tuple[list[str], int]:
        scores = self.scores(neighbors)
        content_count = int(np.count_nonzero(scores))
        ranked = retrieval.top_score_rows(scores, min(content_count, MAX_RANK))
        if content_count < MAX_RANK:
            remaining = self.popular[~np.isin(self.popular, ranked, assume_unique=True)]
            ranked = np.concatenate((ranked, remaining[:MAX_RANK - content_count]))
        return self.user_ids[ranked].tolist(), content_count

    def write_predictions(
        self, neighbors: dict[str, retrieval.Neighbors], path: Path,
    ) -> dict[str, int]:
        users, sizes = [], {}
        for track, matches in neighbors.items():
            ranked, sizes[track] = self.rank(matches)
            users.extend(ranked)
        predictions = pa.table({
            "track_id": [track for track in neighbors for _ in range(MAX_RANK)],
            "user_id": users,
            "rank": np.tile(np.arange(1, MAX_RANK + 1, dtype=np.int32), len(neighbors)),
        })
        pq.write_table(predictions, path, compression="zstd")
        return sizes


def evaluate_predictions(pool, neighbors, path, split_dir):
    sizes = pool.write_predictions(neighbors, path)
    total, zero, metrics = evaluate(path, VALIDATION, split_dir)
    slots = sum(max(0, MAX_RANK - size) for size in sizes.values())
    return {
        "tracks": total, "zero_listener_tracks": zero,
        "fallback_tracks": sum(size < MAX_RANK for size in sizes.values()),
        "fallback_slots": slots, "fallback_fraction": slots / (total * MAX_RANK),
        **{f"recall_at_{k}": recall for k, recall, _ in metrics},
        **{f"ndcg_at_{k}": ndcg for k, _, ndcg in metrics},
    }, sizes


def fallback_group(raw_tags, kept_tags, positive_neighbors, candidates):
    if raw_tags and not kept_tags:
        return "a_emptied"
    if positive_neighbors < TOP_N:
        return "b_few_neighbors"
    if candidates < MAX_RANK:
        return "c_few_candidates"
    return "d_ok"


def diagnose(db, path, neighbors, sizes, raw, kept, exact_counts, split_dir):
    """Attribute the actual ranked predictions using B's per-song recall denominator."""
    hits = db.execute("""
        WITH labels AS (
            SELECT DISTINCT track_id, user_id FROM read_parquet(?) WHERE split = ?
        )
        SELECT l.track_id, count(p.user_id), count(*)
        FROM labels l LEFT JOIN read_parquet(?) p
          ON l.track_id = p.track_id AND l.user_id = p.user_id AND p.rank <= ?
        GROUP BY l.track_id
    """, [str(split_dir / "ground_truth.parquet"), VALIDATION, str(path), MAX_RANK]).fetchall()
    labels = {track: (hit, n) for track, hit, n in hits}
    details = []
    for track, matches in neighbors.items():
        count = sum(score > 0 for _, score in matches)
        size = sizes[track]
        hit, listeners = labels.get(track, (0, 0))
        details.append(dict(
            track_id=track, raw_tags=len(raw[track]), kept_tags=len(kept[track]),
            positive_neighbors=count, exact_positive_neighbors=exact_counts[track],
            candidate_users=size, fallback_slots=max(0, MAX_RANK - size),
            category=fallback_group(raw[track], kept[track], count, size),
            hits=hit, listeners=listeners, recall_at_1000=hit / listeners if listeners else None,
        ))
    return details


def summarize_groups(method, details):
    by_group = defaultdict(list)
    for row in details:
        by_group[row["category"]].append(row)
    summary = []
    for group in ("a_emptied", "b_few_neighbors", "c_few_candidates", "d_ok"):
        songs = by_group[group]
        slots = sum(s["fallback_slots"] for s in songs)
        recalls = [s["recall_at_1000"] for s in songs if s["listeners"]]
        summary.append(dict(
            method=method, category=group, tracks=len(songs),
            filled_tracks=sum(s["fallback_slots"] > 0 for s in songs),
            fallback_slots=slots, mean_fill=slots / len(songs) if songs else None,
            scored_tracks=len(recalls), recall_at_1000=np.mean(recalls).item() if recalls else None,
            filled_with_50_exact=sum(s["fallback_slots"] > 0 and s["exact_positive_neighbors"] == TOP_N
                                     for s in songs),
        ))
    return summary


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tune_labels(db, pool, training_ids, query_ids, output, split_dir):
    trials = []
    best = None
    with TemporaryDirectory(dir=output, prefix="tuning-") as tmp:
        tmp = Path(tmp)
        for min_weight in retrieval_data.WEIGHT_THRESHOLDS:
            raw = retrieval_data.read_tag_sets(db, min_weight=min_weight, track_ids=training_ids + query_ids)
            for max_fraction in MAX_TAG_FRACTIONS:
                kept, excluded = retrieval_data.apply_frequency_cap(raw, training_ids, max_fraction)
                training = {track: kept[track] for track in training_ids if kept[track]}
                index = retrieval.SparseJaccardIndex(training)
                neighbors = {track: index.search(kept[track], TOP_N) for track in query_ids}
                metrics, _ = evaluate_predictions(pool, neighbors, tmp / "trial.parquet", split_dir)
                trial = {
                    "min_weight": min_weight, "max_tag_fraction": max_fraction,
                    "excluded_tags": len(excluded), "indexed_tracks": len(training),
                    "empty_queries": sum(not kept[t] for t in query_ids), **metrics,
                }
                trials.append(trial)
                key = (trial["recall_at_100"], trial["ndcg_at_100"])
                if best is None or key > (best["recall_at_100"], best["ndcg_at_100"]):
                    best = trial
                print(f"Weight {min_weight}, cap {max_fraction}: Recall@100 {key[0]:.6f}", flush=True)
    write_csv(output / "validation_tag_tuning.csv", trials)
    (output / "tag_rule.json").write_text(json.dumps(best, indent=2) + "\n")
    return best, trials


def benchmark_and_compare(db, pool, training_ids, query_ids, output, split_dir, targets, rule):
    kept, excluded = retrieval_data.fit_tag_sets(
        db, training_ids, min_weight=rule["min_weight"], max_tag_fraction=rule["max_tag_fraction"],
    )
    raw = retrieval_data.read_tag_sets(db, min_weight=0, track_ids=query_ids)
    run = benchmark_retrieval.write_benchmarks(
        kept, training_ids, query_ids, TOP_N, output,
        min_weight=rule["min_weight"], max_tag_fraction=rule["max_tag_fraction"],
        target_probability=targets[0], target_recall=targets[1],
    )
    training = {t: kept[t] for t in training_ids if kept[t]}
    methods = {"sparse_exact": None, "lsh_32x4": (32, 4)}
    selected = run["fastest_meeting_targets"]
    if selected:
        b, r = selected["bands"], selected["rows"]
        methods[f"lsh_{b}x{r}"] = (b, r)
    baseline, _ = evaluate_predictions(pool, dict.fromkeys(query_ids, []),
                                        output / "validation_popularity.parquet", split_dir)
    results = [{"method": "popularity", **baseline}]
    groups, per_track = [], []
    for method, config in methods.items():
        index = retrieval.MinHashLSH(training, *config) if config else retrieval.SparseJaccardIndex(training)
        neighbors = {t: index.search(kept[t], TOP_N) for t in query_ids}
        if config is None:
            exact_counts = {t: len(matches) for t, matches in neighbors.items()}
        filename = "exact" if config is None else method
        path = output / f"validation_{filename}.parquet"
        metrics, sizes = evaluate_predictions(pool, neighbors, path, split_dir)
        if config is None and not math.isclose(metrics["recall_at_100"], rule["recall_at_100"]):
            raise ValueError("Final exact predictions disagree with tag selection")
        detail = diagnose(db, path, neighbors, sizes, raw, kept, exact_counts, split_dir)
        summary = summarize_groups(method, detail)
        if (sum(g["tracks"] for g in summary) != metrics["tracks"]
                or sum(g["fallback_slots"] for g in summary) != metrics["fallback_slots"]
                or sum(g["filled_tracks"] for g in summary) != metrics["fallback_tracks"]
                or not math.isclose(np.mean([s["recall_at_1000"] for s in detail if s["listeners"]]),
                                    metrics["recall_at_1000"], abs_tol=1e-12)):
            raise ValueError(f"Group totals disagree with the official predictions for {method}")
        results.append({"method": method, **metrics})
        groups.extend(summary)
        per_track.extend({"method": method, **s} for s in detail)
        del index
    write_csv(output / "validation_metrics.csv", results)
    write_csv(output / "fallback_groups.csv", groups)
    pq.write_table(pa.Table.from_pylist(per_track), output / "fallback_by_track.parquet")
    run.update(indexed_training_tracks=len(training), emptied_training_tracks=len(training_ids)-len(training),
               excluded_tags=sorted(excluded), min_train_plays=MIN_PLAY_COUNT,
               no_positive_queries=sum(n == 0 for n in exact_counts.values()),
               short_positive_queries=sum(n < TOP_N for n in exact_counts.values()))
    (output / "run_metadata.json").write_text(json.dumps(run, indent=2) + "\n")
    return run, results, groups


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=retrieval_data.ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=retrieval_data.ROOT / "reports/retrieval")
    parser.add_argument("--stage", choices=("all", "validation", "benchmark"), default="all")
    parser.add_argument("--target-probability", type=float, default=0.90)
    parser.add_argument("--target-recall", type=float, default=0.90)
    args = parser.parse_args()
    targets = args.target_probability, args.target_recall
    if not all(0 <= target <= 1 for target in targets):
        parser.error("probability and recall must be in [0, 1]")

    parent = args.output_dir / "runs"
    parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    split_dir = args.data_dir / "splits"
    with TemporaryDirectory(prefix=".pending-", dir=parent) as temp:
        output = Path(temp) / run_id
        output.mkdir()
        with retrieval_data.connect_data(args.data_dir) as db:
            training_ids = retrieval_data.training_track_ids(db)
            query_ids = retrieval_data.query_track_ids(db, VALIDATION)
            if not training_ids or not query_ids:
                raise ValueError("B's data must include training and validation tracks")
            print(f"{len(training_ids):,} tagged training tracks and {len(query_ids):,} validation tracks", flush=True)
            pool = ListenerPool(db, training_ids)
            stats = retrieval_data.write_tag_statistics(db, training_ids, output)
            (output / "tag_statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
            rule, trials = tune_labels(db, pool, training_ids, query_ids, output, split_dir)
            if args.stage != "validation":
                run, metrics, groups = benchmark_and_compare(
                    db, pool, training_ids, query_ids, output, split_dir, targets, rule,
                )
                with (output / "benchmark_retrieval.csv").open() as f:
                    benchmark = list(csv.DictReader(f))
                expected = 4 * (1 + sum(len(benchmark_retrieval.band_grid(h)) for h in benchmark_retrieval.HASH_COUNTS))
                if len(benchmark) != expected or len(trials) != len(MAX_TAG_FRACTIONS) * len(retrieval_data.WEIGHT_THRESHOLDS):
                    raise ValueError("Incomplete benchmark or tag grid")
                snapshot = dict(run_id=run_id, split=json.loads((split_dir / "split_config.json").read_text()),
                                run=run, tag_statistics=stats, tag_trials=trials, metrics=metrics,
                                fallback_groups=groups, benchmark=benchmark)
                (output / "snapshot.json").write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
        inputs = [args.data_dir / "processed/tracks.parquet",
                  *(split_dir / name for name in ("eligible_users.parquet", "train_interactions.parquet",
                    "track_splits.parquet", "ground_truth.parquet", "split_config.json"))]
        manifest = dict(run_id=run_id, scope="tuning" if args.stage == "validation" else "full_validation",
                        inputs={str(p.relative_to(args.data_dir)): digest(p) for p in inputs},
                        files={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()})
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        output.rename(parent / run_id)
    print(f"Saved {parent / run_id}", flush=True)


if __name__ == "__main__":
    main()
