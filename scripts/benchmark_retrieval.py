"""Benchmark sparse exact Jaccard and MinHash banding on the fixed validation split."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing as mp
from pathlib import Path
import platform
import random
import resource
import sys
from time import perf_counter

import numpy as np
import scipy

if __package__:
    from . import retrieval, retrieval_data
else:
    import retrieval
    import retrieval_data


HASH_COUNTS = (128, 256)
SEED = 42


def band_grid(num_hashes: int) -> list[tuple[int, int]]:
    return [(bands, num_hashes // bands) for bands in range(1, num_hashes + 1)
            if num_hashes % bands == 0]


def measure_index(
    tracks: retrieval.TagSets, queries: retrieval.TagSets, top_n: int,
    config: tuple[int, int] | None, vocabulary: set[str] | None = None,
    exact_neighbors: list[retrieval.Neighbors] | None = None,
) -> tuple[dict, list[retrieval.Neighbors]]:
    """Run in a fresh process so the peak RSS belongs to this method and scope."""
    started = perf_counter()
    if config is None:
        index = retrieval.SparseJaccardIndex(tracks)
    else:
        index = retrieval.MinHashLSH(tracks, *config, seed=SEED, vocabulary=vocabulary)
    build_seconds = perf_counter() - started
    durations, candidate_counts, answers = [], [], []
    for track, tags in queries.items():
        started = perf_counter()
        if config is None:
            neighbors = index.search(tags, top_n, exclude_track_id=track)
            count = len(tracks) - (track in tracks)
        else:
            neighbors, count = index.search_with_count(tags, top_n, exclude_track_id=track)
        durations.append((perf_counter() - started) * 1000)
        answers.append(neighbors)
        candidate_counts.append(count)
    p50, p95 = np.percentile(durations, (50, 95))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    overlaps = []
    truth_lists = answers if exact_neighbors is None else exact_neighbors
    for expected, actual in zip(truth_lists, answers, strict=True):
        truth = {track for track, similarity in expected if similarity > 0}
        if truth:
            overlaps.append(len(truth.intersection(track for track, _ in actual)) / len(truth))
    return {
        "index_tracks": len(tracks), "method": "sparse_exact" if config is None else "lsh",
        "hashes": config[0] * config[1] if config else None,
        "bands": config[0] if config else None, "rows": config[1] if config else None,
        "build_seconds": build_seconds, "query_p50_ms": float(p50), "query_p95_ms": float(p95),
        "peak_rss_mib": peak / (2**20 if sys.platform == "darwin" else 2**10),
        "neighbor_recall": float(np.mean(overlaps)) if overlaps else None,
        "scored_queries": len(overlaps), "no_positive_queries": len(answers) - len(overlaps),
        "mean_candidates": float(np.mean(candidate_counts)),
    }, answers if config is None else []


def _exact_scores(neighbors: list[retrieval.Neighbors], top_n: int) -> np.ndarray:
    scores = np.full((len(neighbors), top_n), np.nan)
    for row, result in enumerate(neighbors):
        scores[row, :len(result)] = [similarity for _, similarity in result]
    return scores


def fastest_meeting_targets(
    rows: list[dict], *, target_probability: float, target_recall: float,
) -> dict | None:
    eligible = [
        row for row in rows
        if row["method"] == "lsh"
        and row["candidate_probability_at_p10"] >= target_probability
        and row["neighbor_recall"] >= target_recall
    ]
    return min(eligible, key=lambda row: (row["query_p50_ms"], row["query_p95_ms"],
                                         row["build_seconds"]), default=None)


def run_benchmarks(
    tag_sets: retrieval.TagSets, training_ids: list[str], queries: retrieval.TagSets,
    top_n: int, output: Path,
) -> tuple[list[dict], dict[str, dict]]:
    nested = sorted(track for track in training_ids if tag_sets[track])
    if not nested:
        raise ValueError("the selected tag filters left no training songs with usable tags")
    random.Random(SEED).shuffle(nested)
    configs = [config for hashes in HASH_COUNTS for config in band_grid(hashes)]
    results, quantiles = [], {}
    train_vocabulary = {tag for track in training_ids for tag in tag_sets[track]}
    scopes = [
        ("train_25", nested[:max(1, round(len(nested) * 0.25))]),
        ("train_50", nested[:max(1, round(len(nested) * 0.50))]),
        ("train_100", nested),
        ("all_tagged_retrieval_only", tag_sets),
    ]
    with (output / "benchmark_retrieval.csv").open("w", newline="") as file, \
            ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"),
                                max_tasks_per_child=1) as pool:
        writer = None
        for scope, ids in scopes:
            tracks = {track: tag_sets[track] for track in ids if tag_sets[track]}
            print(f"{scope}: {len(tracks):,} indexed songs, {len(queries):,} queries", flush=True)
            # Check the optimized baseline against the scan on real queries before timing.
            exact, truth = pool.submit(measure_index, tracks, queries, top_n, None).result()
            for (track, tags), observed in zip(list(queries.items())[:3], truth[:3], strict=True):
                expected = retrieval.exact_jaccard_scan(tracks, tags, top_n, exclude_track_id=track)
                if observed != expected:
                    raise AssertionError(f"Sparse and scan top-{top_n} differ for {track} in {scope}")
            scores = _exact_scores(truth, top_n)
            p10, p50, p90 = np.nanpercentile(scores, (10, 50, 90))
            quantiles[scope] = {"p10": float(p10), "p50": float(p50), "p90": float(p90)}
            if scope == "train_100":
                np.save(output / "neighbor_jaccard.npy", scores, allow_pickle=False)
            exact.update(scope=scope, catalog_tracks=len(ids), candidate_probability_at_p10=None)
            scope_rows = [exact]
            if writer is None:
                writer = csv.DictWriter(file, fieldnames=list(exact))
                writer.writeheader()
            writer.writerow(exact)
            file.flush()
            # In the all-tagged stress test, extra track tags use the unseen-tag hash path.
            vocabulary = train_vocabulary if scope == "all_tagged_retrieval_only" else None
            for bands, rows in configs:
                result, _ = pool.submit(
                    measure_index, tracks, queries, top_n, (bands, rows), vocabulary, truth,
                ).result()
                result.update(
                    scope=scope, catalog_tracks=len(ids),
                    candidate_probability_at_p10=float(retrieval.candidate_probability(p10, bands, rows)),
                )
                writer.writerow(result)
                file.flush()
                scope_rows.append(result)
                print(f"  {bands}×{rows}: Recall@{top_n} {result['neighbor_recall']:.3f}, "
                      f"p50 {result['query_p50_ms']:.3f} ms", flush=True)
            results.extend(scope_rows)
    return results, quantiles


def write_benchmarks(
    tag_sets, training_ids, query_ids, top_n, output, *,
    min_weight, max_tag_fraction, target_probability, target_recall,
):
    queries = {track: tag_sets[track] for track in query_ids}
    (output / "query_ids.json").write_text(json.dumps(query_ids, indent=2) + "\n", encoding="utf-8")
    rows, quantiles = run_benchmarks(tag_sets, training_ids, queries, top_n, output)
    train_rows = [row for row in rows if row["scope"] == "train_100"]
    selected = fastest_meeting_targets(
        train_rows, target_probability=target_probability, target_recall=target_recall,
    )
    metadata = {
        "split": "validation", "top_n": top_n, "seed": SEED,
        "min_weight": min_weight, "max_tag_fraction": max_tag_fraction,
        "training_tracks": len(training_ids), "queries": len(queries),
        "neighbor_jaccard": quantiles,
        "target_probability": target_probability, "target_recall": target_recall,
        "fastest_meeting_targets": selected,
        "environment": {"architecture": platform.machine(), "os": platform.system(),
                        "python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__},
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=retrieval_data.ROOT / "data")
    parser.add_argument("--output-dir", type=Path, help="New directory for a standalone benchmark")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--min-weight", type=float, default=0)
    parser.add_argument("--max-tag-fraction", type=float)
    parser.add_argument("--target-probability", type=float, default=0.9)
    parser.add_argument("--target-recall", type=float, default=0.9)
    args = parser.parse_args()
    if args.top_n < 1 or not all(0 <= x <= 1 for x in (args.target_probability, args.target_recall)):
        parser.error("top-n must be positive and probability and recall must be in [0, 1]")

    from tempfile import mkdtemp
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    else:
        root = retrieval_data.ROOT / "reports"
        root.mkdir(exist_ok=True)
        args.output_dir = Path(mkdtemp(prefix="benchmark-", dir=root))
    with retrieval_data.connect_data(args.data_dir) as db:
        training_ids = retrieval_data.training_track_ids(db)
        query_ids = retrieval_data.query_track_ids(db, "validation")
        tag_sets, _ = retrieval_data.fit_tag_sets(
            db, training_ids, min_weight=args.min_weight, max_tag_fraction=args.max_tag_fraction,
        )
    write_benchmarks(
        tag_sets, training_ids, query_ids, args.top_n, args.output_dir,
        min_weight=args.min_weight, max_tag_fraction=args.max_tag_fraction,
        target_probability=args.target_probability, target_recall=args.target_recall,
    )


if __name__ == "__main__":
    main()
