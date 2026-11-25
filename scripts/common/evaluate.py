"""Evaluate ranked user recommendations for the held-out tracks."""

import argparse
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
RANK_CUTOFFS = (100, 500, 1000)
SPLITS = ("validation", "test")
INTEGER_TYPES = {
    "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
    "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT",
}


def evaluate(
    predictions: Path,
    split: str,
    split_dir: Path = ROOT / "data/splits",
) -> tuple[int, int, list[tuple[int, float | None, float | None]]]:
    """Return total tracks, zero-listener tracks, and (K, recall, NDCG) macro averages."""
    if split not in SPLITS:
        raise ValueError("Select the validation or test split.")

    max_rank = RANK_CUTOFFS[-1]
    with duckdb.connect(config={"threads": 4}) as db:
        recommendations = db.read_parquet(str(predictions))
        types = dict(zip(recommendations.columns, map(str, recommendations.types)))
        missing = {"track_id", "user_id", "rank"} - types.keys()
        if missing:
            raise ValueError(f"Missing prediction columns ({', '.join(sorted(missing))}).")
        if types["track_id"] != "VARCHAR" or types["user_id"] != "VARCHAR":
            raise ValueError("Prediction track_id and user_id must contain the original string IDs.")
        if types["rank"] not in INTEGER_TYPES:
            raise ValueError("Prediction rank must have an integer type.")
        recommendations.create_view("predictions")
        for name in ("eligible_users", "track_splits", "ground_truth"):
            db.read_parquet(str(split_dir / f"{name}.parquet")).create_view(name)
        db.execute("""
            CREATE TEMP TABLE evaluation_tracks AS
            SELECT DISTINCT track_id FROM track_splits WHERE split = ?
        """, [split])

        invalid = db.execute("""
            SELECT p.track_id, p.user_id, CASE
                WHEN t.track_id IS NULL THEN 'track is outside the selected split'
                WHEN u.user_id IS NULL THEN 'user is ineligible'
                ELSE 'rank must be a positive integer'
            END AS issue
            FROM predictions AS p
            LEFT JOIN evaluation_tracks AS t USING (track_id)
            LEFT JOIN eligible_users AS u USING (user_id)
            WHERE t.track_id IS NULL OR u.user_id IS NULL OR p.rank IS NULL OR p.rank < 1
            LIMIT 1
        """).fetchone()
        if invalid:
            track_id, user_id, issue = invalid
            raise ValueError(f"Invalid prediction for track {track_id!r}, user {user_id!r} ({issue}).")

        coverage = db.execute("""
            SELECT t.track_id, count(p.rank) AS recommendations,
                   count(DISTINCT p.user_id) AS distinct_users,
                   count(DISTINCT p.rank) AS distinct_ranks,
                   count(*) FILTER (WHERE p.rank <= ?) AS top_ranks
            FROM evaluation_tracks AS t LEFT JOIN predictions AS p USING (track_id)
            GROUP BY t.track_id
            HAVING recommendations <> distinct_users OR recommendations <> distinct_ranks
                OR top_ranks <> ?
            ORDER BY t.track_id
            LIMIT 1
        """, [max_rank, max_rank]).fetchone()
        if coverage:
            track_id, total, users, ranks, top = coverage
            if users != total or ranks != total:
                raise ValueError(f"Track {track_id!r} has duplicate users or ranks.")
            raise ValueError(f"Track {track_id!r} needs ranks 1 through {max_rank:,} (found {top:,}).")

        cutoff_rows = ", ".join(f"({k})" for k in RANK_CUTOFFS)
        rows = db.execute(f"""
            WITH cutoffs(k) AS (VALUES {cutoff_rows}),
            discounts AS (
                SELECT rank, 1.0 / log2(rank + 1) AS gain,
                       sum(1.0 / log2(rank + 1)) OVER (ORDER BY rank) AS ideal_dcg
                FROM range(1, ?) AS positions(rank)
            ), labels AS (
                SELECT DISTINCT g.track_id, g.user_id
                FROM ground_truth AS g JOIN evaluation_tracks USING (track_id)
                WHERE g.split = ?
            ), listener_counts AS (
                SELECT track_id, count(*) AS listeners FROM labels GROUP BY track_id
            ), hits AS (
                SELECT p.track_id, p.rank, d.gain
                FROM predictions AS p JOIN labels USING (track_id, user_id)
                JOIN discounts AS d ON p.rank = d.rank
            ), per_track AS (
                SELECT l.track_id, c.k, count(h.rank) / l.listeners AS recall,
                       coalesce(sum(h.gain), 0.0) / d.ideal_dcg AS ndcg
                FROM listener_counts AS l CROSS JOIN cutoffs AS c
                JOIN discounts AS d ON d.rank = least(l.listeners, c.k)
                LEFT JOIN hits AS h ON h.track_id = l.track_id AND h.rank <= c.k
                GROUP BY l.track_id, c.k, l.listeners, d.ideal_dcg
            )
            SELECT c.k, count(p.track_id), avg(p.recall), avg(p.ndcg)
            FROM cutoffs AS c LEFT JOIN per_track AS p USING (k)
            GROUP BY c.k ORDER BY c.k
        """, [max_rank + 1, split]).fetchall()
        total_tracks = db.execute("SELECT count(*) FROM evaluation_tracks").fetchone()[0]

    metrics = [(k, recall, ndcg) for k, _, recall, ndcg in rows]
    return total_tracks, total_tracks - rows[0][1], metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Model predictions in Parquet")
    parser.add_argument("--split", choices=SPLITS, required=True)
    args = parser.parse_args()
    try:
        total, zero, metrics = evaluate(args.predictions, args.split)
    except (OSError, ValueError, duckdb.Error) as error:
        parser.exit(1, f"Evaluation failed ({error})\n")

    print(f"{args.split.title()} tracks ({total:,} total, {total - zero:,} scored, {zero:,} with zero listeners)")
    print(f"{'K':>5} {'Recall@K':>10} {'NDCG@K':>10}")
    for k, recall, ndcg in metrics:
        recall_text = "n/a" if recall is None else f"{recall:.6f}"
        ndcg_text = "n/a" if ndcg is None else f"{ndcg:.6f}"
        print(f"{k:>5} {recall_text:>10} {ndcg_text:>10}")


if __name__ == "__main__":
    main()
