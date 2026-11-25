"""Build the fixed 2018 temporal split from the full preprocessed dataset."""

from datetime import date, timedelta
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb

from scripts.data.inspect_splits import ELIGIBLE_USERS_SQL

ROOT = Path(__file__).resolve().parents[2]
TRAIN_CUTOFF = date(2018, 1, 1)
VALIDATION_END = date(2018, 6, 30)
TEST_START = date(2018, 7, 30)
TEST_END = date(2019, 1, 26)
OBSERVATION = timedelta(days=30)


def main() -> None:
    processed = ROOT / "data/processed"
    output = ROOT / "data/splits"
    scratch = ROOT / "data/tmp"
    for name in ("events", "tracks"):
        path = processed / f"{name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Run preprocessing first. Missing {path}")
    scratch.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=scratch, prefix="splits-") as tmp, duckdb.connect(config={
        "threads": 4,
        "memory_limit": "2GB",
        "temp_directory": str(Path(tmp) / "duckdb"),
    }) as db:
        staged = Path(tmp)
        for name in ("events", "tracks"):
            db.read_parquet(str(processed / f"{name}.parquet")).create_view(name)
        latest_play = db.execute("SELECT max(timestamp) FROM events").fetchone()[0]
        if latest_play is None or latest_play.date() < TEST_END:
            raise ValueError("The listening log ends before the test track window closes.")

        print("Selecting eligible users and aggregating training plays.", flush=True)
        db.execute("CREATE TEMP TABLE eligible_users AS " + ELIGIBLE_USERS_SQL, [TRAIN_CUTOFF])
        db.execute("""
            CREATE TEMP TABLE training AS
            SELECT e.user_id, e.track_id, count(*) AS play_count, max(e.timestamp) AS last_play
            FROM events AS e JOIN eligible_users USING (user_id)
            WHERE e.timestamp < ?
            GROUP BY e.user_id, e.track_id
        """, [TRAIN_CUTOFF])
        train_pairs, train_plays, last_train_play = db.execute("""
            SELECT count(*), sum(play_count), max(last_play) FROM training
        """).fetchone()
        if last_train_play is None or last_train_play.date() >= TRAIN_CUTOFF:
            raise ValueError("Training must contain plays strictly before the cutoff.")

        db.execute("""
            CREATE TEMP TABLE track_splits AS
            SELECT played.track_id, 'train' AS split, t.first_seen
            FROM (SELECT DISTINCT track_id FROM training) AS played
            LEFT JOIN tracks AS t USING (track_id)

            UNION ALL

            SELECT t.track_id, w.split, t.first_seen
            FROM tracks AS t
            JOIN (VALUES ('validation', ?, ?), ('test', ?, ?)) AS w(split, starts, ends)
              ON t.first_seen >= w.starts AND t.first_seen < w.ends
            WHERE t.has_tags
        """, [TRAIN_CUTOFF, VALIDATION_END, TEST_START, TEST_END])

        if db.execute("""
            SELECT EXISTS (
                SELECT track_id FROM track_splits GROUP BY track_id
                HAVING count(*) > 1 OR count(first_seen) = 0
            )
        """).fetchone()[0]:
            raise ValueError("Each track needs one split and a global first_seen value.")
        if db.execute("""
            SELECT EXISTS (
                SELECT 1 FROM events AS e JOIN track_splits AS s USING (track_id)
                WHERE s.split <> 'train' AND e.timestamp < ?
            )
        """, [TRAIN_CUTOFF]).fetchone()[0]:
            raise ValueError("A validation or test track has pre-cutoff plays.")
        if db.execute("""
            SELECT EXISTS (
                SELECT 1 FROM track_splits
                WHERE split <> 'train' AND first_seen + ? > ?
            )
        """, [OBSERVATION, latest_play]).fetchone()[0]:
            raise ValueError("A validation or test track lacks a complete 30-day observation window.")

        print("Collecting distinct validation and test listeners.", flush=True)
        db.execute("""
            CREATE TEMP TABLE ground_truth AS
            SELECT DISTINCT s.split, s.track_id, e.user_id
            FROM track_splits AS s
            JOIN events AS e USING (track_id)
            JOIN eligible_users USING (user_id)
            WHERE s.split <> 'train'
              AND e.timestamp >= s.first_seen
              AND e.timestamp < s.first_seen + ?
              AND e.timestamp >= ? AND e.timestamp < ?
        """, [OBSERVATION, TRAIN_CUTOFF, TEST_END + OBSERVATION])
        if db.execute("""
            SELECT EXISTS (
                SELECT 1 FROM ground_truth ANTI JOIN eligible_users USING (user_id)
            )
        """).fetchone()[0]:
            raise ValueError("Ground truth contains an ineligible user.")

        track_counts = dict(db.execute("SELECT split, count(*) FROM track_splits GROUP BY split").fetchall())
        target_counts = dict(db.execute("SELECT split, count(*) FROM ground_truth GROUP BY split").fetchall())
        zero_counts = dict(db.execute("""
            SELECT s.split, count(*)
            FROM track_splits AS s ANTI JOIN ground_truth USING (split, track_id)
            WHERE s.split <> 'train'
            GROUP BY s.split
        """).fetchall())
        counts = {
            "eligible_users": db.sql("SELECT count(*) FROM eligible_users").fetchall()[0][0],
            "train": {"tracks": track_counts["train"], "user_track_pairs": train_pairs, "plays": train_plays},
        }
        for split in ("validation", "test"):
            counts[split] = {
                "tracks": track_counts.get(split, 0),
                "ground_truth_pairs": target_counts.get(split, 0),
                "zero_listener_tracks": zero_counts.get(split, 0),
            }
        config = {
            "dataset": "Music4All-Onion v2",
            "training_cutoff_exclusive": TRAIN_CUTOFF.isoformat(),
            "validation": {
                "start_inclusive": TRAIN_CUTOFF.isoformat(),
                "end_exclusive": VALIDATION_END.isoformat(),
            },
            "test": {
                "start_inclusive": TEST_START.isoformat(),
                "end_exclusive": TEST_END.isoformat(),
            },
            "observation_days": OBSERVATION.days,
            "eligible_user_rule": "Played at least 20 distinct tracks with positive tags before the training cutoff.",
            "new_track_rule": "Global first_seen in the arrival window, positive tags, and zero pre-cutoff plays.",
            "ground_truth_rule": "Distinct eligible listeners from first_seen inclusive to first_seen plus 30 days exclusive.",
            "counts": counts,
        }

        print("Writing split outputs.", flush=True)
        exports = {
            "eligible_users.parquet": "SELECT user_id FROM eligible_users ORDER BY user_id",
            "train_interactions.parquet": """
                SELECT user_id, track_id, play_count FROM training ORDER BY user_id, track_id
            """,
            "track_splits.parquet": "SELECT track_id, split, first_seen FROM track_splits ORDER BY split, track_id",
            "ground_truth.parquet": "SELECT split, track_id, user_id FROM ground_truth ORDER BY split, track_id, user_id",
        }
        for name, query in exports.items():
            db.sql(query).write_parquet(str(staged / name), compression="zstd")
        config_file = "split_config.json"
        (staged / config_file).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

        output.mkdir(parents=True, exist_ok=True)
        # Publish the config last so it marks a complete build.
        (output / config_file).unlink(missing_ok=True)
        for name in (*exports, config_file):
            (staged / name).replace(output / name)

    print(f"Wrote the fixed 2018 split to {output}")


if __name__ == "__main__":
    main()
