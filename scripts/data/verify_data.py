"""Verify the preprocessed data and final split against a committed reference."""

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path("validation/data_reference.json")
SPLIT_TABLES = ("eligible_users", "train_interactions", "track_splits", "ground_truth")
SPLIT_FILES = tuple(f"data/splits/{name}.parquet" for name in SPLIT_TABLES)
SPLIT_CONFIG = "data/splits/split_config.json"
PARQUET_FILES = (
    "data/processed/events.parquet",
    "data/processed/tracks.parquet",
    "data/samples/dev/users.parquet",
    "data/samples/dev/events.parquet",
    "data/samples/dev/tracks.parquet",
    "data/samples/standard/users.parquet",
    "data/samples/standard/events.parquet",
    "data/samples/standard/tracks.parquet",
) + SPLIT_FILES


def fingerprint(db: duckdb.DuckDBPyConnection, path: Path) -> dict[str, object]:
    """Record the schema and aggregate row hashes independent of row order."""
    schema = [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in pq.read_schema(path)
    ]
    # Summing wide integers keeps duplicate contributions even when XOR cancels them.
    rows, hash_sum, hash_xor = db.execute(
        """
        SELECT count(*), coalesce(sum(CAST(row_hash AS HUGEINT)), 0),
               coalesce(bit_xor(row_hash), 0)
        FROM (SELECT hash(parquet_row) AS row_hash FROM read_parquet(?) AS parquet_row)
        """,
        [str(path)],
    ).fetchall()[0]
    return {
        "row_count": rows,
        "schema": schema,
        "fingerprint": {"sum": str(hash_sum), "xor": str(hash_xor)},
    }


def check_split_consistency(
    db: duckdb.DuckDBPyConnection, config: dict, observed: dict[str, dict[str, object]]
) -> None:
    """Check split boundaries and compare training and labels with source plays."""
    cutoff = date.fromisoformat(config["training_cutoff_exclusive"])
    val_start = date.fromisoformat(config["validation"]["start_inclusive"])
    val_end = date.fromisoformat(config["validation"]["end_exclusive"])
    test_start = date.fromisoformat(config["test"]["start_inclusive"])
    test_end = date.fromisoformat(config["test"]["end_exclusive"])
    observation = timedelta(days=config["observation_days"])
    if not (cutoff == val_start < val_end <= test_start < test_end):
        raise ValueError("Split arrival windows are out of order.")
    if observation <= timedelta(0) or val_end + observation > test_start:
        raise ValueError("Validation outcomes must be available before test starts.")

    counts = config["counts"]
    expected_rows = {
        "eligible_users": counts["eligible_users"],
        "train_interactions": counts["train"]["user_track_pairs"],
        "track_splits": sum(counts[split]["tracks"] for split in ("train", "validation", "test")),
        "ground_truth": sum(counts[split]["ground_truth_pairs"] for split in ("validation", "test")),
    }
    if any(observed[f"data/splits/{name}.parquet"]["row_count"] != count
           for name, count in expected_rows.items()):
        raise ValueError("split_config.json row counts differ from the saved split tables.")

    for name in ("events", "tracks", *SPLIT_TABLES):
        folder = "splits" if name in SPLIT_TABLES else "processed"
        db.read_parquet(str(ROOT / "data" / folder / f"{name}.parquet")).create_view(name)

    latest_play = db.execute("SELECT max(timestamp) FROM events").fetchone()[0]
    if latest_play is None or latest_play.date() < test_end:
        raise ValueError("The log ends before the test track window closes.")
    if db.execute("""
        SELECT (SELECT count(*) <> count(DISTINCT track_id) FROM track_splits)
        OR EXISTS (
            SELECT 1 FROM track_splits AS s LEFT JOIN tracks AS t USING (track_id)
            WHERE s.first_seen IS DISTINCT FROM t.first_seen
               OR NOT coalesce(CASE s.split
                   WHEN 'train' THEN s.first_seen < ?
                   WHEN 'validation' THEN t.has_tags AND s.first_seen >= ? AND s.first_seen < ?
                   WHEN 'test' THEN t.has_tags AND s.first_seen >= ? AND s.first_seen < ?
               END, false)
               OR (s.split <> 'train' AND s.first_seen + ? > ?)
        )
    """, [cutoff, val_start, val_end, test_start, test_end, observation, latest_play]).fetchone()[0]:
        raise ValueError("track_splits.parquet has overlapping tracks, invalid arrivals, or incomplete follow-up.")

    # Weight the same user-track hash by play_count to compare with the individual pre-cutoff plays.
    plays, pair_hash, invalid_rows, played_tracks = db.execute("""
        SELECT sum(i.play_count), sum(CAST(hash(i.user_id, i.track_id) AS HUGEINT) * i.play_count),
               count(*) FILTER (WHERE u.user_id IS NULL OR s.track_id IS NULL OR coalesce(i.play_count, 0) <= 0),
               count(DISTINCT i.track_id)
        FROM train_interactions AS i LEFT JOIN eligible_users AS u USING (user_id)
        LEFT JOIN track_splits AS s ON i.track_id = s.track_id AND s.split = 'train'
    """).fetchone()
    history = db.execute("""
        SELECT count(*), sum(CAST(hash(e.user_id, e.track_id) AS HUGEINT))
        FROM events AS e SEMI JOIN eligible_users USING (user_id) WHERE e.timestamp < ?
    """, [cutoff]).fetchone()
    train_tracks = db.execute("SELECT count(*) FROM track_splits WHERE split = 'train'").fetchone()[0]
    if invalid_rows or played_tracks != train_tracks or (plays, pair_hash) != history:
        raise ValueError("train_interactions.parquet differs from eligible users' pre-cutoff plays.")
    if plays != counts["train"]["plays"]:
        raise ValueError("split_config.json play count differs from train_interactions.parquet.")

    if db.execute("""
        WITH expected AS (
            SELECT DISTINCT s.split, s.track_id, e.user_id
            FROM track_splits AS s JOIN events AS e USING (track_id)
            JOIN eligible_users USING (user_id)
            WHERE s.split IN ('validation', 'test')
              AND e.timestamp >= ? AND e.timestamp < ?
              AND e.timestamp >= s.first_seen AND e.timestamp < s.first_seen + ?
        )
        SELECT EXISTS (
            (SELECT split, track_id, user_id FROM ground_truth EXCEPT ALL SELECT * FROM expected)
            UNION ALL
            (SELECT * FROM expected EXCEPT ALL SELECT split, track_id, user_id FROM ground_truth)
        )
    """, [cutoff, test_end + observation, observation]).fetchone()[0]:
        raise ValueError("ground_truth.parquet differs from the distinct eligible listeners in the observation windows.")


def write_reference(path: Path, files: dict[str, dict[str, object]], config: dict) -> None:
    """Replace the reference atomically after all fingerprints are available."""
    path.parent.mkdir(parents=True, exist_ok=True)
    reference = {"duckdb_version": duckdb.__version__, "files": files, "split_config": config}
    with TemporaryDirectory(dir=path.parent) as temp:
        pending = Path(temp) / path.name
        pending.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
        pending.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Check local data against the reference")
    mode.add_argument("--write-reference", action="store_true", help="Replace the reference from local data")
    args = parser.parse_args(argv)

    path = ROOT / REFERENCE
    expected = None
    if args.check:
        try:
            reference = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(reference, dict) or not isinstance(reference.get("files"), dict):
                raise ValueError("invalid reference format")
            version = reference.get("duckdb_version")
            if version != duckdb.__version__:
                raise ValueError(f"reference uses DuckDB {version}, running {duckdb.__version__}")
            expected = reference["files"]
        except (OSError, ValueError) as error:
            print(f"FAIL {REFERENCE} ({error})")
            return 1

    failed = False
    observed = {}
    if expected is not None:
        for name in sorted(expected.keys() - set(PARQUET_FILES)):
            print(f"FAIL {name} (unexpected reference entry)")
            failed = True
    with duckdb.connect(config={"threads": 4}) as db:
        for name in PARQUET_FILES:
            try:
                result = fingerprint(db, ROOT / name)
            except FileNotFoundError:
                print(f"MISSING {name}", flush=True)
                failed = True
                continue
            except (OSError, ValueError, duckdb.Error) as error:
                print(f"FAIL {name} ({error})", flush=True)
                failed = True
                continue
            observed[name] = result
            if expected is not None:
                saved = expected.get(name)
                if not isinstance(saved, dict):
                    print(f"FAIL {name} (missing or invalid reference entry)", flush=True)
                    failed = True
                    continue
                fields = {"row_count": "row count", "schema": "schema", "fingerprint": "content fingerprint"}
                differences = [label for field, label in fields.items() if result[field] != saved.get(field)]
                if differences:
                    print(f"FAIL {name} (different {', '.join(differences)})", flush=True)
                    failed = True
                    continue
            print(f"PASS {name} ({result['row_count']:,} rows)", flush=True)

        config = None
        try:
            config = json.loads((ROOT / SPLIT_CONFIG).read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError("expected a JSON object")
        except FileNotFoundError:
            print(f"MISSING {SPLIT_CONFIG}", flush=True)
            failed = True
        except (OSError, ValueError) as error:
            print(f"FAIL {SPLIT_CONFIG} ({error})", flush=True)
            failed = True
        else:
            if expected is not None and config != reference.get("split_config"):
                print(f"FAIL {SPLIT_CONFIG} (different JSON content)", flush=True)
                failed = True
            else:
                print(f"PASS {SPLIT_CONFIG}", flush=True)

        split_inputs = {"data/processed/events.parquet", "data/processed/tracks.parquet", *SPLIT_FILES}
        if isinstance(config, dict) and split_inputs <= observed.keys():
            try:
                check_split_consistency(db, config, observed)
            except (KeyError, TypeError, ValueError, OverflowError, duckdb.Error) as error:
                print(f"FAIL split consistency ({error})", flush=True)
                failed = True
            else:
                print("PASS split consistency and timing", flush=True)

    if failed:
        return 1
    if args.write_reference:
        try:
            write_reference(path, observed, config)
        except OSError as error:
            print(f"FAIL {REFERENCE} ({error})")
            return 1
        print(f"Wrote {REFERENCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
