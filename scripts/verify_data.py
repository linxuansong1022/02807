"""Compare the main preprocessing outputs with a committed reference."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path("validation/data_reference.json")
PARQUET_FILES = (
    "data/processed/events.parquet",
    "data/processed/tracks.parquet",
    "data/samples/dev/users.parquet",
    "data/samples/dev/events.parquet",
    "data/samples/dev/tracks.parquet",
    "data/samples/standard/users.parquet",
    "data/samples/standard/events.parquet",
    "data/samples/standard/tracks.parquet",
)


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
    ).fetchone()
    return {
        "row_count": rows,
        "schema": schema,
        "fingerprint": {"sum": str(hash_sum), "xor": str(hash_xor)},
    }


def write_reference(path: Path, files: dict[str, dict[str, object]]) -> None:
    """Replace the reference atomically after all fingerprints are available."""
    path.parent.mkdir(parents=True, exist_ok=True)
    reference = {"duckdb_version": duckdb.__version__, "files": files}
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
            except (OSError, ValueError, duckdb.Error) as error:
                print(f"FAIL {name} ({error})", flush=True)
                failed = True
                continue
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
            observed[name] = result
            print(f"PASS {name} ({result['row_count']:,} rows)", flush=True)

    if failed:
        return 1
    if args.write_reference:
        try:
            write_reference(path, observed)
        except OSError as error:
            print(f"FAIL {REFERENCE} ({error})")
            return 1
        print(f"Wrote {REFERENCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
