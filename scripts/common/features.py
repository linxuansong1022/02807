"""Build aligned sparse interactions and track features from the verified split."""

from collections.abc import Sequence
import json
from pathlib import Path

import duckdb
import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "validation", "test")
WEIGHTINGS = ("binary", "weighted", "tfidf")


def _track_index(db: duckdb.DuckDBPyConnection, root: Path) -> dict[str, list[str]]:
    """Assign zero-based track indices in sorted ID order within each split."""
    db.execute("""
        CREATE TEMP TABLE track_index AS
        SELECT split, track_id, row_number() OVER (PARTITION BY split ORDER BY track_id) - 1 AS idx
        FROM read_parquet(?)
    """, [str(root / "data/splits/track_splits.parquet")])
    tracks = {split: [] for split in SPLITS}
    seen = set()
    for split, track_id in db.execute("SELECT split, track_id FROM track_index ORDER BY split, idx").fetchall():
        if split not in tracks or not isinstance(track_id, str) or track_id in seen:
            raise ValueError("track_splits.parquet must contain unique string track IDs and valid split labels.")
        seen.add(track_id)
        tracks[split].append(track_id)
    return tracks


def _split_counts(root: Path) -> dict:
    config = json.loads((root / "data/splits/split_config.json").read_text(encoding="utf-8"))
    reference = json.loads((root / "validation/data_reference.json").read_text(encoding="utf-8"))
    counts = config["counts"]
    if counts != reference["split_config"]["counts"]:
        raise ValueError("Local split counts differ from the committed reference. Run the data verifier.")
    reference_rows = {
        "eligible_users": counts["eligible_users"],
        "train_interactions": counts["train"]["user_track_pairs"],
        "track_splits": sum(counts[split]["tracks"] for split in SPLITS),
    }
    for name, count in reference_rows.items():
        if reference["files"][f"data/splits/{name}.parquet"]["row_count"] != count:
            raise ValueError(f"Split statistics disagree with the reference row count for {name}.")
    return counts


def load_interactions(root: Path = ROOT) -> tuple[csr_matrix, list[str], list[str]]:
    """Return float32 CSR play counts, eligible user IDs, and training track IDs."""
    counts = _split_counts(root)
    split_dir = root / "data/splits"
    with duckdb.connect(config={"threads": 4, "memory_limit": "2GB"}) as db:
        tracks = _track_index(db, root)
        db.execute("""
            CREATE TEMP TABLE user_index AS
            SELECT user_id, row_number() OVER (ORDER BY user_id) - 1 AS idx FROM read_parquet(?)
        """, [str(split_dir / "eligible_users.parquet")])
        users = [user_id for (user_id,) in db.execute("SELECT user_id FROM user_index ORDER BY idx").fetchall()]
        if len(set(users)) != len(users) or any(not isinstance(user, str) for user in users):
            raise ValueError("eligible_users.parquet must contain unique string user IDs.")
        shape = (len(users), len(tracks["train"]))
        if len(users) != counts["eligible_users"] or any(
            len(tracks[split]) != counts[split]["tracks"] for split in SPLITS
        ):
            raise ValueError("User or track counts differ from the verified split statistics.")

        interactions = db.read_parquet(str(split_dir / "train_interactions.parquet"))
        types = dict(zip(interactions.columns, map(str, interactions.types)))
        if (types.get("user_id"), types.get("track_id"), types.get("play_count")) != (
            "VARCHAR", "VARCHAR", "BIGINT",
        ):
            raise ValueError("train_interactions.parquet must contain string IDs and BIGINT play counts.")
        interactions.create_view("interactions")
        pairs, plays, played_tracks, invalid_counts, max_count = db.execute("""
            SELECT count(*), coalesce(sum(play_count::HUGEINT), 0), count(DISTINCT track_id),
                   count(*) FILTER (WHERE play_count IS NULL OR play_count <= 0), coalesce(max(play_count), 0)
            FROM interactions
        """).fetchall()[0]
        if invalid_counts:
            raise ValueError("Training play counts must be positive integers.")
        if (pairs, plays, played_tracks) != (counts["train"]["user_track_pairs"], counts["train"]["plays"], shape[1]):
            raise ValueError("Training pairs, exact play total, or track count differ from the verified split statistics.")

        # Stream sorted coordinates directly into CSR arrays, with a count for each row.
        index_type = np.int32 if max(*shape, pairs) <= np.iinfo(np.int32).max else np.int64
        data = np.empty(pairs, dtype=np.float32)
        indices = np.empty(pairs, dtype=index_type)
        row_lengths = np.zeros(shape[0], dtype=index_type)
        offset = 0
        reader = db.sql("""
            SELECT u.idx AS row_idx, t.idx AS col_idx, i.play_count
            FROM interactions AS i JOIN user_index AS u USING (user_id)
            JOIN track_index AS t ON i.track_id = t.track_id AND t.split = 'train'
            ORDER BY u.idx, t.idx
        """).to_arrow_reader(batch_size=262_144)
        for batch in reader:
            rows, columns, raw_counts = (column.to_numpy() for column in batch.columns)
            end = offset + len(raw_counts)
            indices[offset:end] = columns
            data[offset:end] = raw_counts
            if max_count > 2**24:
                with np.errstate(invalid="ignore"):
                    if np.any(data[offset:end].astype(np.int64) != raw_counts):
                        raise ValueError("A training play count cannot be represented exactly as float32.")
            row_lengths += np.bincount(rows, minlength=shape[0]).astype(index_type, copy=False)
            offset = end

    if offset != pairs:
        raise ValueError("Training interactions contain IDs outside the eligible users or training tracks.")
    indptr = np.empty(shape[0] + 1, dtype=index_type)
    indptr[0] = 0
    np.cumsum(row_lengths, out=indptr[1:])
    matrix = csr_matrix((data, indices, indptr), shape=shape)
    # Positive, exactly represented integers are finite and give one nonzero per pair.
    if not matrix.has_canonical_format or matrix.nnz != pairs:
        raise ValueError("Training interactions must have one row per user-track pair.")
    return matrix, users, tracks["train"]


def load_track_features(
    weighting: str = "binary", root: Path = ROOT, *, training_ids: Sequence[str] | None = None,
) -> tuple[dict[str, csr_matrix], dict[str, list[str]], list[str]]:
    """Return CSR matrices and ordered track IDs by split, plus the ordered tag vocabulary.

    Binary values are 1 for positive tags. Weighted values are the original
    positive tag weights. TF-IDF uses the original weight as TF and
    log((1 + N) / (1 + df)) + 1 as IDF, where N counts tagged training tracks
    and df counts training tracks with the tag. Normalize each nonempty TF-IDF
    row to unit L2 norm after dropping tags outside the training vocabulary.
    Pass the training IDs returned by load_interactions to check alignment.
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(f"Choose a tag weighting from {', '.join(WEIGHTINGS)}.")
    with duckdb.connect(config={"threads": 4}) as db:
        tracks = _track_index(db, root)
        if training_ids is not None and list(training_ids) != tracks["train"]:
            raise ValueError("Interaction and content training track IDs differ or are out of order.")
        db.execute("""
            CREATE TEMP TABLE split_metadata AS
            SELECT s.split, s.idx, t.tags, t.tag_weights
            FROM track_index AS s LEFT JOIN read_parquet(?) AS t USING (track_id)
        """, [str(root / "data/processed/tracks.parquet")])
        metadata_count, invalid_lists = db.execute("""
            SELECT count(*), count(*) FILTER (
                WHERE tags IS NULL OR tag_weights IS NULL OR len(tags) <> len(tag_weights)
                   OR list_unique(tags) <> len(tags)
            )
            FROM split_metadata
        """).fetchall()[0]
        if invalid_lists or metadata_count != sum(map(len, tracks.values())):
            raise ValueError("Each split track must have one metadata row with aligned weights and unique tag IDs.")
        db.execute("""
            CREATE TEMP TABLE split_tags AS
            SELECT split, idx, unnest(tags) AS tag, unnest(tag_weights) AS weight
            FROM split_metadata
        """)
        if db.execute("""
            SELECT EXISTS (SELECT 1 FROM split_tags WHERE weight IS NULL OR NOT isfinite(weight) OR weight < 0)
        """).fetchall()[0][0]:
            raise ValueError("Tag weights must be finite and nonnegative.")
        db.execute("""
            CREATE TEMP TABLE vocabulary AS
            SELECT tag, row_number() OVER (ORDER BY tag) - 1 AS col, count(*) AS df
            FROM split_tags WHERE split = 'train' AND weight > 0 GROUP BY tag
        """)
        documents = db.execute("SELECT tag, df FROM vocabulary ORDER BY col").fetchall()
        vocabulary = [tag for tag, _ in documents]
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("Feature IDs must be unique.")
        if weighting == "tfidf":
            tagged_train = db.execute("""
                SELECT count(DISTINCT idx) FROM split_tags WHERE split = 'train' AND weight > 0
            """).fetchall()[0][0]
            df = np.array([frequency for _, frequency in documents], dtype=np.float64)
            idf = np.log((1 + tagged_train) / (1 + df)) + 1

        matrices = {}
        for split, ids in tracks.items():
            coordinates = db.execute("""
                SELECT t.idx AS row_idx, v.col AS col_idx, t.weight
                FROM split_tags AS t JOIN vocabulary AS v USING (tag)
                WHERE t.split = ? AND t.weight > 0
            """, [split]).fetchnumpy()
            rows, cols = coordinates["row_idx"], coordinates["col_idx"]
            values = coordinates["weight"]
            if weighting == "binary":
                values = np.ones(len(rows), dtype=np.float32)
            elif weighting == "tfidf":
                values *= idf[cols]
                norms = np.sqrt(np.bincount(rows, weights=values**2, minlength=len(ids)))
                values /= norms[rows]

            values = values.astype(np.float32, copy=False)
            if not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError(f"{split} has invalid float32 tag features.")
            shape = (len(ids), len(vocabulary))
            matrix = csr_matrix((values, (rows, cols)), shape=shape)
            if matrix.nnz != len(values):
                raise ValueError(f"{split} has duplicate track-tag coordinates.")
            matrices[split] = matrix
    return matrices, tracks, vocabulary
