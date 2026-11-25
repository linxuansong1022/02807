"""Select deterministic user rankings and export evaluator-compatible predictions."""

from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def top_k(scores: np.ndarray, k: int, user_indices: np.ndarray | None = None) -> np.ndarray:
    """Return positions by descending score then ascending user index.

    Supply unique indices in the globally sorted user ID order when the score
    vector is in another order. Otherwise its positions give the ID order.
    Supply zero for candidates with no stored score.
    """
    if scores.ndim != 1 or not 1 <= k <= len(scores) or not np.isfinite(scores).all():
        raise ValueError("Top-K requires finite scores in one dimension and K within the candidate count.")
    if user_indices is not None and user_indices.shape != scores.shape:
        raise ValueError("User indices must align with scores.")

    if k == len(scores):
        selected = np.arange(k)
    else:
        boundary = np.partition(scores, len(scores) - k)[len(scores) - k]
        above = np.flatnonzero(scores > boundary)
        tied = np.flatnonzero(scores == boundary)
        slots = k - len(above)
        if user_indices is not None and len(tied) > slots:
            tied = tied[np.argpartition(user_indices[tied], slots - 1)[:slots]]
        selected = np.concatenate((above, tied[:slots]))

    ids = selected if user_indices is None else user_indices[selected].astype(np.intp, copy=False)
    return selected[np.lexsort((-ids, scores[selected]))[::-1]]


def export_predictions(
    path: Path,
    track_ids: Sequence[str],
    user_ids: Sequence[str],
    ranked_users: np.ndarray,
    scores: np.ndarray | None = None,
) -> None:
    """Write one row per track and rank, mapping user indices back to string IDs."""
    if ranked_users.ndim != 2 or ranked_users.shape[0] != len(track_ids) or not ranked_users.shape[1]:
        raise ValueError("Ranked user indices must be a track-by-rank matrix.")
    if not np.issubdtype(ranked_users.dtype, np.integer) or np.any(ranked_users < 0) or np.any(
        ranked_users >= len(user_ids)
    ):
        raise ValueError("Ranking contains invalid user indices.")
    if len(set(track_ids)) != len(track_ids) or len(set(user_ids)) != len(user_ids):
        raise ValueError("Track and user IDs must be unique.")
    if np.any(np.diff(np.sort(ranked_users, axis=1), axis=1) == 0):
        raise ValueError("A user appears at multiple ranks for a track.")

    k = ranked_users.shape[1]
    columns = {
        "track_id": pa.array(np.repeat(np.asarray(track_ids, dtype=object), k), type=pa.string()),
        "user_id": pa.array(np.asarray(user_ids, dtype=object)[ranked_users.ravel()], type=pa.string()),
        "rank": pa.array(np.tile(np.arange(1, k + 1, dtype=np.int32), len(track_ids))),
    }
    if scores is not None:
        if scores.shape != ranked_users.shape or not np.isfinite(scores).all():
            raise ValueError("Prediction scores must be finite and match the rankings.")
        columns["score"] = pa.array(scores.ravel())

    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        staged = Path(directory) / path.name
        pq.write_table(pa.table(columns), staged, compression="zstd")
        staged.replace(path)
