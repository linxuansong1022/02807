"""Generate random, most-active, and TF-IDF tag-profile baseline predictions."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix

from scripts.common.evaluate import RANK_CUTOFFS, SPLITS
from scripts.common.features import ROOT, load_interactions, load_track_features
from scripts.common.scoring import export_predictions, top_k

TOP_K = RANK_CUTOFFS[-1]
SEED = 42
USER_BATCH_SIZE = 2048


def _normalize_rows(matrix: csr_matrix) -> int:
    """Normalize nonzero rows in place and return the number of zero rows."""
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    inverse = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    matrix.data *= np.repeat(inverse, np.diff(matrix.indptr))
    return len(norms) - np.count_nonzero(norms)


def _tag_rankings(
    interactions: csr_matrix, training: csr_matrix, evaluation: csr_matrix, k: int, batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    evaluation = evaluation.copy()
    zero_tracks = _normalize_rows(evaluation)
    evaluation_transposed = evaluation.T.tocsr()
    n_users = interactions.shape[0]
    n_tracks = evaluation.shape[0]
    best_users = np.empty((n_tracks, 0), dtype=np.int32)
    best_scores = np.empty((n_tracks, 0), dtype=np.float32)
    profile_nnz = zero_profiles = peak_profile_bytes = peak_score_bytes = 0
    profile_seconds = 0.0

    for start in range(0, n_users, batch_size):
        stop = min(start + batch_size, n_users)
        tick = perf_counter()
        profiles = interactions[start:stop] @ training
        zero_profiles += _normalize_rows(profiles)
        profile_seconds += perf_counter() - tick
        profile_nnz += profiles.nnz
        peak_profile_bytes = max(
            peak_profile_bytes, profiles.data.nbytes + profiles.indices.nbytes + profiles.indptr.nbytes,
        )

        # Each track sees every user, including implicit zeros in this score block.
        block = (profiles @ evaluation_transposed).toarray(order="F").T
        del profiles
        peak_score_bytes = max(peak_score_bytes, block.nbytes)
        batch_users = np.arange(start, stop, dtype=np.int32)
        width = min(k, stop)
        next_users = np.empty((n_tracks, width), dtype=np.int32)
        next_scores = np.empty((n_tracks, width), dtype=np.float32)
        for track in range(n_tracks):
            users = np.concatenate((best_users[track], batch_users))
            scores = np.concatenate((best_scores[track], block[track]))
            selected = top_k(scores, width, users)
            next_users[track] = users[selected]
            next_scores[track] = scores[selected]
        best_users, best_scores = next_users, next_scores
        del block

    return best_users, best_scores, {
        "shape": (n_users, training.shape[1]),
        "nnz": profile_nnz,
        "zero_profiles": zero_profiles,
        "zero_feature_tracks": zero_tracks,
        "profile_seconds": profile_seconds,
        "peak_profile_batch_bytes": peak_profile_bytes,
        "peak_score_block_bytes": peak_score_bytes,
    }


def generate_baselines(
    split: str, root: Path = ROOT, *, batch_size: int = USER_BATCH_SIZE,
) -> tuple[dict[str, Path], np.ndarray, dict]:
    """Generate all baseline predictions and return paths, user activity, and diagnostics."""
    if split not in SPLITS or batch_size < 1:
        raise ValueError("Select validation or test and a positive user batch size.")
    interactions, users, train_ids = load_interactions(root)
    features, track_ids, _ = load_track_features("tfidf", root, training_ids=train_ids)
    if interactions.shape[1] != features["train"].shape[0]:
        raise ValueError("Interaction columns must align with training tag rows.")
    if len(users) < TOP_K:
        raise ValueError(f"The eligible pool must contain at least {TOP_K} users.")

    tracks = track_ids[split]
    shape = (len(tracks), TOP_K)
    paths = {
        model: root / "data" / "predictions" / f"baseline_{model}_{split}.parquet"
        for model in ("random", "most_active", "tag_profile")
    }
    start = perf_counter()
    rng = np.random.default_rng([SEED, SPLITS.index(split)])
    random_users = np.empty(shape, dtype=np.int32)
    for row in random_users:
        row[:] = rng.choice(len(users), TOP_K, replace=False)
    seconds = {"random": perf_counter() - start}
    export_predictions(paths["random"], tracks, users, random_users)
    del random_users

    start = perf_counter()
    activity = np.asarray(interactions.sum(axis=1, dtype=np.int64)).ravel()
    active = top_k(activity, TOP_K)
    seconds["most_active"] = perf_counter() - start
    export_predictions(
        paths["most_active"], tracks, users, np.broadcast_to(active, shape),
        np.broadcast_to(activity[active], shape),
    )

    start = perf_counter()
    ranked_users, scores, profile = _tag_rankings(
        interactions, features["train"], features[split], TOP_K, batch_size,
    )
    seconds["tag_profile"] = perf_counter() - start
    export_predictions(paths["tag_profile"], tracks, users, ranked_users, scores)
    return paths, activity, {"seconds": seconds, "profile": profile}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--batch-size", type=int, default=USER_BATCH_SIZE, help="Users per tag-profile batch")
    args = parser.parse_args()
    paths, _, _ = generate_baselines(args.split, batch_size=args.batch_size)
    for path in paths.values():
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
