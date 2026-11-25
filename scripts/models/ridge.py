"""Map cold-start track features into the existing ALS space with sparse Ridge."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

from scripts.common.evaluate import RANK_CUTOFFS, SPLITS
from scripts.common.features import ROOT, WEIGHTINGS, load_track_features
from scripts.common.scoring import export_predictions, top_k
from scripts.models.als import artifact_sha256, write_factors

DEFAULTS = {"weighting": "tfidf", "regularization": 1.0, "atol": 1e-6, "btol": 1e-6, "iter_lim": 200}
ALS_FILES = ("als_config.json", "als_user_factors.parquet", "als_track_factors.parquet")
WEIGHTS = "ridge_weights.npy"
VOCABULARY = "ridge_vocabulary.json"
CONFIG = "ridge_config.json"


def _read_factors(path: Path, id_column: str, dimension: int) -> tuple[list[str], np.ndarray]:
    table = pq.read_table(path, columns=[id_column, "factors"])
    column = table["factors"].combine_chunks()
    if table[id_column].type != pa.string() or column.type != pa.list_(pa.float32(), dimension):
        raise ValueError(f"Incorrect ID or factor schema in {path.name}.")
    ids = table[id_column].to_pylist()
    if None in ids or len(set(ids)) != len(ids) or column.null_count or column.values.null_count:
        raise ValueError(f"Null values or duplicate IDs in {path.name}.")
    factors = column.values.to_numpy().reshape(len(ids), dimension)
    if not np.isfinite(factors).all():
        raise ValueError(f"Nonfinite factors in {path.name}.")
    return ids, factors


def fit_weights(
    features: csr_matrix, targets: np.ndarray, *,
    regularization: float = 1.0, atol: float = 1e-6, btol: float = 1e-6, iter_lim: int = 200,
) -> tuple[np.ndarray, list[dict]]:
    """Solve each output dimension with damped LSQR and require successful termination."""
    if not (0 < regularization < np.inf and 0 <= atol < np.inf and 0 <= btol < np.inf and iter_lim > 0):
        raise ValueError("Use positive regularization and iteration limit and finite, nonnegative tolerances.")
    if targets.ndim != 2 or targets.shape[0] != features.shape[0] or not features.nnz:
        raise ValueError("Nonempty training features must align with the ALS target rows.")
    design = features.astype(np.float64, copy=False)
    damping = np.sqrt(regularization)
    weights = np.empty((design.shape[1], targets.shape[1]), dtype=np.float32)
    diagnostics = []
    failures = []
    for dimension, target in enumerate(targets.T):
        vector, stop, iterations, residual, damped_residual, _, condition, normal_residual, *_ = lsqr(
            design, target.astype(np.float64), damp=damping, atol=atol, btol=btol, iter_lim=iter_lim,
        )
        info = {
            "dimension": dimension,
            "stop_code": int(stop),
            "iterations": int(iterations),
            "residual_norm": float(residual),
            "damped_residual_norm": float(damped_residual),
            "normal_residual_norm": float(normal_residual),
        }
        diagnostics.append(info)
        # Codes 3 and 6 indicate conditioning limits. Code 7 is the iteration limit.
        if stop not in (0, 1, 2, 4, 5) or not np.isfinite(
            [residual, damped_residual, normal_residual, condition]
        ).all():
            failures.append(f"{dimension} (code {stop}, {iterations} iterations, residual {residual:g})")
        weights[:, dimension] = vector
    if failures:
        raise RuntimeError("LSQR did not converge for dimensions " + ", ".join(failures))
    if not np.isfinite(weights).all():
        raise ValueError("Ridge coefficients must be finite float32 values.")
    return weights, diagnostics


def rank_tracks(
    users: np.ndarray, tracks: np.ndarray, k: int, batch_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Score every user in bounded track blocks and keep the top K per track."""
    if not 1 <= k <= len(users) or batch_size < 1 or users.shape[1] != tracks.shape[1]:
        raise ValueError("Scoring dimensions, K, or batch size are invalid.")
    ranked_users = np.empty((len(tracks), k), dtype=np.int32)
    ranked_scores = np.empty((len(tracks), k), dtype=np.float32)
    peak_bytes = 0
    for start in range(0, len(tracks), batch_size):
        block = tracks[start:start + batch_size] @ users.T
        peak_bytes = max(peak_bytes, block.nbytes)
        for row, scores in enumerate(block, start):
            indices = top_k(scores, k)
            ranked_users[row] = indices
            ranked_scores[row] = scores[indices]
        del block
    return ranked_users, ranked_scores, peak_bytes


def generate_ridge(
    split: str = "validation", root: Path = ROOT, *,
    weighting: str | None = None, regularization: float | None = None,
    atol: float | None = None, btol: float | None = None, iter_lim: int | None = None,
    batch_size: int = 64,
) -> dict:
    """Fit on training tracks for validation or apply a frozen mapping for test."""
    if split not in SPLITS or batch_size < 1:
        raise ValueError("Select validation or test and a positive track batch size.")
    options = {
        "weighting": weighting, "regularization": regularization, "atol": atol,
        "btol": btol, "iter_lim": iter_lim,
    }
    overrides = {name: value for name, value in options.items() if value is not None}
    models = root / "data/models"
    predictions = root / "data/predictions"
    splits = root / "data/splits"
    als_hashes = {name: artifact_sha256(models / name) for name in ALS_FILES}
    als = json.loads((models / ALS_FILES[0]).read_text(encoding="utf-8"))
    paired_hashes = als.get("factor_sha256")
    if paired_hashes is not None and paired_hashes != {name: als_hashes[name] for name in ALS_FILES[1:]}:
        raise ValueError("The ALS factor files do not match the ALS run configuration.")
    split_config = json.loads((splits / "split_config.json").read_text(encoding="utf-8"))
    counts = split_config["counts"]
    training_split = {
        "dataset": split_config["dataset"],
        "training_cutoff_exclusive": split_config["training_cutoff_exclusive"],
        "eligible_users": counts["eligible_users"],
        **counts["train"],
    }
    if als["training_split"] != training_split:
        raise ValueError("ALS factors must come from the current training split.")

    feature_hash = artifact_sha256(root / "data/processed/tracks.parquet")
    if split == "test":
        if overrides:
            raise ValueError("Test uses the saved training settings. Supply solver options during validation.")
        config = json.loads((models / CONFIG).read_text(encoding="utf-8"))
        settings = config["settings"]
        if config["als_sha256"] != als_hashes or config["track_metadata_sha256"] != feature_hash:
            raise ValueError("Saved Ridge mapping was built with different ALS artifacts or track metadata.")
        for name in (WEIGHTS, VOCABULARY):
            if artifact_sha256(models / name) != config["ridge_sha256"][name]:
                raise ValueError(f"{name} does not match the saved Ridge configuration.")
    else:
        settings = DEFAULTS | overrides

    dimension = als["factors"]
    user_ids, users = _read_factors(models / ALS_FILES[1], "user_id", dimension)
    train_ids, track_targets = _read_factors(models / ALS_FILES[2], "track_id", dimension)
    with duckdb.connect() as db:
        expected_users = [user for (user,) in db.execute(
            "SELECT user_id FROM read_parquet(?) ORDER BY user_id", [str(splits / "eligible_users.parquet")],
        ).fetchall()]
    if user_ids != expected_users or len(user_ids) != counts["eligible_users"]:
        raise ValueError("ALS user IDs must match the complete sorted eligible-user pool.")
    features, track_ids, vocabulary = load_track_features(settings["weighting"], root, training_ids=train_ids)
    if any(len(ids) != counts[name]["tracks"] or features[name].shape != (len(ids), len(vocabulary))
           for name, ids in track_ids.items()):
        raise ValueError("Feature rows or columns differ from the split IDs or vocabulary.")
    shape = (len(vocabulary), dimension)

    if split == "validation":
        train = features["train"]
        usable = np.diff(train.indptr) != 0
        training_features, targets = train[usable], track_targets[usable]
        tick = perf_counter()
        weights, solver = fit_weights(
            training_features, targets, regularization=settings["regularization"],
            atol=settings["atol"], btol=settings["btol"], iter_lim=settings["iter_lim"],
        )
        training_seconds = perf_counter() - tick
        error = training_features @ weights - targets
        squared_error = np.sum(np.square(error, dtype=np.float64))
        config = {
            "settings": settings,
            "solver": "scipy.sparse.linalg.lsqr",
            "scipy_version": scipy.__version__,
            "intercept": False,
            "damp": float(np.sqrt(settings["regularization"])),
            "training_seconds": training_seconds,
            "training_split": training_split,
            "training_features": {
                "tracks": len(train_ids),
                "usable_tracks": int(usable.sum()),
                "zero_feature_tracks": int((~usable).sum()),
                "vocabulary_size": len(vocabulary),
                "latent_dimension": dimension,
            },
            "training_reconstruction_frobenius": float(np.sqrt(squared_error)),
            "training_reconstruction_rmse": float(np.sqrt(squared_error / error.size)),
            "weight_frobenius": float(np.linalg.norm(weights)),
            "lsqr_diagnostics": solver,
            "als_sha256": als_hashes,
            "als_has_factor_hashes": paired_hashes is not None,
            "track_metadata_sha256": feature_hash,
            "ridge_sha256": {},
            "evaluation": {},
        }
    else:
        if json.loads((models / VOCABULARY).read_text(encoding="utf-8")) != vocabulary:
            raise ValueError("Current feature column order differs from the saved Ridge vocabulary.")
        weights = np.load(models / WEIGHTS, allow_pickle=False)
    if weights.shape != shape or weights.dtype != np.float32 or not np.isfinite(weights).all():
        raise ValueError("Invalid Ridge weights for the current feature or ALS dimensions.")

    evaluation = features[split]
    tick = perf_counter()
    embeddings = evaluation @ weights
    embedding_seconds = perf_counter() - tick
    zeros = np.diff(evaluation.indptr) == 0
    if (embeddings.shape != (len(track_ids[split]), dimension) or not np.isfinite(embeddings).all()
            or np.any(embeddings[zeros] != 0)):
        raise ValueError("Invalid cold-start embeddings or nonzero embeddings for empty feature rows.")
    k = RANK_CUTOFFS[-1]
    tick = perf_counter()
    ranked, scores, peak_bytes = rank_tracks(users, embeddings, k, batch_size)
    scoring_seconds = perf_counter() - tick
    sample_rows = (0, len(embeddings) // 2, -1) if len(embeddings) else ()
    for row in sample_rows:
        positions = [0, k // 2, k - 1]
        direct = np.sum(users[ranked[row, positions]].astype(np.float64) * embeddings[row], axis=1)
        if not np.allclose(scores[row, positions], direct, rtol=2e-5, atol=2e-6):
            raise ValueError("Batched ranking scores differ from direct dot products.")
    config["evaluation"][split] = {
        "tracks": len(embeddings),
        "zero_feature_tracks": int(zeros.sum()),
        "track_batch_size": batch_size,
        "embedding_seconds": embedding_seconds,
        "scoring_seconds": scoring_seconds,
        "largest_score_block_bytes": peak_bytes,
        "estimated_full_score_bytes": len(users) * len(embeddings) * scores.itemsize,
    }

    factor_file = f"ridge_{split}_factors.parquet"
    prediction_file = f"ridge_{split}.parquet"
    predictions.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=models) as tmp_models, TemporaryDirectory(dir=predictions) as tmp_predictions:
        staged = Path(tmp_models)
        prediction_path = Path(tmp_predictions) / prediction_file
        model_files = [factor_file]
        write_factors(staged / factor_file, "track_id", track_ids[split], embeddings, dimension)
        export_predictions(prediction_path, track_ids[split], user_ids, ranked, scores)
        if split == "validation":
            np.save(staged / WEIGHTS, weights, allow_pickle=False)
            (staged / VOCABULARY).write_text(json.dumps(vocabulary, ensure_ascii=False) + "\n", encoding="utf-8")
            model_files.extend((WEIGHTS, VOCABULARY))
        config["ridge_sha256"].update({name: artifact_sha256(staged / name) for name in model_files})
        config["ridge_sha256"][prediction_file] = artifact_sha256(prediction_path)
        (staged / CONFIG).write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if any(artifact_sha256(models / name) != expected for name, expected in als_hashes.items()):
            raise ValueError("ALS artifacts changed during Ridge generation.")
        for name in model_files:
            (staged / name).replace(models / name)
        prediction_path.replace(predictions / prediction_file)
        (staged / CONFIG).replace(models / CONFIG)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, default="validation")
    parser.add_argument("--weighting", choices=WEIGHTINGS)
    parser.add_argument("--regularization", type=float)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--btol", type=float)
    parser.add_argument("--iter-lim", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    generate_ridge(**vars(args))
    print(f"Exported data/models/ridge_{args.split}_factors.parquet and data/predictions/ridge_{args.split}.parquet.")


if __name__ == "__main__":
    main()
