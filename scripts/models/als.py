"""Train CPU implicit ALS and export user and track factors keyed by original IDs."""

import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import implicit
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from implicit.cpu.als import AlternatingLeastSquares

from scripts.common.features import ROOT, load_interactions

ALS_FILES = ("als_config.json", "als_user_factors.parquet", "als_track_factors.parquet")

DEFAULTS = {
    "factors": 64,
    "regularization": 0.1,
    "confidence_scale": 10.0,
    "iterations": 15,
    "random_seed": 42,
    "num_threads": 0,
}


def artifact_sha256(path: Path) -> str:
    """Return the SHA-256 digest of an artifact."""
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def verify_als_export(model_dir: Path) -> tuple[dict, dict[str, str]]:
    """Load the ALS configuration and verify the two factor files against it."""
    missing = [name for name in ALS_FILES if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete ALS export. Missing files in {model_dir}: {', '.join(missing)}")
    config_bytes = (model_dir / ALS_FILES[0]).read_bytes()
    config = json.loads(config_bytes)
    if not isinstance(config, dict) or "factor_sha256" not in config:
        raise ValueError(
            "als_config.json must contain factor_sha256. Regenerate with uv run --locked python -m scripts.models.als."
        )
    factor_hashes = {name: artifact_sha256(model_dir / name) for name in ALS_FILES[1:]}
    if config["factor_sha256"] != factor_hashes:
        raise ValueError("The ALS factor files do not match the ALS run configuration.")
    return config, {ALS_FILES[0]: sha256(config_bytes).hexdigest(), **factor_hashes}


def write_factors(
    path: Path, id_column: str, ids: list[str], vectors: np.ndarray, factors: int,
) -> None:
    """Write a fixed-size float32 vector per ID and verify the Parquet round trip."""
    if vectors.shape != (len(ids), factors) or vectors.dtype != np.float32:
        raise ValueError(f"Unexpected shape or dtype for {id_column} factors.")
    if not np.isfinite(vectors).all():
        raise ValueError(f"Nonfinite {id_column} factors.")
    if len(set(ids)) != len(ids) or any(not isinstance(value, str) for value in ids):
        raise ValueError(f"{id_column} values must be unique strings.")

    values = pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel(), type=pa.float32()), factors)
    table = pa.table({id_column: pa.array(ids, type=pa.string()), "factors": values})
    pq.write_table(table, path, compression="zstd")

    exported = pq.read_table(path)
    if exported.schema != table.schema or exported[id_column].to_pylist() != ids:
        raise ValueError(f"Exported {id_column} schema or order differs from the training input.")
    exported_values = exported["factors"].combine_chunks().values.to_numpy().reshape(vectors.shape)
    if not np.array_equal(exported_values, vectors):
        raise ValueError(f"Exported vectors differ from the trained {id_column} factors.")


def train_als(
    *,
    factors: int = DEFAULTS["factors"],
    regularization: float = DEFAULTS["regularization"],
    confidence_scale: float = DEFAULTS["confidence_scale"],
    iterations: int = DEFAULTS["iterations"],
    random_seed: int = DEFAULTS["random_seed"],
    num_threads: int = DEFAULTS["num_threads"],
    root: Path = ROOT,
    show_progress: bool = True,
) -> dict:
    """Train on the verified interaction matrix, export factors, and return the saved config."""
    if factors < 1 or iterations < 1 or random_seed < 0 or num_threads < 0:
        raise ValueError("Factors and iterations must be positive, and the seed and thread count nonnegative.")
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("Regularization must be positive and finite.")
    if not np.isfinite(confidence_scale) or confidence_scale < 0:
        raise ValueError("Confidence scaling must be nonnegative and finite.")

    interactions, user_ids, track_ids = load_interactions(root)
    confidence = interactions.copy()
    del interactions
    confidence.data *= confidence_scale
    confidence.data += 1.0
    if not np.isfinite(confidence.data).all():
        raise ValueError("Confidence weights exceed the float32 range.")

    # Positive CSR values supply both preference 1 and the full observed confidence.
    # Missing entries have preference 0 and unit confidence in the CPU solver.
    model = AlternatingLeastSquares(
        factors=factors, regularization=regularization, alpha=1.0, dtype=np.float32,
        iterations=iterations, random_state=random_seed, num_threads=num_threads,
    )
    start = perf_counter()
    model.fit(confidence, show_progress=show_progress)
    training_seconds = perf_counter() - start
    del confidence

    split = json.loads((root / "data/splits/split_config.json").read_text(encoding="utf-8"))
    config = {
        "factors": factors,
        "regularization": regularization,
        "confidence_scale": confidence_scale,
        "iterations": iterations,
        "random_seed": random_seed,
        "num_threads": num_threads,
        "device": "cpu",
        "implicit_version": implicit.__version__,
        "implicit_alpha": 1.0,
        "preference": "1 if play_count > 0, else 0",
        "confidence": f"1 + {confidence_scale:g} * play_count for observed pairs, 1 otherwise",
        "training_seconds": training_seconds,
        "training_split": {
            "dataset": split["dataset"],
            "training_cutoff_exclusive": split["training_cutoff_exclusive"],
            "eligible_users": split["counts"]["eligible_users"],
            **split["counts"]["train"],
        },
    }
    output = root / "data/models"
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as directory:
        staged = Path(directory)
        factor_files = ALS_FILES[1:]
        write_factors(staged / factor_files[0], "user_id", user_ids, model.user_factors, factors)
        write_factors(staged / factor_files[1], "track_id", track_ids, model.item_factors, factors)
        config["factor_sha256"] = {name: artifact_sha256(staged / name) for name in factor_files}
        config_path = staged / ALS_FILES[0]
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        for name in (*factor_files, ALS_FILES[0]):
            (staged / name).replace(output / name)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=int, default=DEFAULTS["factors"])
    parser.add_argument("--regularization", type=float, default=DEFAULTS["regularization"])
    parser.add_argument("--confidence-scale", type=float, default=DEFAULTS["confidence_scale"])
    parser.add_argument("--iterations", type=int, default=DEFAULTS["iterations"])
    parser.add_argument("--random-seed", type=int, default=DEFAULTS["random_seed"])
    parser.add_argument("--num-threads", type=int, default=DEFAULTS["num_threads"], help="0 uses the library default")
    config = train_als(**vars(parser.parse_args()))
    print(f"Trained ALS in {config['training_seconds']:.2f} s and exported factors to data/models/.")


if __name__ == "__main__":
    main()
