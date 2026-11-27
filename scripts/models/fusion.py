"""D: listener clustering, A + B fusion, and the ablation variants.

Scores for one new track s over all eligible users u (sorted user IDs):

  pool(u)    C's candidate-pool score: sum over the 50 tag neighbours of Jaccard × log1p(plays)
  cluster(u) cosine of u's L2-normalised ALS vector with the k-means centres of the top-N pool users
  ridge(u)   B's score: ALS user vector · Ridge-predicted track vector

  A(u)     = β · n(pool) + (1 − β) · n(cluster)
  score(u) = γ · A(u) + (1 − γ) · n(ridge)

n() normalises each score vector per track (min–max or z-score). Users are ranked by score alone;
only exactly equal scores are ordered by training activity (most active first), so users without any
signal fall back to the most active users, as in C's pool fill.

Stages (run from the repository root):
  --stage tune-a     validation only. Grid over N, k, aggregation and β with γ = 1 (A path alone).
  --stage tune-full  validation only. Fix N, k, aggregation from tune-a; grid over β, γ and n().
                     Picks the five ablation variants and saves them to selected.json.
  --stage final      validation or test. Applies the frozen variants, exports ranked predictions and
                     scores them with the shared evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import zlib
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts import retrieval, retrieval_data
from scripts.common.evaluate import RANK_CUTOFFS, evaluate
from scripts.common.features import ROOT
from scripts.common.scoring import export_predictions, top_k
from scripts.evaluate_retrieval import TOP_N, ListenerPool
from scripts.models.als import ALS_FILES, verify_als_export
from scripts.models.clustering import AGGREGATIONS, aggregate, cluster_similarities, normalize_rows
from scripts.models.ridge import CONFIG as RIDGE_CONFIG
from scripts.models.ridge import _read_factors

MAX_RANK = RANK_CUTOFFS[-1]
METRICS = [f"{name}_at_{k}" for name in ("recall", "ndcg") for k in RANK_CUTOFFS]
NORMS = ("minmax", "zscore")
VARIANTS = ("full", "no_cluster", "a_only", "b_only", "pool_only")
SEED = 42


# ---------------------------------------------------------------- data

class Context:
    """Everything shared by all tracks of one split."""

    def __init__(self, split: str, neighbors: str, bands: int, rows: int,
                 min_weight: float, max_tag_fraction: float | None, root: Path = ROOT):
        start = perf_counter()
        self.split = split
        data, models = root / "data", root / "data/models"
        with retrieval_data.connect_data(data) as db:
            train_ids = retrieval_data.training_track_ids(db)
            self.track_ids = retrieval_data.query_track_ids(db, split)
            raw = retrieval_data.read_tag_sets(db, min_weight=min_weight, track_ids=train_ids + self.track_ids)
            self.tags, excluded = retrieval_data.apply_frequency_cap(raw, train_ids, max_tag_fraction)
            self.pool = ListenerPool(db, train_ids)
            activity = dict(db.execute(
                "SELECT user_id, sum(play_count) FROM train_interactions GROUP BY user_id").fetchall())
            labels = db.execute("SELECT track_id, user_id FROM read_parquet(?) WHERE split = ?",
                                [str(data / "splits/ground_truth.parquet"), split]).fetchall()

        training = {t: self.tags[t] for t in train_ids if self.tags.get(t)}
        if neighbors == "exact":
            self.index = retrieval.SparseJaccardIndex(training)
        else:
            self.index = retrieval.MinHashLSH(training, bands, rows, seed=SEED)

        self.user_ids = self.pool.user_ids.tolist()
        row_of = {user: row for row, user in enumerate(self.user_ids)}
        plays = np.array([activity.get(user, 0) for user in self.user_ids], dtype=np.float64)
        # Tie-break order for scoring.top_k: 0 = most active user, then ascending user ID.
        order = np.lexsort((np.arange(len(plays)), -plays))
        self.activity_rank = np.empty(len(plays), dtype=np.int64)
        self.activity_rank[order] = np.arange(len(plays))

        # B's ALS users and Ridge track vectors; refuse to mix artifacts from different ALS runs.
        als, als_hashes = verify_als_export(models)
        dimension = als["factors"]
        als_users, users = _read_factors(models / ALS_FILES[1], "user_id", dimension)
        if als_users != self.user_ids:
            raise ValueError("ALS user order differs from ListenerPool.user_ids (sorted eligible users).")
        ridge_config = json.loads((models / RIDGE_CONFIG).read_text(encoding="utf-8"))
        if ridge_config["als_sha256"] != als_hashes:
            raise ValueError("Ridge was fitted on a different ALS export. Rerun scripts.models.ridge.")
        ridge_path = models / f"ridge_{split}_factors.parquet"
        if not ridge_path.is_file():
            raise FileNotFoundError(f"Missing {ridge_path}. Run: python -m scripts.models.ridge --split {split}")
        ridge_ids, ridge_vectors = _read_factors(ridge_path, "track_id", dimension)
        self.ridge = dict(zip(ridge_ids, ridge_vectors))
        missing = [t for t in self.track_ids if t not in self.ridge]
        if missing:
            raise ValueError(f"{len(missing)} {split} tracks have no Ridge vector, e.g. {missing[0]}.")
        self.users = users
        self.unit_users = normalize_rows(users)

        self.truth: dict[str, np.ndarray] = {}
        for track, user in labels:
            self.truth.setdefault(track, []).append(row_of[user])
        self.truth = {t: np.unique(rows) for t, rows in self.truth.items()}
        self.excluded_tags = sorted(excluded)
        print(f"setup {perf_counter() - start:.0f} s | {split}: {len(self.track_ids):,} tracks, "
              f"{sum(1 for t in self.track_ids if t in self.truth):,} with listeners | "
              f"{len(self.user_ids):,} users | {len(training):,} indexed training tracks | "
              f"neighbours: {neighbors} | excluded tags {self.excluded_tags}", flush=True)


class TrackScores:
    """Raw score vectors for one track, with cached cluster similarities."""

    def __init__(self, ctx: Context, track: str):
        self.ctx = ctx
        tags = ctx.tags.get(track, set())
        self.neighbors = ctx.index.search(tags, TOP_N) if tags else []
        self.pool = ctx.pool.scores(self.neighbors)
        positive = np.flatnonzero(self.pool > 0)
        self.candidates = positive[np.argsort(-self.pool[positive], kind="stable")]
        self.ridge = ctx.users @ ctx.ridge[track]
        self.seed = [SEED, zlib.crc32(track.encode())]
        self._similarities: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def cluster(self, size: int, k: int, how: str) -> np.ndarray:
        """Cluster score for all users; zeros when the track has no candidate users."""
        if len(self.candidates) == 0:
            return np.zeros(len(self.pool))
        key = (size, k)
        if key not in self._similarities:
            chosen = self.candidates[:size]
            rng = np.random.default_rng(self.seed + [size, k])
            self._similarities[key] = cluster_similarities(
                self.ctx.unit_users, chosen, self.pool[chosen], k, rng)
        similarities, shares = self._similarities[key]
        return aggregate(similarities, shares, how).astype(np.float64)


def normalize(scores: np.ndarray, how: str) -> np.ndarray:
    if how == "minmax":
        low, high = scores.min(), scores.max()
        return (scores - low) / (high - low) if high - low > 1e-12 else np.zeros_like(scores)
    if how == "zscore":
        spread = scores.std()
        return (scores - scores.mean()) / spread if spread > 1e-12 else np.zeros_like(scores)
    raise ValueError(f"Unknown normalisation {how!r}")


def component_weights(beta: float, gamma: float) -> tuple[float, float, float]:
    """(pool, cluster, ridge) weights of the final score."""
    return round(gamma * beta, 6), round(gamma * (1 - beta), 6), round(1 - gamma, 6)


def fuse(parts: dict[str, np.ndarray], weights: tuple[float, float, float]) -> np.ndarray:
    score = np.zeros(len(next(iter(parts.values()))))
    for name, weight in zip(("pool", "cluster", "ridge"), weights):
        if weight:
            score += weight * parts[name]
    return score


# ---------------------------------------------------------------- metrics

class Metrics:
    """Macro Recall@K and binary NDCG@K, matching scripts.common.evaluate."""

    def __init__(self, n_users: int):
        self.discount = 1.0 / np.log2(np.arange(2, MAX_RANK + 2))
        self.ideal = np.cumsum(self.discount)
        self.mask = np.zeros(n_users, dtype=bool)

    def track(self, ranked: np.ndarray, listeners: np.ndarray) -> np.ndarray:
        self.mask[listeners] = True
        hits = self.mask[ranked]
        self.mask[listeners] = False
        values = []
        for k in RANK_CUTOFFS:
            values.append(hits[:k].sum() / len(listeners))
        for k in RANK_CUTOFFS:
            values.append((hits[:k] * self.discount[:k]).sum() / self.ideal[min(len(listeners), k) - 1])
        return np.array(values)


class Grid:
    """Running sums of per-track metrics for many configurations."""

    def __init__(self):
        self.sums: dict[tuple, np.ndarray] = {}
        self.tracks = 0

    def add(self, key: tuple, values: np.ndarray) -> None:
        self.sums[key] = self.sums.get(key, 0) + values

    def rows(self, fields: list[str]) -> list[dict]:
        out = []
        for key, total in self.sums.items():
            row = dict(zip(fields, key))
            row.update({m: float(v) for m, v in zip(METRICS, total / self.tracks)})
            out.append(row)
        return out


def rank(score: np.ndarray, activity_rank: np.ndarray) -> np.ndarray:
    """Top 1,000 user rows by score; exact ties go to the more active user."""
    return top_k(score, MAX_RANK, activity_rank)


# ---------------------------------------------------------------- helpers

def tracks_to_run(ctx: Context, limit: int | None) -> list[str]:
    tracks = ctx.track_ids if limit is None else ctx.track_ids[:limit]
    return tracks


def progress(i: int, total: int, start: float) -> None:
    if i == total or i % 100 == 0:
        elapsed = perf_counter() - start
        print(f"  {i:,}/{total:,} tracks | {elapsed / 60:.1f} min | ~{elapsed / i * (total - i) / 60:.1f} min left",
              flush=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def best(rows: list[dict], metric: str, keep=lambda row: True) -> dict:
    candidates = [row for row in rows if keep(row)]
    if not candidates:
        raise ValueError("No configuration matches this selection.")
    return max(candidates, key=lambda row: (row[metric], row["ndcg_at_100"]))


def show(rows: list[dict], fields: list[str], metric: str, top: int = 10) -> None:
    ordered = sorted(rows, key=lambda row: -row[metric])[:top]
    print(" | ".join(f"{f:>8}" for f in fields) + " | R@100   R@500   R@1000  N@100")
    for row in ordered:
        print(" | ".join(f"{str(row[f]):>8}" for f in fields)
              + f" | {row['recall_at_100']:.4f}  {row['recall_at_500']:.4f}  "
                f"{row['recall_at_1000']:.4f}  {row['ndcg_at_100']:.4f}")


# ---------------------------------------------------------------- stages

def tune_a(ctx: Context, args, out: Path) -> None:
    """A path alone (γ = 1): which candidate-pool size, number of clusters, aggregation and β."""
    tracks = tracks_to_run(ctx, args.limit)
    metrics, grid = Metrics(len(ctx.user_ids)), Grid()
    empty = 0
    start = perf_counter()
    for i, track in enumerate(tracks, 1):
        listeners = ctx.truth.get(track)
        if listeners is None:
            progress(i, len(tracks), start)
            continue
        grid.tracks += 1
        scores = TrackScores(ctx, track)
        empty += len(scores.candidates) == 0
        pool = normalize(scores.pool, "minmax")
        grid.add(("-", "-", "-", 1.0), metrics.track(rank(fuse({"pool": pool}, (1, 0, 0)), ctx.activity_rank), listeners))
        for size in args.pool_sizes:
            for k in args.clusters:
                for how in (["wsum"] if k == 1 else args.aggregations):
                    cluster = normalize(scores.cluster(size, k, how), "minmax")
                    parts = {"pool": pool, "cluster": cluster}
                    for beta in args.betas:
                        if beta == 1:
                            continue
                        ranked = rank(fuse(parts, (beta, 1 - beta, 0)), ctx.activity_rank)
                        grid.add((size, k, how, beta), metrics.track(ranked, listeners))
        progress(i, len(tracks), start)

    fields = ["pool_size", "clusters", "aggregation", "beta"]
    rows = grid.rows(fields)
    write_csv(out / "tune_a_grid.csv", rows)
    chosen = best(rows, args.metric, lambda row: row["beta"] != 1.0)
    reference = next(row for row in rows if row["beta"] == 1.0)
    summary = {
        "metric": args.metric, "tracks": grid.tracks, "empty_pool_tracks": empty,
        "clustering": {"pool_size": chosen["pool_size"], "clusters": chosen["clusters"],
                       "aggregation": chosen["aggregation"]},
        "best_a": chosen, "pool_only": reference,
    }
    if 0.0 in args.betas:
        summary["best_cluster_only"] = best(rows, args.metric, lambda row: row["beta"] == 0.0)
    (out / "tune_a_best.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{grid.tracks:,} tracks with listeners, {empty} without any candidate user. Top configurations:")
    show(rows, fields, args.metric)
    print(f"\npool only (β = 1): {args.metric} {reference[args.metric]:.4f}")
    print(f"selected clustering: {summary['clustering']} with β = {chosen['beta']} "
          f"({args.metric} {chosen[args.metric]:.4f})")
    print(f"saved {out / 'tune_a_grid.csv'} and tune_a_best.json")


def tune_full(ctx: Context, args, out: Path) -> None:
    """Full system: β, γ and the normalisation, with the clustering fixed by tune-a."""
    clustering = json.loads((out / "tune_a_best.json").read_text(encoding="utf-8"))["clustering"]
    tracks = tracks_to_run(ctx, args.limit)
    metrics, grid = Metrics(len(ctx.user_ids)), Grid()
    start = perf_counter()
    for i, track in enumerate(tracks, 1):
        listeners = ctx.truth.get(track)
        if listeners is None:
            progress(i, len(tracks), start)
            continue
        grid.tracks += 1
        scores = TrackScores(ctx, track)
        raw = {"pool": scores.pool, "ridge": scores.ridge,
               "cluster": scores.cluster(clustering["pool_size"], clustering["clusters"], clustering["aggregation"])}
        seen = set()
        for norm in args.norms:
            parts = {name: normalize(vector, norm) for name, vector in raw.items()}
            for beta in args.betas:
                for gamma in args.gammas:
                    weights = component_weights(beta, gamma)
                    single = sum(w > 0 for w in weights) == 1     # ranking does not depend on n()
                    key = ("-" if single else norm, beta if gamma else "-", gamma) + weights
                    if key in seen:
                        continue
                    seen.add(key)
                    grid.add(key, metrics.track(rank(fuse(parts, weights), ctx.activity_rank), listeners))
        progress(i, len(tracks), start)

    fields = ["norm", "beta", "gamma", "w_pool", "w_cluster", "w_ridge"]
    rows = grid.rows(fields)
    write_csv(out / "tune_full_grid.csv", rows)
    rules = {
        "full": lambda r: r["w_pool"] > 0 and r["w_cluster"] > 0 and r["w_ridge"] > 0,
        "no_cluster": lambda r: r["w_pool"] > 0 and r["w_cluster"] == 0 and r["w_ridge"] > 0,
        "a_only": lambda r: r["w_pool"] > 0 and r["w_cluster"] > 0 and r["w_ridge"] == 0,
        "b_only": lambda r: r["w_ridge"] == 1,
        "pool_only": lambda r: r["w_pool"] == 1,
    }
    extra = {
        "cluster_only": lambda r: r["w_cluster"] == 1,
        "cluster_ridge": lambda r: r["w_pool"] == 0 and r["w_cluster"] > 0 and r["w_ridge"] > 0,
        "best_any": lambda r: True,
    }
    variants = {name: best(rows, args.metric, rule) for name, rule in rules.items()}
    others = {name: best(rows, args.metric, rule) for name, rule in extra.items()}
    selected = {
        "metric": args.metric, "neighbors": args.tag, "tracks": grid.tracks,
        "tag_rule": {"min_weight": args.min_weight, "max_tag_fraction": args.max_tag_fraction},
        "clustering": clustering, "variants": variants, "other_validation_rows": others,
    }
    (out / "selected.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(f"\n{grid.tracks:,} tracks. Top configurations:")
    show(rows, fields, args.metric)
    print("\nselected variants (validation):")
    for name, row in {**variants, **others}.items():
        print(f"  {name:<13} norm {row['norm']:<6} β {str(row['beta']):<5} γ {row['gamma']:<4} | "
              f"R@100 {row['recall_at_100']:.4f}  R@1000 {row['recall_at_1000']:.4f}  N@100 {row['ndcg_at_100']:.4f}")
    print(f"saved {out / 'tune_full_grid.csv'} and selected.json")


def final(ctx: Context, args, out: Path) -> None:
    """Frozen variants → predictions → shared evaluator."""
    source = args.params_from or out.name
    params_dir = ROOT / "reports/fusion" / source
    selected = json.loads((params_dir / "selected.json").read_text(encoding="utf-8"))
    clustering, variants = selected["clustering"], selected["variants"]
    tracks = tracks_to_run(ctx, args.limit)
    ranked = {name: np.empty((len(tracks), MAX_RANK), dtype=np.int64) for name in variants}
    kept = {name: np.empty((len(tracks), MAX_RANK)) for name in variants}
    metrics, grid = Metrics(len(ctx.user_ids)), Grid()
    start = perf_counter()
    for i, track in enumerate(tracks, 1):
        scores = TrackScores(ctx, track)
        raw = {"pool": scores.pool, "ridge": scores.ridge,
               "cluster": scores.cluster(clustering["pool_size"], clustering["clusters"], clustering["aggregation"])}
        listeners = ctx.truth.get(track)
        grid.tracks += listeners is not None
        for name, row in variants.items():
            norm = "minmax" if row["norm"] == "-" else row["norm"]
            parts = {part: normalize(vector, norm) for part, vector in raw.items()}
            score = fuse(parts, (row["w_pool"], row["w_cluster"], row["w_ridge"]))
            ranked[name][i - 1] = rank(score, ctx.activity_rank)
            kept[name][i - 1] = score[ranked[name][i - 1]]
            if listeners is not None:
                grid.add((name,), metrics.track(ranked[name][i - 1], listeners))
        progress(i, len(tracks), start)

    rows = []
    for name, row in variants.items():
        local = dict(zip(METRICS, grid.sums[(name,)] / grid.tracks))
        result = {"variant": name, "norm": row["norm"], "beta": row["beta"], "gamma": row["gamma"], **local}
        if args.limit is None:
            path = ROOT / "data/predictions" / f"fusion_{name}_{args.tag}_{ctx.split}.parquet"
            export_predictions(path, tracks, ctx.user_ids, ranked[name], kept[name])
            _, _, official = evaluate(path, ctx.split)
            for k, recall, ndcg in official:
                result[f"official_recall_at_{k}"] = recall
                result[f"official_ndcg_at_{k}"] = ndcg
            if not np.isclose(result["official_recall_at_1000"], local["recall_at_1000"], atol=1e-9):
                print(f"  warning: {name} local and official Recall@1000 differ", flush=True)
        rows.append(result)

    write_csv(out / f"final_{ctx.split}.csv", rows)
    header = "| Variant | Pool | Clusters | Ridge | Recall@100 | Recall@500 | Recall@1000 | NDCG@100 | NDCG@1000 |"
    lines = [header, "|" + " --- |" * 9]
    for result in rows:
        row = variants[result["variant"]]
        marks = ["✓" if row[w] > 0 else "" for w in ("w_pool", "w_cluster", "w_ridge")]
        pick = "official_" if args.limit is None else ""
        lines.append(f"| {result['variant']} | " + " | ".join(marks) + " | " + " | ".join(
            f"{result[pick + m]:.4f}" for m in ("recall_at_100", "recall_at_500", "recall_at_1000",
                                                 "ndcg_at_100", "ndcg_at_1000")) + " |")
    (out / f"final_{ctx.split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{ctx.split}: {grid.tracks:,} tracks with listeners | parameters from reports/fusion/{source}")
    print("\n".join(lines))
    print(f"saved {out / f'final_{ctx.split}.csv'} and .md")


# ---------------------------------------------------------------- CLI

def numbers(text: str, kind=float) -> list:
    return [kind(value) for value in text.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("tune-a", "tune-full", "final"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--neighbors", choices=("exact", "lsh"), default="exact")
    parser.add_argument("--lsh-bands", type=int, default=128)
    parser.add_argument("--lsh-rows", type=int, default=1)
    parser.add_argument("--min-weight", type=float, default=0.0, help="C's tag rule")
    parser.add_argument("--max-tag-fraction", type=float, default=0.25, help="C's tag rule (use -1 for none)")
    parser.add_argument("--metric", choices=METRICS, default="recall_at_100", help="Selection metric")
    parser.add_argument("--pool-sizes", type=lambda s: numbers(s, int), default=[500, 1000, 2000])
    parser.add_argument("--clusters", type=lambda s: numbers(s, int), default=[1, 2, 3, 5, 8])
    parser.add_argument("--aggregations", type=lambda s: s.split(","), default=list(AGGREGATIONS))
    parser.add_argument("--betas", type=numbers, default=None)
    parser.add_argument("--gammas", type=numbers, default=[round(0.1 * i, 1) for i in range(11)])
    parser.add_argument("--norms", type=lambda s: s.split(","), default=list(NORMS))
    parser.add_argument("--params-from", help="final only: reuse selected.json of another run, e.g. exact")
    parser.add_argument("--limit", type=int, help="Only the first N tracks (smoke test; no official evaluation)")
    args = parser.parse_args()

    if args.stage != "final" and args.split != "validation":
        parser.error("Tuning uses the validation split only.")
    if args.betas is None:
        args.betas = [0.0, 0.25, 0.5, 0.75, 1.0]
    if args.max_tag_fraction is not None and args.max_tag_fraction < 0:
        args.max_tag_fraction = None
    args.tag = "exact" if args.neighbors == "exact" else f"lsh{args.lsh_bands}x{args.lsh_rows}"
    out = ROOT / "reports/fusion" / (args.tag + ("_smoke" if args.limit else ""))
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "tune-full" and args.limit and not (out / "tune_a_best.json").exists():
        parser.error("Run the tune-a smoke test with the same --limit first.")

    ctx = Context(args.split, args.neighbors, args.lsh_bands, args.lsh_rows,
                  args.min_weight, args.max_tag_fraction)
    {"tune-a": tune_a, "tune-full": tune_full, "final": final}[args.stage](ctx, args, out)


if __name__ == "__main__":
    main()
