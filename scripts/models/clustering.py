"""Weighted k-means on ALS user vectors and cluster-based listener scores (D).

The candidate users of one new track (C's candidate pool) are clustered on their
L2-normalised ALS vectors. Each user's weight is C's content score, so users who
played many close neighbours pull the centres harder. Every eligible user is
then scored by cosine similarity to the centres.
"""

from __future__ import annotations

import numpy as np

AGGREGATIONS = ("wsum", "max")


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit L2 norm; zero rows stay zero."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def weighted_kmeans(
    points: np.ndarray, weights: np.ndarray, k: int, rng: np.random.Generator, *, max_iter: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lloyd's algorithm with weighted k-means++ seeding.

    Returns centres (k × d), labels (n), and each cluster's share of the total weight.
    An empty cluster keeps its previous centre. k is capped at the number of points.
    """
    n = len(points)
    if n == 0 or k < 1:
        raise ValueError("k-means needs at least one point and one cluster.")
    if weights.shape != (n,) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Weights must be nonnegative with a positive sum, one per point.")
    k = min(k, n)
    points = points.astype(np.float64, copy=False)
    weights = weights.astype(np.float64, copy=False)

    # k-means++: first centre by weight, later ones by weight × squared distance to the nearest centre.
    centres = np.empty((k, points.shape[1]))
    centres[0] = points[rng.choice(n, p=weights / weights.sum())]
    nearest = ((points - centres[0]) ** 2).sum(axis=1)
    for j in range(1, k):
        mass = weights * nearest
        total = mass.sum()
        index = rng.choice(n, p=mass / total) if total > 0 else int(rng.integers(n))
        centres[j] = points[index]
        nearest = np.minimum(nearest, ((points - centres[j]) ** 2).sum(axis=1))

    labels = np.full(n, -1)
    squared = (points ** 2).sum(axis=1)
    for _ in range(max_iter):
        distances = squared[:, None] - 2 * points @ centres.T + (centres ** 2).sum(axis=1)[None, :]
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        mass = np.bincount(labels, weights=weights, minlength=k)
        sums = np.zeros_like(centres)
        np.add.at(sums, labels, points * weights[:, None])
        filled = mass > 0
        centres[filled] = sums[filled] / mass[filled, None]

    shares = np.bincount(labels, weights=weights, minlength=k) / weights.sum()
    return centres, labels, shares


def cluster_similarities(
    unit_users: np.ndarray, candidates: np.ndarray, weights: np.ndarray, k: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster the candidates and return (all users × clusters cosine matrix, cluster weight shares)."""
    centres, _, shares = weighted_kmeans(unit_users[candidates], weights, k, rng)
    unit_centres = normalize_rows(centres).astype(unit_users.dtype)
    return unit_users @ unit_centres.T, shares


def aggregate(similarities: np.ndarray, shares: np.ndarray, how: str) -> np.ndarray:
    """Combine per-cluster similarities into one score per user.

    wsum: share-weighted sum of cosines. It equals the cosine with a single combined
          direction, so it behaves much like k = 1.
    max:  similarity to the closest centre, so each cluster keeps its own audience.
    """
    if how == "wsum":
        return similarities @ shares.astype(similarities.dtype)
    if how == "max":
        return similarities.max(axis=1)
    raise ValueError(f"Unknown aggregation {how!r}; choose from {AGGREGATIONS}.")
