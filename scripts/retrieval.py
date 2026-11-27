"""Exact and MinHash retrieval for the song-to-listener candidate pool."""

from __future__ import annotations

from collections import Counter
import hashlib
import heapq
import math
from typing import AbstractSet, Iterable, Mapping, Sequence

import numpy as np
from scipy.sparse import csc_matrix


TagSets = Mapping[str, AbstractSet[str]]
Neighbors = list[tuple[str, float]]
_MAX_HASH = np.iinfo(np.uint64).max


def tag_sets_from_rows(
    rows: Iterable[tuple[str, str, float]],
    *,
    min_weight: float = 0,
    excluded_tags: AbstractSet[str] = frozenset(),
) -> dict[str, set[str]]:
    """Group A's normalized (track_id, tag, weight) rows, keeping empty sets."""
    tracks: dict[str, set[str]] = {}
    for track_id, tag, weight in rows:
        tags = tracks.setdefault(track_id, set())
        if weight >= min_weight and tag not in excluded_tags:
            tags.add(tag)
    return tracks


def frequent_tags(training_sets: TagSets, max_fraction: float) -> frozenset[str]:
    """Fit exclusions using only the supplied training songs, including empty sets."""
    if not 0 <= max_fraction <= 1:
        raise ValueError("max_fraction must be between 0 and 1")
    counts = Counter(tag for tags in training_sets.values() for tag in tags)
    return frozenset(tag for tag, count in counts.items() if count / len(training_sets) > max_fraction)


def jaccard(first: AbstractSet[str], second: AbstractSet[str]) -> float:
    overlap = len(first & second)
    union = len(first) + len(second) - overlap
    return overlap / union if union else 0.0


def _neighbor_key(neighbor: tuple[str, float]) -> tuple[float, str]:
    track_id, similarity = neighbor
    return -similarity, track_id


def exact_jaccard_scan(
    tracks: TagSets, query: AbstractSet[str], top_n: int, *, exclude_track_id: str | None = None
) -> Neighbors:
    """Reference scan with descending Jaccard and ascending ID at equal scores."""
    scored = ((track, jaccard(tags, query)) for track, tags in tracks.items() if track != exclude_track_id)
    return heapq.nsmallest(top_n, (pair for pair in scored if pair[1] > 0), key=_neighbor_key)


def top_score_rows(scores: np.ndarray, limit: int) -> np.ndarray:
    """Pick the highest scores and break ties by ascending row (ID) order."""
    if limit <= 0:
        return np.empty(0, dtype=np.intp)
    start = len(scores) - limit
    boundary = scores[np.argpartition(scores, start)[start]]
    above = np.flatnonzero(scores > boundary)
    tied = np.flatnonzero(scores == boundary)[:limit - len(above)]
    rows = np.concatenate((above, tied))
    return rows[np.lexsort((rows, -scores[rows]))]


class SparseJaccardIndex:
    """Sparse binary track × tag matrix. Scores include query tags absent from the index."""

    def __init__(self, tracks: TagSets):
        self.track_ids = sorted(tracks)
        self._track_rows = {track: row for row, track in enumerate(self.track_ids)}
        self._tag_columns = {tag: column for column, tag in enumerate(
            sorted({tag for tags in tracks.values() for tag in tags})
        )}
        self._sizes = np.array([len(tracks[track]) for track in self.track_ids], dtype=np.int32)
        row_indices = np.repeat(np.arange(len(self.track_ids), dtype=np.int32), self._sizes)
        column_indices = np.fromiter(
            (self._tag_columns[tag] for track in self.track_ids for tag in tracks[track]),
            dtype=np.int32, count=len(row_indices),
        )
        # Column storage lets the sparse multiply visit only tags in the query.
        self._matrix = csc_matrix(
            (np.ones(len(row_indices), dtype=np.int32), (row_indices, column_indices)),
            shape=(len(self.track_ids), len(self._tag_columns)),
        )

    def search(
        self, query: AbstractSet[str], top_n: int, *, exclude_track_id: str | None = None
    ) -> Neighbors:
        excluded_row = self._track_rows.get(exclude_track_id)
        limit = min(top_n, len(self.track_ids) - (excluded_row is not None))
        if limit <= 0 or not query:
            return []
        columns = sorted(self._tag_columns[tag] for tag in query if tag in self._tag_columns)
        vector = csc_matrix(
            (np.ones(len(columns), dtype=np.int32), columns, [0, len(columns)]),
            shape=(len(self._tag_columns), 1),
        )
        overlap = (self._matrix @ vector).toarray().ravel()
        union = self._sizes + len(query) - overlap
        scores = np.divide(overlap, union, out=np.zeros(len(union)), where=union > 0)
        if excluded_row is not None:
            scores[excluded_row] = 0

        rows = top_score_rows(scores, min(limit, int(np.count_nonzero(scores))))
        return [(self.track_ids[row], float(scores[row])) for row in rows]


def _tag_priorities(tag: str, num_hashes: int, seed: int) -> np.ndarray:
    digest = hashlib.shake_256(f"{seed}:{tag}".encode("utf-8")).digest(8 * num_hashes)
    return np.frombuffer(digest, dtype="<u8")


def minhash_reference(tags: AbstractSet[str], num_hashes: int, seed: int = 42) -> np.ndarray:
    """Compute each tag's SHAKE-256 priorities on demand as a reference."""
    signature = np.full(num_hashes, _MAX_HASH, dtype=np.uint64)
    for tag in tags:
        np.minimum(signature, _tag_priorities(tag, num_hashes, seed), out=signature)
    return signature


class MinHasher:
    """Precompute the same SHAKE-256 priorities once for the training vocabulary."""

    def __init__(self, vocabulary: AbstractSet[str], num_hashes: int, seed: int = 42):
        if num_hashes < 1:
            raise ValueError("num_hashes must be positive")
        self.num_hashes = num_hashes
        self.seed = seed
        self._positions = {tag: row for row, tag in enumerate(sorted(vocabulary))}
        self._priorities = np.empty((len(self._positions), num_hashes), dtype=np.uint64)
        for tag, row in self._positions.items():
            self._priorities[row] = _tag_priorities(tag, num_hashes, seed)

    def signature(self, tags: AbstractSet[str]) -> np.ndarray:
        known = []
        signature = np.full(self.num_hashes, _MAX_HASH, dtype=np.uint64)
        for tag in tags:
            position = self._positions.get(tag)
            if position is None:
                np.minimum(signature, _tag_priorities(tag, self.num_hashes, self.seed), out=signature)
            else:
                known.append(position)
        if known:
            np.minimum(signature, np.minimum.reduce(self._priorities[known], axis=0), out=signature)
        return signature

    def signatures(self, tag_sets: Sequence[AbstractSet[str]]) -> np.ndarray:
        """Reduce flattened tag IDs in batches so the gather stays bounded in memory."""
        result = np.full((len(tag_sets), self.num_hashes), _MAX_HASH, dtype=np.uint64)
        max_tag_ids = max(1, (32 * 1024**2) // (self.num_hashes * 8))
        track_rows, offsets, tag_ids = [], [], []

        def reduce_batch() -> None:
            if tag_ids:
                gathered = self._priorities[tag_ids]
                minima = np.minimum.reduceat(gathered, offsets, axis=0)
                result[track_rows] = np.minimum(result[track_rows], minima)
                track_rows.clear()
                offsets.clear()
                tag_ids.clear()

        for row, tags in enumerate(tag_sets):
            if len(tag_ids) + len(tags) > max_tag_ids:
                reduce_batch()
            start = len(tag_ids)
            for tag in tags:
                position = self._positions.get(tag)
                if position is None:
                    np.minimum(
                        result[row], _tag_priorities(tag, self.num_hashes, self.seed), out=result[row]
                    )
                else:
                    tag_ids.append(position)
            if len(tag_ids) > start:
                track_rows.append(row)
                offsets.append(start)
        reduce_batch()
        return result


class MinHashLSH:
    """Index equal MinHash bands and rerank their union by exact Jaccard."""

    def __init__(
        self, tracks: TagSets, bands: int, rows: int, *, seed: int = 42,
        vocabulary: AbstractSet[str] | None = None,
    ):
        if bands < 1 or rows < 1:
            raise ValueError("bands and rows must be positive")
        self.track_ids = sorted(tracks)
        self.tracks = tracks
        self.rows = rows
        self._band_dtype = np.dtype(f"V{8 * rows}")
        # Empty sets share a sentinel signature. They cannot yield a useful LSH match.
        indexed = np.array(
            [row for row, track in enumerate(self.track_ids) if tracks[track]], dtype=np.int32
        )
        indexed_sets = [tracks[self.track_ids[row]] for row in indexed]
        if vocabulary is None:
            vocabulary = {tag for tags in indexed_sets for tag in tags}
        self._hasher = MinHasher(vocabulary, bands * rows, seed)
        signatures = self._hasher.signatures(indexed_sets)
        self._bands = []
        for band in range(bands):
            start = band * rows
            keys = np.ascontiguousarray(signatures[:, start:start + rows]).view(self._band_dtype).ravel()
            order = np.argsort(keys, kind="stable")
            self._bands.append((keys[order], indexed[order]))

    def search_with_count(
        self, query: AbstractSet[str], top_n: int, *, exclude_track_id: str | None = None
    ) -> tuple[Neighbors, int]:
        if not query:
            return [], 0
        query_bands = self._hasher.signature(query).view(self._band_dtype)
        matches = np.zeros(len(self.track_ids), dtype=bool)
        for key, (keys, row_ids) in zip(query_bands, self._bands):
            left = keys.searchsorted(key, side="left")
            right = keys.searchsorted(key, side="right")
            matches[row_ids[left:right]] = True
        candidates = [self.track_ids[row] for row in np.flatnonzero(matches)]
        scored = ((track, jaccard(self.tracks[track], query)) for track in candidates
                  if track != exclude_track_id)
        count = len(candidates) - (exclude_track_id in candidates)
        return heapq.nsmallest(top_n, (pair for pair in scored if pair[1] > 0), key=_neighbor_key), count

    def search(
        self, query: AbstractSet[str], top_n: int, *, exclude_track_id: str | None = None
    ) -> Neighbors:
        return self.search_with_count(query, top_n, exclude_track_id=exclude_track_id)[0]


def candidate_probability(similarity: float | np.ndarray, bands: int, rows: int):
    """Probability of a MinHash band match as a function of Jaccard similarity."""
    with np.errstate(divide="ignore"):
        return -np.expm1(bands * np.log1p(-np.asarray(similarity) ** rows))


def candidate_pool(
    neighbors: Neighbors,
    listeners_by_track: Mapping[str, Sequence[tuple[str, int]]],
    top_m: int,
    *,
    min_play_count: int,
) -> Neighbors:
    """Rank listeners by weighted log play counts from B's eligible training data."""
    if min_play_count < 1:
        raise ValueError("min_play_count must be positive")
    scores: dict[str, float] = {}
    for track, similarity in neighbors:
        if similarity <= 0:
            continue
        for user, count in listeners_by_track.get(track, ()):
            if count >= min_play_count:
                scores[user] = scores.get(user, 0.0) + similarity * math.log1p(count)
    return heapq.nsmallest(top_m, scores.items(), key=_neighbor_key)
