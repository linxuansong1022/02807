"""Validate C against the real preprocessed dataset and B's split."""

from collections import defaultdict
from functools import lru_cache
import unittest

import numpy as np

from scripts import retrieval, retrieval_data
from scripts.evaluate_retrieval import ListenerPool, MAX_RANK, MIN_PLAY_COUNT


DATA = retrieval_data.ROOT / "data"
SPLIT = DATA / "splits" / "split_config.json"


@unittest.skipUnless(SPLIT.is_file(), "Build the Music4All-Onion split first")
class RetrievalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with retrieval_data.connect_data(DATA) as db:
            cls.training_ids = retrieval_data.training_track_ids(db)
            cls.query_ids = retrieval_data.query_track_ids(db, "validation")
            tags = retrieval_data.read_tag_sets(
                db, min_weight=0, track_ids=cls.training_ids + cls.query_ids,
            )
        cls.training = {track: tags[track] for track in cls.training_ids}
        cls.queries = {track: tags[track] for track in cls.query_ids}

    def test_sparse_top_50_matches_scan_on_real_queries(self):
        index = retrieval.SparseJaccardIndex(self.training)
        tied_at_cutoff = False
        for track in self.query_ids[::40]:
            tags = self.queries[track]
            reference = retrieval.exact_jaccard_scan(self.training, tags, 51)
            actual = index.search(tags, 50)
            self.assertEqual(actual, reference[:50], track)
            if len(reference) == 51 and reference[49][1] == reference[50][1]:
                tied_at_cutoff = True
                self.assertLess(actual[-1][0], reference[50][0], track)
        self.assertTrue(tied_at_cutoff)

    def test_empty_and_short_positive_neighborhoods(self):
        all_tags = {**self.training, **self.queries}
        kept, _ = retrieval_data.apply_frequency_cap(all_tags, self.training_ids, .25)
        training = {t: kept[t] for t in self.training_ids}
        empty_old = {t: tags for t, tags in training.items() if not tags}
        empty_query = next(kept[t] for t in self.query_ids if not kept[t])
        self.assertTrue(empty_old)
        self.assertEqual(retrieval.jaccard(empty_query, next(iter(empty_old.values()))), 0)
        exact = retrieval.SparseJaccardIndex(training)
        self.assertEqual(exact.search(empty_query, 50), [])
        lsh = retrieval.MinHashLSH(empty_old, 32, 4)
        self.assertTrue(all(keys.size == 0 for keys, _ in lsh._bands))
        self.assertEqual(lsh.search_with_count(empty_query, 50), ([], 0))
        short = next(kept[t] for t in self.query_ids if kept[t] and 0 < len(exact.search(kept[t], 50)) < 50)
        found = exact.search(short, 50)
        self.assertEqual(found, retrieval.exact_jaccard_scan(training, short, 50))
        self.assertTrue(all(score > 0 for _, score in found))

    def test_minhash_matches_per_tag_shake_on_real_tracks(self):
        training_tags = list(self.training.values())
        vocabulary = {tag for tags in training_tags for tag in tags}
        for hashes in (128, 256):
            with self.subTest(hashes=hashes):
                priorities = lru_cache(maxsize=8_192)(retrieval._tag_priorities)
                expected = np.full((len(training_tags), hashes), np.iinfo(np.uint64).max,
                                   dtype=np.uint64)
                for signature, tags in zip(expected, training_tags, strict=True):
                    for tag in tags:
                        np.minimum(signature, priorities(tag, hashes, 42), out=signature)
                hasher = retrieval.MinHasher(vocabulary, hashes)
                np.testing.assert_array_equal(hasher.signatures(training_tags), expected)
                for track in self.query_ids[::40]:
                    query = self.queries[track]
                    np.testing.assert_array_equal(
                        hasher.signature(query), retrieval.minhash_reference(query, hashes),
                    )

    def test_short_lsh_pool_uses_unique_training_activity_fallback(self):
        with retrieval_data.connect_data(DATA) as db:
            pool = ListenerPool(db, self.training_ids)
        index = retrieval.MinHashLSH(self.training, 32, 4)
        popular, count = pool.rank([])
        self.assertEqual(count, 0)
        self.assertEqual(popular, pool.user_ids[pool.popular].tolist())
        for query in self.queries.values():
            neighbors = index.search(query, 50)
            ranked, content_count = pool.rank(neighbors)
            if 0 < content_count < MAX_RANK:
                self.assertEqual(len(ranked), MAX_RANK)
                self.assertEqual(len(set(ranked)), MAX_RANK)
                content_users = set(ranked[:content_count])
                activity_fill = [user for user in popular if user not in content_users]
                self.assertEqual(ranked[content_count:], activity_fill[:MAX_RANK - content_count])
                listeners = defaultdict(list)
                with retrieval_data.connect_data(DATA) as db:
                    rows = db.execute("""
                        SELECT track_id, user_id, play_count FROM train_interactions
                        WHERE track_id IN (SELECT unnest(?::VARCHAR[])) AND play_count >= ?
                    """, [[track for track, _ in neighbors], MIN_PLAY_COUNT]).fetchall()
                for track, user, plays in rows:
                    listeners[track].append((user, plays))
                reference = retrieval.candidate_pool(
                    neighbors, listeners, MAX_RANK, min_play_count=MIN_PLAY_COUNT,
                )
                self.assertEqual(content_users, {user for user, _ in reference})
                scores = dict(zip(pool.user_ids, pool.scores(neighbors), strict=True))
                np.testing.assert_allclose([scores[user] for user, _ in reference],
                                           [score for _, score in reference])
                break
        else:
            self.fail("Expected a real query with a nonempty pool shorter than B's maximum cutoff")


if __name__ == "__main__":
    unittest.main()
