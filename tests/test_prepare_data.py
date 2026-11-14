"""Tiny end-to-end fixture: preserve plays, global first_seen, content joins and ID samples."""
import bz2
import importlib.util
from pathlib import Path
import tempfile
import unittest

import pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location("prepare", Path(__file__).parents[1] / "scripts/prepare_data.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PreparationTest(unittest.TestCase):
    def test_lossless_events_and_missing_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, out, samples, reports = [root / x for x in ["raw", "processed", "samples", "reports"]]
            for p in [raw, out, samples, reports]: p.mkdir()
            def source(name, text):
                (raw / name).write_bytes(bz2.compress(text.encode()))
            source("id_tags_dict.tsv.bz2", "id\t(tag, weight)\nS1\t{' Rock ': 20, 'rock': 10}\nS2\t{}\nS4\t{'jazz': 50}\nS2\t{}\n")
            source("id_genres_tf-idf.tsv.bz2", "id\trock\tjazz\nS1\t2.0\t0\nS2\t0\t0\nS4\t0\t3\nS1\t2.0\t0\n")
            source("userid_trackid_timestamp.tsv.bz2", "user_id\ttrack_id\ttimestamp\n"
                   "001\tS1\t2020-02-03 12:00:00\n"
                   "001\tS1\t2020-02-03 12:00:00\n"
                   "002\tS1\t2020-01-01 00:00:00\n"
                   "002\tS2\t2020-03-01 00:00:00\n"
                   "003\tS3\t2020-04-01 00:00:00\n"
                   "003\tS1\t1970-01-01 01:00:36\n")
            content = prepare.prepare_content(raw, out)
            self.assertEqual(content["identical_tag_rows_removed"], 1)
            self.assertEqual(content["identical_genre_rows_removed"], 1)
            conversion = prepare.prepare_events(raw, out)
            summary = prepare.summarize(out, samples, reports, content, conversion, 6)
            self.assertEqual(conversion["rows"], 6)  # Source copy retains every event.
            self.assertEqual(summary["analysis_rows"], 5)
            self.assertEqual(summary["excluded_pre2005_events"], 1)
            events = pq.read_table(out / "events.parquet").to_pylist()
            self.assertEqual(events[0]["user_id"], "001")
            tracks = {x["track_id"]: x for x in pq.read_table(out / "tracks.parquet").to_pylist()}
            self.assertEqual(str(tracks["S1"]["first_seen"]), "2020-01-01 00:00:00")
            self.assertEqual(tracks["S1"]["tags"], ["rock"])
            self.assertEqual(tracks["S1"]["tag_weights"], [20])
            self.assertFalse(tracks["S2"]["has_tags"])
            self.assertFalse(tracks["S3"]["has_tags"])
            self.assertIsNone(tracks["S4"]["first_seen"])
            self.assertEqual(summary["coverage"]["event_tracks_with_positive_tags"], 1)
            self.assertEqual(summary["samples"]["dev"]["events"], 5)
            first_sample = (samples / "dev/users.parquet").read_bytes()
            prepare.summarize(out, samples, reports, content, conversion, 6)
            self.assertEqual(first_sample, (samples / "dev/users.parquet").read_bytes())


if __name__ == "__main__":
    unittest.main()
