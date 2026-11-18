"""Prepare listening events, tags, and genres for downstream analysis.

Reads the compressed event log, tag dictionary, genre TF-IDF, and download
manifest from data/raw/. The other files in the manifest are verified too.

Writes the full source events, analysis events from 2005 onward, track
metadata, and descriptive statistics to data/processed/. The two user
samples go in data/samples/. Run summaries and figures go in reports/.
Model training and formal dataset splits are outside this workflow.

Run with `uv run python scripts/prepare_data.py`. Add
`--output-dir data/reproduced` to write processed/, samples/, and reports/
under a separate directory.
"""
from __future__ import annotations

import argparse
import ast
import bz2
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
import unicodedata

import duckdb
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


def write_json(path, value):
    """Write readable JSON to a path whose parent directory already exists."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def normalize_tag(tag):
    """Normalize the Unicode form, case, and spacing of a tag."""
    return " ".join(unicodedata.normalize("NFC", tag).lower().split())


def prepare_content(raw, out):
    """Write tag and genre features and return counts from the source files.

    The feature tables store one track-feature pair per row. Separate ID
    tables preserve tracks with no tag or nonzero genre value. Identical rows
    for a track are collapsed, and conflicting rows fail validation.
    """
    rows, ids, empty, merged = [], {}, 0, 0
    tag_rows, tag_duplicates = 0, 0
    with bz2.open(raw / "id_tags_dict.tsv.bz2", "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        assert next(reader) == ["id", "(tag, weight)"]
        for track, encoded in reader:
            tag_rows += 1
            assert track, "Empty track ID"
            tags = ast.literal_eval(encoded)
            assert isinstance(tags, dict)
            if track in ids:
                assert ast.literal_eval(ids[track]) == tags, f"Conflicting tags for {track}"
                tag_duplicates += 1
                continue
            ids[track] = encoded
            empty += not tags
            clean = {}
            for tag, weight in tags.items():
                assert isinstance(tag, str) and isinstance(weight, (float, int))
                assert math.isfinite(weight) and 0 <= weight <= 100
                key = normalize_tag(tag)
                assert key, f"Blank tag: {track}"
                merged += key in clean
                # Use the strongest weight when spelling variants normalize to the same tag.
                clean[key] = max(clean.get(key, 0), weight)
            for tag, weight in sorted(clean.items()):
                rows.append({"track_id": track, "tag": tag, "weight": float(weight)})
    schema = pa.schema([("track_id", pa.string()), ("tag", pa.string()), ("weight", pa.float64())])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out / "tags.parquet", compression="zstd")
    pq.write_table(pa.table({"track_id": sorted(ids)}), out / "tag_track_ids.parquet", compression="zstd")

    # Store the nonzero entries from the source's wide genre table.
    genres, genre_ids, zero = [], {}, 0
    genre_rows, genre_duplicates = 0, 0
    with bz2.open(raw / "id_genres_tf-idf.tsv.bz2", "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        assert header[0] == "id" and len(set(header)) == len(header)
        for row in reader:
            assert len(row) == len(header)
            track = row[0]
            genre_rows += 1
            assert track
            values = [float(x) for x in row[1:]]
            assert all(math.isfinite(x) and x >= 0 for x in values)
            digest = hashlib.sha256(repr(values).encode()).digest()
            if track in genre_ids:
                assert genre_ids[track] == digest, f"Conflicting genres for {track}"
                genre_duplicates += 1
                continue
            genre_ids[track] = digest
            zero += not any(values)
            for genre, value in zip(header[1:], values):
                if value:
                    genres.append({"track_id": track, "genre": genre, "tfidf": value})
    schema = pa.schema([("track_id", pa.string()), ("genre", pa.string()), ("tfidf", pa.float64())])
    pq.write_table(pa.Table.from_pylist(genres, schema=schema), out / "genres.parquet", compression="zstd")
    pq.write_table(pa.table({"track_id": sorted(genre_ids)}), out / "genre_track_ids.parquet", compression="zstd")
    return {"tag_source_rows": tag_rows, "identical_tag_rows_removed": tag_duplicates,
            "genre_source_rows": genre_rows, "identical_genre_rows_removed": genre_duplicates,
            "tag_file_tracks": len(ids), "empty_tag_tracks": empty,
            "normalized_tag_pairs": len(rows), "format_collisions_merged": merged,
            "genre_file_tracks": len(genre_ids), "genre_dimensions": len(header)-1,
            "zero_genre_tracks": zero}


def prepare_events(raw, out):
    """Stream the full event log to Parquet and return conversion statistics.

    The conversion preserves every row, string IDs, and timezone-naive
    timestamps. It publishes the completed temporary file by renaming it
    and can reuse a previous conversion with matching row counts.
    """
    target = out / "events_source.parquet"
    metrics = out / "event_conversion.json"
    if target.exists() and metrics.exists():
        result = json.loads(metrics.read_text())
        assert pq.ParquetFile(target).metadata.num_rows == result["rows"]
        print("Reusing complete events_source.parquet", flush=True)
        result["reused_in_this_run"] = True
        return result
    start = time.perf_counter()
    # Keep IDs as strings so leading zeros survive parsing.
    schema = pa.schema([("user_id", pa.string()), ("track_id", pa.string()),
                        ("timestamp", pa.timestamp("s"))])
    partial = target.with_suffix(".parquet.partial")
    count, next_log = 0, 10_000_000
    with bz2.open(raw / "userid_trackid_timestamp.tsv.bz2", "rb") as f:
        reader = pacsv.open_csv(f, read_options=pacsv.ReadOptions(block_size=16*1024*1024),
                               parse_options=pacsv.ParseOptions(delimiter="\t"),
                               convert_options=pacsv.ConvertOptions(column_types=schema,
                                   strings_can_be_null=True,
                                   timestamp_parsers=["%Y-%m-%d %H:%M:%S"]))
        assert reader.schema == schema, reader.schema
        with pq.ParquetWriter(partial, schema, compression="zstd") as writer:
            for batch in reader:
                assert all(c.null_count == 0 for c in batch.columns), "Null required field"
                writer.write_batch(batch)
                count += batch.num_rows
                if count >= next_log:
                    print(f"Converted {count:,} events; {time.perf_counter()-start:.0f}s", flush=True)
                    next_log += 10_000_000
    assert count > 0
    partial.replace(target)
    result = {"rows": count, "seconds": round(time.perf_counter()-start, 2),
              "parquet_bytes": target.stat().st_size, "dropped_rows": 0,
              "reused_in_this_run": False,
              "deduplicated": False, "timezone": "unspecified in source; kept timezone-naive"}
    write_json(metrics, result)
    return result


def summarize(out, samples, reports, content, conversion, expected_rows=None):
    """Build analysis tables, descriptive statistics, and deterministic user samples.

    The summary records event counts, time ranges, metadata coverage, and
    sample sizes. If supplied, expected_rows validates the original event
    count. Track first_seen values come from the entire analysis log.
    """
    db = duckdb.connect()
    db.execute("SET threads=4")
    db.execute("SET memory_limit='2GB'")
    temp = out.parent / "tmp"
    temp.mkdir(exist_ok=True)
    db.execute(f"SET temp_directory='{temp.as_posix()}'")
    def view(name, file):
        """Expose a Parquet file as a view on this connection."""
        db.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{file.as_posix()}')")
    def save(sql, path):
        """Write the query result as compressed Parquet."""
        db.execute(f"COPY ({sql}) TO '{path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    # Start the analysis log in 2005. The source conversion keeps the lone 1970 event.
    view("source_events", out / "events_source.parquet")
    raw_n, raw_lo = db.execute("SELECT count(*), min(timestamp) FROM source_events").fetchone()
    assert raw_n == conversion["rows"]
    if expected_rows is not None:
        assert raw_n == expected_rows
    save("SELECT * FROM source_events WHERE timestamp >= TIMESTAMP '2005-01-01'", out / "events.parquet")
    for name in ["events", "tags", "genres", "tag_track_ids", "genre_track_ids"]:
        view(name, out / f"{name}.parquet")
    n, users, songs, lo, hi = db.execute("""SELECT count(*), count(DISTINCT user_id),
          count(DISTINCT track_id), min(timestamp), max(timestamp) FROM events""").fetchone()
    assert 0 < n <= raw_n
    print(f"Full scan: {n:,} events, {users:,} users, {songs:,} songs, {lo} .. {hi}", flush=True)
    # Compute first plays and activity over the complete analysis log.
    save("""SELECT track_id, min(timestamp) first_seen, max(timestamp) last_seen,
         count(*) event_count, count(DISTINCT user_id) listener_count
         FROM events GROUP BY track_id""", out / "track_stats.parquet")
    save("""SELECT user_id, min(timestamp) first_seen, max(timestamp) last_seen,
         count(*) event_count, count(DISTINCT track_id) track_count
         FROM events GROUP BY user_id""", out / "user_stats.parquet")
    view("track_stats", out / "track_stats.parquet")
    view("user_stats", out / "user_stats.parquet")
    save("SELECT track_id, first_seen FROM track_stats", out / "track_first_seen.parquet")
    save("SELECT user_id, first_seen FROM user_stats", out / "user_first_seen.parquet")
    # Keep IDs from every source and represent missing content with empty lists.
    # Only positive tags populate the aligned tags and tag_weights lists in tracks.parquet.
    save("""WITH ids AS (SELECT track_id FROM track_stats UNION SELECT track_id FROM tag_track_ids
                          UNION SELECT track_id FROM genre_track_ids),
        t AS (SELECT track_id, list(tag ORDER BY tag) tags,
              list(weight ORDER BY tag) tag_weights FROM tags WHERE weight > 0 GROUP BY track_id),
        g AS (SELECT track_id, list(genre ORDER BY genre) genres,
              list(tfidf ORDER BY genre) genre_tfidf FROM genres GROUP BY track_id)
        SELECT ids.track_id, coalesce(t.tags, []::VARCHAR[]) tags,
               coalesce(t.tag_weights, []::DOUBLE[]) tag_weights,
               coalesce(g.genres, []::VARCHAR[]) genres,
               coalesce(g.genre_tfidf, []::DOUBLE[]) genre_tfidf,
               t.track_id IS NOT NULL has_tags, g.track_id IS NOT NULL has_genres,
               s.first_seen
        FROM ids LEFT JOIN t USING(track_id) LEFT JOIN g USING(track_id)
        LEFT JOIN track_stats s USING(track_id) ORDER BY ids.track_id""", out / "tracks.parquet")
    view("tracks", out / "tracks.parquet")
    coverage = db.execute("""SELECT count(*), count(*) FILTER (WHERE has_tags),
        count(*) FILTER (WHERE has_genres), count(*) FILTER (WHERE has_tags AND has_genres)
        FROM track_stats JOIN tracks USING(track_id)""").fetchone()
    save("""SELECT strftime(timestamp, '%Y-%m') AS month, count(*) event_count
        FROM events GROUP BY month ORDER BY month""", out / "monthly_events.parquet")
    save("""SELECT tag, count(*) track_count FROM tags WHERE weight > 0
        GROUP BY tag ORDER BY track_count DESC, tag""", out / "tag_frequency.parquet")
    save("""SELECT weight, count(*) tag_track_pairs FROM tags GROUP BY weight ORDER BY weight""",
         out / "tag_weight_distribution.parquet")

    # Rank user IDs by a seeded hash and keep the full histories of the selected users.
    all_users = [r[0] for r in db.execute("SELECT user_id FROM user_stats").fetchall()]
    ordered = sorted(all_users, key=lambda x: hashlib.sha256(f"42:{x}".encode()).digest())
    sample_results = {}
    for name, size in [("dev", 2000), ("standard", 20000)]:
        folder = samples / name
        folder.mkdir(parents=True, exist_ok=True)
        chosen = ordered[:size]
        pq.write_table(pa.table({"user_id": chosen, "sample_order": list(range(len(chosen)))}),
                       folder / "users.parquet")
        db.execute("DROP VIEW IF EXISTS sample_users")
        view("sample_users", folder / "users.parquet")
        save("SELECT e.* FROM events e JOIN sample_users USING(user_id)", folder / "events.parquet")
        db.execute("DROP VIEW IF EXISTS sample_events")
        view("sample_events", folder / "events.parquet")
        save("SELECT * FROM tracks WHERE track_id IN (SELECT DISTINCT track_id FROM sample_events)",
             folder / "tracks.parquet")
        sample_results[name] = dict(zip(["events", "users", "tracks"], db.execute(
            "SELECT count(*), count(DISTINCT user_id), count(DISTINCT track_id) FROM sample_events").fetchone()))
        print(f"Sample {name}: {sample_results[name]}", flush=True)

    summary = {"content": content, "events": conversion,
               "analysis_rows": n, "excluded_pre2005_events": raw_n-n,
               "source_min_timestamp": str(raw_lo), "date_floor": "2005-01-01",
               "users": users, "tracks": songs,
               "min_timestamp": str(lo), "max_timestamp": str(hi),
               "coverage": dict(zip(["event_tracks", "event_tracks_with_positive_tags",
                   "event_tracks_with_genres", "event_tracks_with_both"], coverage)),
               "samples": sample_results, "seed": 42,
               "sampling": "SHA-256(42:user_id) priority; not a final training population"}
    write_json(reports / "data_summary.json", summary)
    return summary


def figures(out, reports):
    """Plot song popularity, tag frequency, and monthly activity from full-log statistics."""
    os.environ.setdefault("MPLCONFIGDIR", str(out.parent / "tmp" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    db = duckdb.connect()
    folder = reports / "figures"
    folder.mkdir(exist_ok=True)
    counts = [x[0] for x in db.execute(f"SELECT listener_count FROM read_parquet('{out}/track_stats.parquet') ORDER BY listener_count DESC").fetchall()]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.loglog(range(1, len(counts)+1), counts)
    ax.set(xlabel="Song rank", ylabel="Distinct listeners (entire log)", title="Song popularity: descriptive full-log distribution")
    ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(folder / "song_popularity.png", dpi=160); plt.close(fig)
    top = db.execute(f"SELECT tag, track_count FROM read_parquet('{out}/tag_frequency.parquet') LIMIT 20").fetchall()[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([x[0] for x in top], [x[1] for x in top])
    ax.set(xlabel="Songs with positive tag weight (content file)", title="Most common tags: descriptive only")
    fig.tight_layout(); fig.savefig(folder / "tag_frequency.png", dpi=160); plt.close(fig)
    months = db.execute(f"SELECT * FROM read_parquet('{out}/monthly_events.parquet') ORDER BY month").fetchall()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(months)), [x[1] for x in months])
    step = max(1, len(months)//10)
    ax.set_xticks(list(range(0,len(months),step)), [x[0] for x in months[::step]], rotation=35)
    ax.set(ylabel="Listening events", title="Monthly events: entire log")
    fig.tight_layout(); fig.savefig(folder / "monthly_events.png", dpi=160); plt.close(fig)


def main():
    """Verify source files, prepare data and figures, and write run reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--expected-rows", type=int, default=252984396)
    parser.add_argument("--output-dir", type=Path,
                        help="Write processed/, samples/, reports/ under a separate directory")
    args = parser.parse_args()
    root = args.root.resolve()
    raw, out = root / "data/raw", root / "data/processed"
    samples, reports = root / "data/samples", root / "reports"
    if args.output_dir:
        destination = args.output_dir.resolve()
        out, samples, reports = [destination / name for name in ["processed", "samples", "reports"]]
    for p in [out, samples, reports]: p.mkdir(parents=True, exist_ok=True)
    print(f"Source files: {raw}", flush=True)
    print(f"Processed data: {out}", flush=True)
    start = time.perf_counter()
    # Recheck the downloaded files before preprocessing.
    manifest = json.loads((raw / "download_manifest.json").read_text())
    for item in manifest["files"]:
        h = hashlib.md5()
        with (raw / item["file"]).open("rb") as f:
            for block in iter(lambda: f.read(8*1024*1024), b""): h.update(block)
        assert h.hexdigest() == item["md5"], item["file"]
    write_json(reports / "source_manifest.json", manifest)
    content = prepare_content(raw, out)
    print(f"Content prepared: {content}", flush=True)
    conversion = prepare_events(raw, out)
    summary = summarize(out, samples, reports, content, conversion, args.expected_rows)
    figures(out, reports)
    summary["total_seconds"] = round(time.perf_counter()-start, 2)
    summary["runtime_scope"] = "this preparation process; reused event conversion time reported separately"
    import sys
    summary["peak_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                                    (1024**2 if sys.platform == "darwin" else 1024), 1)
    write_json(reports / "data_summary.json", summary)
    completion_report = reports / "processing_complete.txt"
    completion_report.write_text(
        f"Source events: {summary['events']['rows']:,}\n"
        f"Analysis events: {summary['analysis_rows']:,}\n"
        f"Events excluded before 2005: {summary['excluded_pre2005_events']:,}\n"
        f"Users: {summary['users']:,}\nTracks: {summary['tracks']:,}\n"
        f"Time range: {summary['min_timestamp']} to {summary['max_timestamp']}\n"
        f"Processed data: {out}\nUser samples: {samples}\n"
        "Formal dataset splits are outside this preprocessing run.\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Processing complete. Summary saved to {completion_report}", flush=True)


if __name__ == "__main__":
    main()
