"""Check the feasibility of candidate time-based evaluation windows.

Reads the default processed tables, user samples, and reports/data_summary.json.
Writes the feasibility CSV, settings, and data handoff under reports/.

Run with uv run python scripts/inspect_splits.py after preprocessing.
"""
from pathlib import Path
import csv
import datetime as dt
import json
import time

import duckdb

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Count eligible users and new tracks for candidate cutoff dates.

    Candidates are based on the years covered by the log. Each window
    includes only the first 180 days after the cutoff and requires 30 days
    of follow-up per track. Eligible users heard at least 20 distinct
    tagged tracks before the cutoff. The CSV counts tracks with at least
    1, 10, or 20 distinct eligible listeners in the follow-up window.
    """
    start = time.perf_counter()
    out = ROOT / "data/processed"
    db = duckdb.connect()
    db.execute("SET threads=4")
    db.execute("SET memory_limit='2GB'")
    db.execute(f"SET temp_directory='{ROOT}/data/tmp'")
    for name in ["events", "tracks", "track_stats"]:
        db.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{out}/{name}.parquet')")
    lo, hi = db.execute("SELECT min(first_seen), max(last_seen) FROM track_stats").fetchone()
    years = sorted(set(lo.year + round((hi.year-lo.year)*f) for f in (.4, .6, .8)))
    rows = []
    for year in years:
        cutoff = dt.datetime(year, 1, 1)
        end = min(cutoff + dt.timedelta(days=180), hi - dt.timedelta(days=30))
        if end <= cutoff: continue
        db.execute("""CREATE OR REPLACE TEMP TABLE eligible AS
            SELECT user_id FROM events JOIN tracks USING(track_id)
            WHERE timestamp < ? AND has_tags GROUP BY user_id
            HAVING count(DISTINCT track_id) >= 20""", [cutoff])
        db.execute("""CREATE OR REPLACE TEMP TABLE new_tracks AS
            SELECT track_id, first_seen FROM tracks
            WHERE has_tags AND first_seen >= ? AND first_seen < ?""", [cutoff, end])
        db.execute("""CREATE OR REPLACE TEMP TABLE listeners AS
            SELECT DISTINCT e.track_id, e.user_id FROM events e
            JOIN new_tracks n USING(track_id) JOIN eligible u USING(user_id)
            WHERE e.timestamp >= n.first_seen
              AND e.timestamp < n.first_seen + INTERVAL 30 DAY""")
        for name in ["full", "dev", "standard"]:
            condition = "TRUE" if name == "full" else (
                f"user_id IN (SELECT user_id FROM read_parquet('{ROOT}/data/samples/{name}/users.parquet'))")
            n_users = db.execute(f"SELECT count(*) FROM eligible WHERE {condition}").fetchone()[0]
            counts = db.execute(f"""SELECT count(*) FILTER (WHERE n >= 1),
                    count(*) FILTER (WHERE n >= 10), count(*) FILTER (WHERE n >= 20)
                    FROM (SELECT track_id, count(*) n FROM listeners WHERE {condition} GROUP BY track_id)""").fetchone()
            row = {"T": str(cutoff.date()), "song_window_end_exclusive": str(end.date()),
                   "observation_days": 30, "sample": name, "eligible_users": n_users,
                   "tagged_new_tracks": db.execute("SELECT count(*) FROM new_tracks").fetchone()[0],
                   "songs_ge1_listener": counts[0], "songs_ge10_listeners": counts[1],
                   "songs_ge20_listeners": counts[2]}
            rows.append(row)
            print(row, flush=True)
    with (ROOT / "reports/split_feasibility.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    (ROOT / "reports/split_feasibility_settings.json").write_text(json.dumps({
        "purpose": "descriptive candidate grid, NOT a chosen train/validation/test split",
        "train_user_rule": "at least 20 distinct tagged songs before T",
        "new_song_rule": "global first_seen in [T, window_end), has positive tags",
        "target_rule": "distinct eligible users in [song first_seen, first_seen+30 days)",
        "years": years, "seconds": round(time.perf_counter()-start,2)}, indent=2)+"\n")
    write_handoff(rows)


def write_handoff(rows):
    """Summarize data preparation and split feasibility for downstream users."""
    s = json.loads((ROOT / "reports/data_summary.json").read_text())
    c, cover = s["content"], s["coverage"]
    sample_lines = "\n".join(
        f"- `{name}` has {v['users']:,} users and {v['events']:,} events."
        for name, v in s["samples"].items())
    cutoff_lines = "\n".join(
        f"- {r['T']} ({r['sample']}) has {r['eligible_users']:,} eligible users and "
        f"{r['songs_ge20_listeners']:,} new tracks with at least 20 eligible listeners in 30 days."
        for r in rows)
    text = f"""# Data preprocessing handoff

The source is Music4All-Onion v2. The [source manifest](source_manifest.json) records the files and their verified MD5 checksums.

## Listening events and content

The lossless source conversion contains {s['events']['rows']:,} events. The analysis log excludes {s['excluded_pre2005_events']:,} events dated before 2005 and preserves repeated plays. It contains {s['analysis_rows']:,} events for {s['users']:,} users and {s['tracks']:,} tracks. The analysis spans {s['min_timestamp']} to {s['max_timestamp']}. Source timestamps have no specified timezone.

The tag source contains {c['tag_source_rows']:,} rows for {c['tag_file_tracks']:,} distinct track IDs. Preprocessing collapsed {c['identical_tag_rows_removed']:,} identical duplicate rows. Of the distinct tracks in the tag source, {c['empty_tag_tracks']:,} have empty tag dictionaries.

Among tracks with listening events, {cover['event_tracks_with_positive_tags']:,} ({cover['event_tracks_with_positive_tags']/s['tracks']:.1%}) have positive-weight tags, {cover['event_tracks_with_genres']:,} have nonzero genre features, and {cover['event_tracks_with_both']:,} have both. The analysis log retains tracks with missing content.

Tag normalization standardizes Unicode, case, and spacing. If several tags normalize to the same label, the largest weight is retained. Fit any vocabulary, rare-tag filter, or IDF on the training set once the split is chosen. The source tag TF-IDF has 178,108 columns and should be handled as sparse data.

## User samples

{sample_lines}

Each sample contains `users.parquet`, `events.parquet`, and `tracks.parquet`. Users are ranked by `SHA-256("42:" + user_id)`. The development sample is a subset of the standard sample. Each sample keeps the full analysis history of its selected users. Apply the pre-cutoff eligibility rule when building the final training population.

## Candidate time windows

Each candidate window spans up to 180 days after the cutoff and includes tagged tracks based on their global first appearance. An eligible user has heard at least 20 distinct tagged tracks before the cutoff. The CSV counts distinct eligible listeners during each new track's first 30 days.

{cutoff_lines}

Use these counts to assess feasibility before choosing a final split. The development sample can have too few target tracks for evaluation at strict listener thresholds.

## Using the tables

- `events.parquet` has string `user_id` and `track_id` columns and second-resolution `timestamp` values.
- `tracks.parquet` stores tags and weights, genres and TF-IDF values, content flags, and `first_seen`. Each feature list lines up with its weight list.
- `first_seen` is the earliest play in the full analysis log. Use it to determine a track's first appearance in the data.
- Calculate training activity and popularity from pre-cutoff events. The provided activity tables and figures describe the full analysis period.
- Tag collection times are unknown. Interpret the cold-start setup as known tags for tracks without training interactions.

## Run details

The source conversion recorded {s['events']['seconds']:.2f} seconds. The current preparation process took {s['total_seconds']:.2f} seconds with a peak RSS of {s['peak_rss_mib']:.1f} MiB. The [data summary](data_summary.json) records whether the process reused the source conversion.

See the [README](../README.md) for reproduction instructions and the [feasibility CSV](split_feasibility.csv) for the full candidate grid.
"""
    (ROOT / "reports/data_handoff.md").write_text(text)


if __name__ == "__main__":
    main()
