"""把原始听歌记录、标签和流派整理成后续程序可以直接读取的数据。

输入（默认都在项目的 data/raw/）：
    userid_trackid_timestamp.tsv.bz2：用户、歌曲、听歌时间。
    id_tags_dict.tsv.bz2：歌曲标签及权重。
    id_genres_tf-idf.tsv.bz2：歌曲流派的数值表示。
    download_manifest.json：下载文件的校验清单，由 download_data.py 生成。
    清单中的其他文件也会检查完整性，但标签数字版和 README 不参与数据整理。

主要输出：
    data/processed/events_source.parquet：原始听歌记录的完整转换，保留全部行。
    data/processed/events.parquet：排除 2005 年前记录后的完整听歌表。
    data/processed/tracks.parquet：每首歌的标签、权重、流派和最早出现时间。
    data/samples/：另外保存的 2,000 人、20,000 人副本，不是训练/测试划分。
    reports/：统计说明、处理完成.txt 和三张图。
    其他 Parquet 是中间表或单项统计，各函数中列出了用途。

执行顺序：检查原始文件 → prepare_content → prepare_events → summarize → figures。
本文件不训练模型，也不划分正式训练集、验证集和测试集。

运行：.venv/bin/python -u scripts/prepare_data.py
另存一份：在命令后加 --output-dir data/reproduced。
此时输出位于 data/reproduced/ 下的 processed/、samples/、reports/。
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
    """把一份 Python 统计结果保存为可读的 JSON 文本。

    输入：path 是保存路径（Path）；value 是字典、列表等统计结果。
    返回：无；在 path 写入文件，同名文件会覆盖。父文件夹须已存在。
    例如：write_json(Path('reports/counts.json'), {'users': 2000})。
    """
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def normalize_tag(tag):
    """统一一个标签的文字格式，不改变标签的含义。

    输入：tag 是标签文字，例如 '  Hard   Rock  '。
    返回：整理后的文字 'hard rock'，不生成文件。
    做法：统一 Unicode 字符表示、转小写、去首尾空格、合并重复空格。
    不把不同词语强行合并，例如 synthpop 和 synth pop 仍是两个标签。
    """
    return " ".join(unicodedata.normalize("NFC", tag).lower().split())


def prepare_content(raw, out):
    """读取并整理歌曲标签和流派。此处还不连接听歌记录。

    输入：
        raw：原始文件夹（Path），须包含标签字典和流派 TF-IDF 压缩文件。
        out：已存在的输出文件夹（Path）。
    返回：统计字典，包含读取行数、不同歌曲数、重复行数、空标签数等。
    写入 out 下的四个文件（同名覆盖）：
        tags.parquet：每行是 track_id、tag、weight，一个歌曲—标签组合。
        tag_track_ids.parquet：标签源文件出现过的全部不同歌曲编号，含空标签歌。
        genres.parquet：每行是 track_id、genre、tfidf，只保存非零流派特征。
        genre_track_ids.parquet：流派源文件出现过的全部不同歌曲编号。

    标签字典中的文字先解析为标签和权重，再统一文字格式。
    相同歌曲编号、相同内容只保留一份；内容冲突时停止，不任意选择。
    格式统一后发生重名的标签取最大权重，不相加；零权重暂时保留。
    """
    # 第一部分：整理标签，例如 S01 的 {' Rock ': 100} 变为 S01、rock、100。
    rows, ids, empty, merged = [], {}, 0, 0
    tag_rows, tag_duplicates = 0, 0
    with bz2.open(raw / "id_tags_dict.tsv.bz2", "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        assert next(reader) == ["id", "(tag, weight)"]
        for track, encoded in reader:
            tag_rows += 1
            assert track, "Empty track ID"
            tags = ast.literal_eval(encoded)  # 把字典文字读成数据，不执行文字中的代码。
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
                # 若 'Rock' 和 ' rock ' 合成一个词，取最大权重，避免人为加大权重。
                clean[key] = max(clean.get(key, 0), weight)
            for tag, weight in sorted(clean.items()):
                rows.append({"track_id": track, "tag": tag, "weight": float(weight)})
    schema = pa.schema([("track_id", pa.string()), ("tag", pa.string()), ("weight", pa.float64())])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out / "tags.parquet", compression="zstd")
    pq.write_table(pa.table({"track_id": sorted(ids)}), out / "tag_track_ids.parquet", compression="zstd")

    # 第二部分：流派原表一列一个流派。这里只保存数值非零的项，减少无用的零。
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
    """把完整压缩听歌日志转换为 Parquet，并把时间文字读成日期时间。

    输入：raw 和 out 是原始、输出文件夹（Path）。
        raw 中须有 userid_trackid_timestamp.tsv.bz2；out 须已存在。
    返回：包含原始记录数、转换耗时、输出大小等信息的字典。
    写入：out/events_source.parquet、out/event_conversion.json。
        表的三列为 user_id（文字）、track_id（文字）、timestamp（精确到秒的时间）。

    分块读取，避免一次装入全部记录。每转换约一千万条打印一次进度。
    时间按“年-月-日 时:分:秒”解析，保留源文件未指定时区的状态。
    本函数不删行；异常日期筛选在 summarize 中进行。
    先写 .partial 临时文件，全部完成后才改成正式文件名。
    如果正式文件及转换记录已存在、且行数一致，直接复用，不重新解压。
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
    # 明确每列如何读取：编号保留为文字；时间读成日期，不只是一串字符。
    schema = pa.schema([("user_id", pa.string()), ("track_id", pa.string()),
                        ("timestamp", pa.timestamp("s"))])
    partial = target.with_suffix(".parquet.partial")
    count, next_log = 0, 10_000_000
    with bz2.open(raw / "userid_trackid_timestamp.tsv.bz2", "rb") as f:
        reader = pacsv.open_csv(f, read_options=pacsv.ReadOptions(block_size=16*1024*1024),
                               parse_options=pacsv.ParseOptions(delimiter="\t"),
                               convert_options=pacsv.ConvertOptions(column_types=schema,
                                   strings_can_be_null=True,
                                   # 例如 2013-01-27 21:42:38：年、月、日、小时、分钟、秒。
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
    """从已转换的数据生成清洗版、歌曲总表、统计和用户小副本。

    输入：
        out：前两个函数输出的文件夹，含原始事件转换、标签和流派表。
        samples：小副本输出文件夹；reports：说明输出文件夹。
        content：prepare_content 返回的标签/流派统计字典。
        conversion：prepare_events 返回的转换统计字典。
        expected_rows：预期的原始事件行数；None 表示不核对指定行数。
    返回：summary 字典，记录清洗前后数量、时间范围、内容覆盖和副本规模。
    写入（已有同名结果会覆盖）：
        out/events.parquet：保留 timestamp >= 2005-01-01 的完整分析事件。
        out/tracks.parquet：一首歌一行，汇集标签、权重、流派和 first_seen。
        out/track_stats.parquet、user_stats.parquet：全时期歌曲/用户统计。
        out/track_first_seen.parquet、user_first_seen.parquet：最早出现时间。
        out/monthly_events.parquet：每月播放数。
        out/tag_frequency.parquet：每个正权重标签涉及多少首歌。
        out/tag_weight_distribution.parquet：每种标签权重出现多少次。
        samples/dev/、samples/standard/：用户清单、对应事件和歌曲信息。
        reports/data_summary.json：本阶段统计；main 会再补充耗时和内存。

    first_seen 来自完整清洗日志，不是小副本。没有播放的歌曲该值为空。
    全时期播放数等统计只供了解数据，不能直接当作训练日期前的历史。
    本函数不生成最终训练/验证/测试集合。
    """
    db = duckdb.connect()
    db.execute("SET threads=4")
    db.execute("SET memory_limit='2GB'")
    temp = out.parent / "tmp"
    temp.mkdir(exist_ok=True)
    db.execute(f"SET temp_directory='{temp.as_posix()}'")
    def view(name, file):
        """输入：临时表名 name 和 Parquet 路径 file；返回：无。

        让当前数据库能用 name 查询这个文件，不复制或修改文件。
        这里只建立本函数运行期间的查询名称，不创建新的磁盘文件。
        """
        db.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{file.as_posix()}')")
    def save(sql, path):
        """输入：查询语句 sql 和输出路径 path；返回：无。

        执行查询，并把得到的表保存成压缩 Parquet；同名输出会覆盖。
        """
        db.execute(f"COPY ({sql}) TO '{path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    # 第一步：生成分析用听歌记录。当前原数据中只有一条孤立的 1970 年记录。
    # 明确排除 2005 年前记录；events_source.parquet 仍保留全部原始事件。
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
    # 第二步：分别按歌曲、用户统计。min(timestamp) 就是最早出现时间。
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
    # 第三步：用相同的歌曲编号，把标签、流派和最早时间放到一行中。
    # 保留所有来源的歌曲编号；缺失列表填为空，不虚构标签。
    # tags 与 tag_weights 按位置对应；genres 与 genre_tfidf 同样按位置对应。
    # 可用标签列表仅取正权重，含零权重的原整理表仍保存在 tags.parquet。
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

    # 第四步：另外生成小副本，完整 events.parquet 不因此缩小。
    # 根据用户编号和固定数字 42 算出稳定的排序，再取前 2000/20000 位。
    # 每次选同一批用户，不按未来播放数筛选；保留入选用户全部分析事件。
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
    """根据已生成的统计画三张图，不修改数据。

    输入：out 下的 track_stats、tag_frequency、monthly_events 三份 Parquet。
        reports 是说明输出文件夹（Path）。
    返回：无。
    写入 reports/figures/ 下的三张 PNG 图片（同名覆盖）：
        song_popularity.png：按听众人数排序后的歌曲分布。
        tag_frequency.png：出现歌曲数最多的 20 个标签。
        monthly_events.png：每个月有多少条听歌记录。
    图中统计使用全时期数据，只用于理解数据。
    """
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
    """读取终端参数，安排整个处理过程，最后打印结果。

    输入：终端命令中的可选参数：
        --root：项目位置；默认使用当前脚本所在项目。
        --expected-rows：预期的原始事件数，默认 252984396。
        --output-dir：另存结果的位置；不指定则使用项目默认目录。
    返回：无。
    行为：检查原始文件完整性，依次调用上面的处理函数，写入统计和图片。
    额外输出：reports/source_manifest.json（来源与校验记录）、
        reports/处理完成.txt（可直接打开的数量说明）。
        reports/data_summary.json 最后补上总耗时与当前进程峰值内存。
    指定 --output-dir 后，processed/、samples/、reports/ 都放在该目录下。
    """
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
    # 先根据下载清单重新计算文件校验值；不一致就停止，避免处理损坏的文件。
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
    # 换算峰值内存：macOS 返回字节，Linux 返回 KiB，这里统一显示为 MiB。
    import sys
    summary["peak_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                                    (1024**2 if sys.platform == "darwin" else 1024), 1)
    write_json(reports / "data_summary.json", summary)
    (reports / "处理完成.txt").write_text(
        f"原始记录：{summary['events']['rows']:,} 条\n"
        f"处理后记录：{summary['analysis_rows']:,} 条\n"
        f"排除2005年前记录：{summary['excluded_pre2005_events']} 条\n"
        f"用户：{summary['users']:,} 位\n歌曲：{summary['tracks']:,} 首\n"
        f"时间：{summary['min_timestamp']} 至 {summary['max_timestamp']}\n"
        f"完整结果：{out}\n小副本：{samples}\n"
        "本次未划分训练集和测试集。\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"处理完成。可打开：{reports / '处理完成.txt'}", flush=True)


if __name__ == "__main__":
    # 在终端直接运行本文件时，从 main 开始；被别的 Python 文件导入时不自动运行。
    main()
