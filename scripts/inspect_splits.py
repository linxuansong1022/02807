"""检查几个候选日期下，有多少用户和新歌能满足示例评估条件。

用途：帮助队友判断数据数量是否够用；不生成正式训练集、验证集或测试集。
输入：data/processed/ 下的 events.parquet、tracks.parquet、track_stats.parquet，
    data/samples/dev/、standard/ 下的 users.parquet，以及 reports/data_summary.json。
输出：reports/split_feasibility.csv（候选日期的数量统计）、
    split_feasibility_settings.json（统计条件）、A_数据交付说明.md（交接说明）。
固定读取默认处理目录，不会自动改用 data/reproduced/。

运行：先完成 prepare_data.py，再运行 .venv/bin/python scripts/inspect_splits.py。
"""
from pathlib import Path
import csv
import datetime as dt
import json
import time

import duckdb

ROOT = Path(__file__).resolve().parents[1]


def main():
    """按示例条件统计三个候选日期，保存结果并生成交付说明。

    输入：本文件开头列出的处理结果；不接收终端参数。
    返回：无；打印每个日期在完整、2000 人、20000 人数据中的统计结果。
    输出：reports/ 下的切分规模 CSV、统计条件 JSON 和 Markdown 交付说明。

    规则：
        1. 从数据覆盖年份中选三个候选日期 T，不根据推荐模型效果选日期。
        2. 找出 T 前听过至少 20 首不同有标签歌曲的用户。
        3. 找出 T 后 180 天内首次出现、有标签的新歌；留足 30 天观察时间。
        4. 数这些新歌前 30 天有多少位上述合格听众；同一听众只计一次。
        5. 分别统计至少有 1、10、20 位合格听众的歌曲数。
    这些条件只是规模检查示例，最终实验规则仍由小组决定。
    """
    start = time.perf_counter()
    out = ROOT / "data/processed"
    db = duckdb.connect()
    db.execute("SET threads=4")
    db.execute("SET memory_limit='2GB'")
    db.execute(f"SET temp_directory='{ROOT}/data/tmp'")
    for name in ["events", "tracks", "track_stats"]:
        db.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{out}/{name}.parquet')")
    # 根据日志覆盖年份选三个候选日期，只查看样本量，不训练或评价模型。
    lo, hi = db.execute("SELECT min(first_seen), max(last_seen) FROM track_stats").fetchone()
    years = sorted(set(lo.year + round((hi.year-lo.year)*f) for f in (.4, .6, .8)))
    rows = []
    for year in years:
        cutoff = dt.datetime(year, 1, 1)
        end = min(cutoff + dt.timedelta(days=180), hi - dt.timedelta(days=30))
        if end <= cutoff: continue
        # eligible：在 T 之前已有足够听歌历史的用户。
        db.execute("""CREATE OR REPLACE TEMP TABLE eligible AS
            SELECT user_id FROM events JOIN tracks USING(track_id)
            WHERE timestamp < ? AND has_tags GROUP BY user_id
            HAVING count(DISTINCT track_id) >= 20""", [cutoff])
        # new_tracks：T 后窗口内首次出现的新歌，且必须有标签。
        db.execute("""CREATE OR REPLACE TEMP TABLE new_tracks AS
            SELECT track_id, first_seen FROM tracks
            WHERE has_tags AND first_seen >= ? AND first_seen < ?""", [cutoff, end])
        # listeners：新歌出现后 30 天内的不同听众，只统计 T 前符合条件的用户。
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
    """把实际数据规模和候选日期统计写成给队友看的中文说明。

    输入：rows 是 main 生成的统计字典列表；每项包含 T、sample、用户和歌曲数量。
        还会读取 reports/data_summary.json，取得清洗数量、覆盖、耗时和样本规模。
    返回：无。
    输出：reports/A_数据交付说明.md，同名文件会覆盖。
    只整理已有统计，不重新处理原始数据，也不训练模型。
    """
    s = json.loads((ROOT / "reports/data_summary.json").read_text())
    c, cover = s["content"], s["coverage"]
    sample_lines = "\n".join(
        f"- `{name}`：{v['users']:,} 位用户，{v['events']:,} 条事件，{v['tracks']:,} 首歌。"
        for name, v in s["samples"].items())
    cutoff_lines = "\n".join(
        f"- T={r['T']}，{r['sample']}：训练期合格用户 {r['eligible_users']:,}；"
        f"新歌中前 30 天有至少 20 位合格听众的歌曲 {r['songs_ge20_listeners']:,} 首。"
        for r in rows)
    text = f"""# A 的数据交付说明

日期：2026-11-11。原始数据来自 Music4All-Onion v2：https://zenodo.org/records/15394646 。

## 已完成

- 下载 5 个核心文件，并与官方 MD5 比对。来源及校验值见 [source_manifest.json](source_manifest.json)。
- 转换原始 {s['events']['rows']:,} 条播放事件。分析版保留 {s['analysis_rows']:,} 条。
- 统计全部 {s['users']:,} 位用户、{s['tracks']:,} 首有播放记录的歌曲。
- 分析时间范围：{s['min_timestamp']} 至 {s['max_timestamp']}。源时间不含时区，未擅自指定 UTC。
- 整理标签与流派，保存全局首次出现时间、嵌套用户样本和 3 张 EDA 图。

## 实际处理与数据覆盖

标签源文件有 {c['tag_source_rows']:,} 行，包含 {c['tag_file_tracks']:,} 个不同歌曲 ID。
合并 {c['identical_tag_rows_removed']:,} 条内容完全相同的重复歌曲记录，防止连接后重复计算播放。
不同歌曲中有 {c['empty_tag_tracks']:,} 首的标签字典为空。

{cover['event_tracks_with_positive_tags']:,} 首有播放记录的歌曲具备正权重标签，约占 {cover['event_tracks_with_positive_tags']/s['tracks']:.1%}。
{cover['event_tracks_with_genres']:,} 首具备非零流派特征。
{cover['event_tracks_with_both']:,} 首同时具备两类内容。
完整分析日志保留缺标签歌曲，B 可按统一规则筛选；本次未将它们默默丢弃。

标签只做 Unicode、小写和空格规范化。保留原始权重。
人工同义词合并、稀有标签过滤、词表和 IDF 均留待训练集确定后处理。
官方标签 TF-IDF 有 178,108 个特征列；B 应使用稀疏表示，避免读成稠密矩阵。

## 一条异常时间记录

源文件有一条 `1970-01-01 01:00:36` 的事件，用户 `20744`，歌曲 `hfyWWbpjDcX7vCbx`。
其余记录最早出现在 2005 年。
原始转换文件 `data/processed/events_source.parquet` 保留该记录。
分析文件 `events.parquet` 使用明确的 `timestamp >= 2005-01-01` 规则，仅排除这一条。
全局首次出现时间、统计和样本均基于该分析文件重新计算。
未进行全量事件去重；重复播放仍然保留。

## 可直接使用的样本

{sample_lines}

路径为 `data/samples/dev/` 和 `data/samples/standard/`，各含 `users.parquet`、`events.parquet`、`tracks.parquet`。
按 `SHA-256("42:" + user_id)` 排序抽样。开发用户包含在标准用户中。
保留入选用户的全部分析事件，不按未来活跃度选择用户。

**样本不是最终训练用户集合。** B 选择 T 后仍需按 T 前历史筛用户。

## 候选时间切分规模

以下只用于判断数据是否够用，不是最终训练/验证/测试划分。
用户条件为 T 前至少听过 20 首不同的有标签歌曲。
新歌在 T 后 180 天窗口内全局首次出现，并具备完整 30 天观察期。
目标只统计上述合格用户在新歌前 30 天的收听，不预留前十位听众。

{cutoff_lines}

2,000 人样本主要用于开发。若严格门槛下目标歌曲太少，不应用它汇报最终推荐质量。
B 可使用 20,000 人样本或完整数据，再共同确定验证/测试窗口。
不得根据测试模型表现选择最有利的时间切分。

## 字段与使用边界

- `events.parquet`：`user_id`、`track_id` 为字符串，`timestamp` 为秒级日期时间。
- `tracks.parquet`：`track_id`、`tags`、`tag_weights`、`genres`、`genre_tfidf`、`has_tags`、`has_genres`、`first_seen`。
- 标签与权重列表按位置对应。流派与其 TF-IDF 列表也按位置对应。
- `first_seen` 是清洗后全局日志的首次出现，不是歌曲发行日期，也不是样本中的首次播放。
- `track_stats`、`user_stats` 和图使用全时期数据，只能作描述；训练播放次数、活跃度必须由 T 前记录重算。
- 标签采集时间未知，不能证明零播放时标签已存在。报告应限定为“标签已知、训练中无该歌曲交互”的物品冷启动。

## 运行与交接

原始事件转换耗时 {s['events']['seconds']:.2f} 秒。
本次整理、统计和样本生成进程耗时 {s['total_seconds']:.2f} 秒，复用了原始转换文件。
该进程峰值 RSS 为 {s['peak_rss_mib']:.1f} MiB；不包含此前独立事件转换进程的峰值。
这是本机实测值，不代表其他机器性能。

运行步骤见 [DATA_README.md](../DATA_README.md)。
查看真实样本和图，打开 [01_data_preparation.ipynb](../notebooks/01_data_preparation.ipynb)。
完整统计见 [data_summary.json](data_summary.json)，切分统计见 [split_feasibility.csv](split_feasibility.csv)。

B 接手时间切分、训练期计数、ALS 和评估函数。C 接手训练老歌 LSH。D 接入 B、C 的输出。
本次 A 交付完成；最终模型和训练/验证/测试划分尚未实施。
"""
    (ROOT / "reports/A_数据交付说明.md").write_text(text)


if __name__ == "__main__":
    main()
