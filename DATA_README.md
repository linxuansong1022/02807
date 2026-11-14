# A 的数据交付

本目录提供数据准备，不训练模型，也不确定最终训练/验证/测试划分。

## 从哪里开始

1. 阅读 `reports/A_数据交付说明.md`，查看实测规模和已知限制。
2. 打开 `notebooks/01_data_preparation.ipynb`，查看真实样本、覆盖统计和图。
3. 开发时使用 `data/samples/dev/`；需要更多数据时使用 `data/samples/standard/`。

## 重新生成

在项目根目录运行。使用 Python 3.14；本次运行的直接依赖版本固定在 `requirements.txt`。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/inspect_splits.py
.venv/bin/python -m unittest discover -s tests -v
```

`download_data.py` 下载约 2.27 GB 核心文件，按官方 MD5 检查完整性。已下载且校验正确的文件会复用。

至少预留约 8 GB 磁盘用于原始文件、两份事件 Parquet、样本和临时统计。

`prepare_data.py` 流式解压收听文件并写入 Parquet，不生成巨大的中间 TSV。完成的原始事件转换文件会复用；标签、统计和样本会重新生成。若修改了事件解析规则，应先将旧 `data/processed/` 移至备份目录，再重新运行。

`inspect_splits.py` 检查三个候选时间点。它仅统计数据是否足够，不生成 `split.json`，不选择效果最好的切分。

原始文件、处理后的大文件、样本和虚拟环境都被 Git 忽略。将代码分享给队友后，队友需运行上述命令，或另行传输已处理的数据文件。不要把只包含文件名的代码仓库当作完整数据交付。

## 数据文件

- `data/processed/events_source.parquet`：原始日志的完整无损转换，保留全部 252,984,396 条事件。
- `data/processed/events.parquet`：排除 2005 年以前孤立时间记录后的分析事件。字段为 `user_id`、`track_id`、`timestamp`。ID 为字符串；时间保留源文件无时区的日期时间表示。
- `data/processed/tracks.parquet`：一首歌一行。包含标签、对应权重、流派、对应 TF-IDF、是否有可用内容、全局分析日志的 `first_seen`。未在事件中出现的歌曲，`first_seen` 为空。
- `data/processed/tags.parquet`：一行一个歌曲—标签组合，包含 `track_id`、`tag`、`weight`。
- `data/processed/genres.parquet`：只保存非零流派特征，包含 `track_id`、`genre`、`tfidf`。
- `data/processed/track_first_seen.parquet`：分析日志中每首歌的最早时间（已排除孤立的 1970 年事件），不因抽样变化。
- `data/processed/user_first_seen.parquet`：分析日志中每位用户的最早时间。
- `data/processed/track_stats.parquet`、`user_stats.parquet`：全时期统计，仅用于数据检查，不能直接用作训练特征或训练期活跃度。
- `data/processed/tag_frequency.parquet`、`tag_weight_distribution.parquet`、`monthly_events.parquet`：描述性统计，不是拟合好的训练词表。
- `data/samples/dev/`：最多 2,000 位用户的 `users.parquet`、`events.parquet`、`tracks.parquet`。
- `data/samples/standard/`：最多 20,000 位用户的同名文件。开发样本包含在这份样本中。

`tags` 与 `tag_weights` 按同一顺序排列。`genres` 与 `genre_tfidf` 也按同一顺序排列。列表为空表示无可用内容，不能根据空列表给歌曲虚构标签。

## 处理规则

- 原始转换保留全部收听事件。分析版只排除 2005 年前记录：实际仅 1 条 1970 年事件，其余最早为 2005 年。重复听歌和时间相同的事件均保留；未做全量事件去重。
- 原始标签 TF-IDF 有 178,108 个特征列，不要直接转成稠密矩阵；建议 B 从训练标签建立稀疏特征。
- 标签做 Unicode NFC、小写和空格规范化。格式合并时取最大权重，避免权重相加被人为放大。
- 同一歌曲 ID 的标签记录完全相同时，只保留一份。内容冲突时停止运行，不选择任意一条。
- `tags.parquet` 保留零权重；`tracks.parquet` 的可用标签集合只包含正权重。
- 保留缺标签和缺流派的歌曲，以便统计覆盖。B 在统一协议中确定评估歌曲范围。
- 不做人工语义合并、不删除低频歌曲、不在全库上过滤稀有标签。训练词表、IDF、阈值在 B 确定训练集后拟合。
- 官方标签 TF-IDF 原文件保留作参考，没有把其全库表示当作严格归纳实验的训练特征。

## 抽样规则与 B 的接入点

按 `SHA-256("42:" + user_id)` 排序，取前 2,000/20,000 位用户。抽样不使用播放次数或未来活跃度。保留入选用户全部时间的事件；因此样本中可能有训练截止后才出现的用户。

这些是开发与规模候选样本，不是最终可推荐用户集合。B 选择 T 后，必须按 T 前历史筛出合格用户，并将目标听众限制在同一集合。`reports/split_feasibility.csv` 已展示此类筛选后的样本量。

```python
import duckdb

db = duckdb.connect()
db.sql("SELECT * FROM read_parquet('data/samples/dev/events.parquet') LIMIT 5").show()
db.sql("SELECT track_id, tags, first_seen FROM read_parquet('data/samples/dev/tracks.parquet') LIMIT 5").show()
```

B 接手后生成训练期播放次数、最终 `split.json` 和评估函数。C 使用训练老歌的标签建立索引。D 等 B 的 ALS 向量和 C 的候选池接入聚类。

## 已知边界

源时间没有时区标记，本次未擅自标为 UTC。日志最早时间不等于发行时间。Last.fm 标签没有逐条采集时间，不能保证它们在歌曲最早播放时已存在。音频、歌词和视频不在本次核心交付范围内。
