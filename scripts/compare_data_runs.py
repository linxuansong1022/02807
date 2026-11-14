"""比较“原处理结果”和“手动重新运行结果”的记录内容是否一致。

输入：data/processed/、data/samples/ 中的原结果，以及
    data/reproduced/processed/、data/reproduced/samples/ 中的重新运行结果。
输出：终端中的四项比较结果；不生成文件，也不修改任何数据。
    比较完整听歌表、歌曲信息表、2000 人副本事件、20000 人副本事件。

做法：对所有记录计算行数、内容哈希总和及异或值，不要求两份表的行顺序相同。
这些数值可理解为整张表的“内容指纹”；不同则报错停止。
这是实用的一致性检查，不是逐字节比较或密码学上的相等证明。
即使数据记录相同，Parquet 的内部保存顺序和文件字节也可能不同。
本脚本不判断清洗规则是否合理，只核对两次结果是否一致。

运行：.venv/bin/python scripts/compare_data_runs.py
"""
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]


def main():
    """读取两次结果，逐项比较，打印通过或报错。

    输入：ROOT 下两套已完成的数据文件；具体路径和列名在 comparisons 中。
    返回：无；成功时打印四项“一致”和“对比完成”。
    如果文件缺失，或行数/内容指纹不同，立即停止并显示原因。
    不比较运行耗时、内存或图片，因为它们不是表内记录内容。
    """
    db = duckdb.connect()
    db.execute("SET threads=4")
    # 每项依次是：显示名称、复跑结果的相对路径、原结果的相对路径、参与比较的列。
    comparisons = [
        ("完整听歌记录", "processed/events.parquet", "data/processed/events.parquet",
         "user_id, track_id, CAST(timestamp AS TIMESTAMP)"),
        ("歌曲信息", "processed/tracks.parquet", "data/processed/tracks.parquet",
         "track_id, tags, tag_weights, genres, genre_tfidf, has_tags, has_genres, CAST(first_seen AS TIMESTAMP)"),
        ("2000人副本", "samples/dev/events.parquet", "data/samples/dev/events.parquet",
         "user_id, track_id, CAST(timestamp AS TIMESTAMP)"),
        ("20000人副本", "samples/standard/events.parquet", "data/samples/standard/events.parquet",
         "user_id, track_id, CAST(timestamp AS TIMESTAMP)"),
    ]
    for label, rerun, original, columns in comparisons:
        paths = [ROOT / original, ROOT / "data/reproduced" / rerun]
        assert all(p.exists() for p in paths), "请先完成重新运行，缺少：" + str(paths)
        # 为两张表分别计算整表指纹；重复记录也会参与计数和哈希总和。
        signatures = []
        for path in paths:
            signatures.append(db.execute(f"""SELECT count(*),
                sum(CAST(hash({columns}) AS HUGEINT)), bit_xor(hash({columns}))
                FROM read_parquet(?)""", [str(path)]).fetchone())
        assert signatures[0] == signatures[1], f"发现差异：{label}，{signatures}"
        print(f"一致：{label}，{signatures[0][0]:,} 条；全体记录的内容指纹相同。", flush=True)
    print("对比完成。运行耗时等信息可能不同，不影响记录内容。")


if __name__ == "__main__":
    main()
