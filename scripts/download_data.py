"""下载本项目用到的五个 Music4All-Onion v2 文件，并检查是否下载完整。

输入：下方 BASE 指定的官方地址，以及 FILES 中的文件名和官方 MD5 校验值。
输出：项目 data/raw/ 下的五个原始文件，以及 download_manifest.json 下载清单。
五个文件分别是：听歌日志、标签字典、标签数字版、流派数字版和 README 说明。
本文件只负责下载，不清洗数据，不训练模型。

运行：.venv/bin/python scripts/download_data.py
"""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
BASE = "https://zenodo.org/records/15394646"
# MD5 可以理解为文件的校验号码。下载后重算，判断文件是否与官方版本一致。
FILES = {
    "userid_trackid_timestamp.tsv.bz2": "dfe82201036765f7463e6f3ce3d0f991",
    "id_tags_dict.tsv.bz2": "9c1ca4a3e04a9696b9449ec54ac40cd5",
    "id_tags_tf-idf.tsv.bz2": "6681d5c2bf214ad54465286fbbab1ed4",
    "id_genres_tf-idf.tsv.bz2": "a742b5fa1d2e2ce780101773e57bb7f5",
    "README.md": "df379ed27dcefe5652b8db4e6783e304",
}


def checksum(path):
    """计算一个文件的 MD5 校验号码，不修改文件。

    输入：path 是本地文件路径（Path）。
    返回：由数字和字母组成的校验文字，用于与官方值比较。
    分块读取大文件，不一次把整个文件装进内存。
    """
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""): h.update(block)
    return h.hexdigest()


def main():
    """按 FILES 清单下载并检查所有文件。

    输入：模块中的官方地址 BASE、下载目录 RAW、文件清单 FILES。
    返回：无；在终端打印每个文件是否验证通过。
    输出：data/raw/ 下的五个原始文件和 download_manifest.json。
    清单记录每个文件的名称、字节大小、MD5 和验证状态。

    已存在的文件仍检查 MD5，但不重复下载。
    未完成的下载先保存为 .part，支持继续下载；校验通过后改为正式名称。
    若校验失败，程序停止，避免把不完整文件当作可用数据。
    """
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, expected in FILES.items():
        target = RAW / name
        if not target.exists():
            partial = RAW / (name + ".part")
            subprocess.run(["curl", "-L", "--fail", "--retry", "3", "--continue-at", "-",
                            f"{BASE}/files/{name}?download=1", "-o", str(partial)], check=True)
            assert checksum(partial) == expected, f"Checksum mismatch: {partial}"
            partial.replace(target)
        assert checksum(target) == expected, f"Checksum mismatch: {target}"
        manifest.append({"file": name, "bytes": target.stat().st_size,
                         "md5": expected, "verified": True})
        print(name, "verified", flush=True)
    (RAW / "download_manifest.json").write_text(json.dumps({"source": BASE, "files": manifest}, indent=2)+"\n")


if __name__ == "__main__":
    # 直接运行这个文件时，执行上面的下载流程。
    main()
