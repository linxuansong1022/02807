"""Download and verify the five Music4All-Onion v2 source files.

The files and download_manifest.json are stored in data/raw/.
Run with uv run python scripts/download_data.py.
"""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
BASE = "https://zenodo.org/records/15394646"
FILES = {
    "userid_trackid_timestamp.tsv.bz2": "dfe82201036765f7463e6f3ce3d0f991",
    "id_tags_dict.tsv.bz2": "9c1ca4a3e04a9696b9449ec54ac40cd5",
    "id_tags_tf-idf.tsv.bz2": "6681d5c2bf214ad54465286fbbab1ed4",
    "id_genres_tf-idf.tsv.bz2": "a742b5fa1d2e2ce780101773e57bb7f5",
    "README.md": "df379ed27dcefe5652b8db4e6783e304",
}


def checksum(path):
    """Return the MD5 digest without loading the entire file into memory."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""): h.update(block)
    return h.hexdigest()


def main():
    """Verify cached or newly downloaded files and write their manifest.

    Downloads can resume from .part files, which are renamed after their
    checksums match the published values.
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
    main()
