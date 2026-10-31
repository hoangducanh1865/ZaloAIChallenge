"""Download a public Google Drive folder recursively (gdown --folder stops at 50 entries per folder).

Usage: python scripts/fetch_drive_folder.py <folder_id> <out_dir> [--workers 16]
Lists folders through drive.google.com/embeddedfolderview (no 50-entry cap); skips files already on disk.

Author: Duc Anh Hoang
"""
import argparse
import html
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ENTRY = re.compile(
    r'<div class="flip-entry" id="entry-([\w-]+)".*?<a href="([^"]+)".*?<div class="flip-entry-title">([^<]*)</div>', re.S
)
SKIP = {".DS_Store", "__pycache__"}


def get(url, **kw):
    for i in range(5):
        try:
            r = requests.get(url, timeout=120, **kw)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(2 ** i)
    raise RuntimeError(f"failed: {url}")


def list_folder(fid):
    page = get(f"https://drive.google.com/embeddedfolderview?id={fid}").text
    for eid, href, title in ENTRY.findall(page):
        yield eid, html.unescape(title), "/folders/" in href


def walk(root_id, out, ex):
    """[(file id, path)] of the whole tree, listing every level of folders in parallel."""
    files, level = [], [(root_id, out)]
    while level:
        nxt = []
        for (fid, path), entries in zip(level, ex.map(lambda x: list(list_folder(x[0])), level)):
            for eid, name, is_dir in entries:
                if name not in SKIP:
                    (nxt if is_dir else files).append((eid, path / name))
        level = nxt
        print(f"listed {len(files)} files, {len(level)} folders to list", flush=True)
    return files


def download(item):
    eid, path = item
    if path.exists() and path.stat().st_size > 0:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    r = get("https://drive.usercontent.google.com/download", params={"id": eid, "export": "download", "confirm": "t"})
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(r.content)
    tmp.rename(path)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder_id")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    with ThreadPoolExecutor(a.workers) as ex:
        files = walk(a.folder_id, a.out_dir, ex)
        done = sum(ex.map(download, files))
    print(f"downloaded {done} new, {len(files) - done} already present", flush=True)


if __name__ == "__main__":
    main()
