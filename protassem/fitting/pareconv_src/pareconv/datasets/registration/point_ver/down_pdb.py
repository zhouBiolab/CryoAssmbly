#!/usr/bin/env python3
"""
从 .npz 文件名前4位提取 PDB ID，下载对应的 .cif 文件。
用法: python download_pdb.py <npz文件夹> <输出文件夹> [--format pdb/cif/both]
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PDB_URLS = {
    "cif": "https://files.rcsb.org/download/{pid}.cif",
    "pdb": "https://files.rcsb.org/download/{pid}.pdb",
}


def get_pdb_ids(npz_dir: str):
    ids = set()
    for f in os.listdir(npz_dir):
        if f.endswith(".npz") and len(f) >= 4:
            ids.add(f[:4].lower())
    return sorted(ids)


def download_one(pid: str, out_dir: str, fmt: str) -> str:
    url = PDB_URLS[fmt].format(pid=pid)
    out_path = os.path.join(out_dir, f"{pid}.{fmt}")
    if os.path.exists(out_path):
        return f"[跳过] {pid}.{fmt} 已存在"
    try:
        urllib.request.urlretrieve(url, out_path)
        return f"[完成] {pid}.{fmt}"
    except Exception as e:
        return f"[失败] {pid}.{fmt} - {e}"


def main():
    parser = argparse.ArgumentParser(description="从npz文件名提取PDB ID并下载结构文件")
    parser.add_argument("npz_dir", help="包含.npz文件的文件夹")
    parser.add_argument("out_dir", help="下载文件的输出文件夹")
    parser.add_argument("--format", choices=["pdb", "cif", "both"], default="cif",
                        help="下载格式 (默认: cif)")
    parser.add_argument("--workers", type=int, default=32, help="并行下载线程数 (默认: 4)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pdb_ids = get_pdb_ids(args.npz_dir)
    print(f"找到 {len(pdb_ids)} 个PDB ID: {', '.join(pdb_ids[:10])}{'...' if len(pdb_ids) > 10 else ''}")

    fmts = ["pdb", "cif"] if args.format == "both" else [args.format]
    tasks = [(pid, args.out_dir, fmt) for pid in pdb_ids for fmt in fmts]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, *t): t for t in tasks}
        for future in as_completed(futures):
            print(future.result())

    print("全部完成!")


if __name__ == "__main__":
    main()