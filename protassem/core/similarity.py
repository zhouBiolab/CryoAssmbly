"""USalign wrapper for TM-score / sequence-identity calculation.

老卡收口 P5：

- **失败不再伪装成 0.0**：超时、非零退出、解析失败一律抛 `USalignError`（带两个输入与 stderr 摘要）；
  合法的 `TM-score = 0` 是正常返回值，且允许缓存。调用点若允许失败，必须在该调用点显式处理。
- **两层缓存同键**：进程内层与 SQLite 层都按 `(结构内容指纹, USalign 指纹, 参数, 版本)` 建键，
  禁止"先命中只按路径建的内存缓存、再查库"。
- **父进程负责查询与写入**；worker 只跑 USalign（`usalign_pair`）。
- 默认库位：`$XDG_CACHE_HOME/protassem/tm.sqlite3`，未设置该变量时用 `~/.cache/protassem/tm.sqlite3`；
  `RuntimeConfig.tm_cache` = `"auto"` / `"off"` / 显式路径。
"""

import hashlib
import os
import re
import sqlite3
import subprocess
import time

USALIGN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "USalign")

TM_CACHE_VERSION = 1
TM_USALIGN_ARGS = ("-TMscore", "7", "-ter", "0")
_USALIGN_TIMEOUT_S = 300


class USalignError(RuntimeError):
    """USalign 执行失败（超时/非零退出/解析失败）——不返回伪分数。"""


def default_tm_cache_path():
    """默认库位：XDG 优先，其次 ~/.cache。"""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"),
                                                            ".cache")
    return os.path.join(base, "protassem", "tm.sqlite3")


# ----------------------------------------------------------------------
# 指纹
# ----------------------------------------------------------------------

_FILE_FINGERPRINTS = {}


def structure_fingerprint(path):
    """结构内容指纹：sha256；按 (path, size, mtime_ns) 记忆化避免重复读盘。"""
    absolute = os.path.abspath(str(path))
    stat = os.stat(absolute)
    cache_key = (absolute, stat.st_size, stat.st_mtime_ns)
    cached = _FILE_FINGERPRINTS.get(cache_key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with open(absolute, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    value = "%s:%d:%s" % (absolute, stat.st_size, digest.hexdigest())
    _FILE_FINGERPRINTS.clear()          # 只保留最近一批，避免无界增长
    _FILE_FINGERPRINTS[cache_key] = value
    return value


def usalign_fingerprint(usalign_path=None):
    """工具指纹：路径 + size + mtime_ns + 内容 sha256（二进制或脚本变更必须失效）。"""
    path = os.path.abspath(str(usalign_path or USALIGN_PATH))
    stat = os.stat(path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "%s:%d:%s" % (path, stat.st_size, digest.hexdigest()[:32])


def pair_key(pdb1, pdb2, usalign_path=None):
    """对称的缓存键：两个结构指纹排序后 + 工具指纹 + 参数 + 版本。"""
    first, second = sorted((structure_fingerprint(pdb1), structure_fingerprint(pdb2)))
    return "|".join([str(TM_CACHE_VERSION), first, second,
                     usalign_fingerprint(usalign_path), " ".join(TM_USALIGN_ARGS)])


# ----------------------------------------------------------------------
# USalign 调用（worker 与父进程共用；不做缓存）
# ----------------------------------------------------------------------

def usalign_pair(pdb1, pdb2, usalign_path=None):
    """跑一次 USalign，返回 `(tm, seqid)`；失败抛 `USalignError`。"""
    path = str(usalign_path or USALIGN_PATH)
    command = [path, str(pdb1), str(pdb2)] + list(TM_USALIGN_ARGS)
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=_USALIGN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise USalignError("USalign 超时（%ds）：%s vs %s"
                           % (_USALIGN_TIMEOUT_S, pdb1, pdb2))
    except OSError as exc:
        raise USalignError("USalign 无法执行（%s）：%s" % (path, exc))
    if result.returncode != 0:
        raise USalignError("USalign 退出码 %d：%s vs %s\nstderr: %s"
                           % (result.returncode, pdb1, pdb2,
                              (result.stderr or "").strip()[:500]))
    return _parse_usalign(result.stdout, pdb1, pdb2)


def _parse_usalign(stdout, pdb1, pdb2):
    tm1 = re.search(r"TM-score=\s*([\d.]+) \(normalized by length of Structure_1",
                    stdout)
    tm2 = re.search(r"TM-score=\s*([\d.]+) \(normalized by length of Structure_2",
                    stdout)
    seqid = re.search(r"Seq_ID=n_identical/n_aligned=\s*([\d.]+)", stdout)
    if not (tm1 and tm2):
        raise USalignError("USalign 输出无法解析 TM-score：%s vs %s\nstdout: %s"
                           % (pdb1, pdb2, (stdout or "").strip()[:500]))
    tm = min(float(tm1.group(1)), float(tm2.group(1)))
    return tm, (float(seqid.group(1)) if seqid else 0.0)


# ----------------------------------------------------------------------
# SQLite 缓存（父进程查询/写入）
# ----------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tm_scores (
    key TEXT PRIMARY KEY,
    tm REAL NOT NULL,
    seqid REAL NOT NULL,
    created_at REAL NOT NULL
)
"""


class TMStore:
    """SQLite 缓存；只由父进程打开。查询命中区分"缺失"与合法 0.0。"""

    def __init__(self, path):
        self.path = str(path)
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(_SCHEMA)
        self.connection.commit()
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def get(self, key):
        row = self.connection.execute(
            "SELECT tm, seqid FROM tm_scores WHERE key = ?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return {"tm": float(row[0]), "seqid": float(row[1])}

    def put(self, key, value):
        self.connection.execute(
            "INSERT OR REPLACE INTO tm_scores (key, tm, seqid, created_at) "
            "VALUES (?, ?, ?, ?)",
            (key, float(value["tm"]), float(value["seqid"]), time.time()))
        self.connection.commit()
        self.writes += 1

    def count(self):
        return int(self.connection.execute(
            "SELECT COUNT(*) FROM tm_scores").fetchone()[0])

    def snapshot(self):
        return {"tm_cache": os.path.basename(self.path), "path": self.path,
                "hits": self.hits, "misses": self.misses, "writes": self.writes,
                "rows": self.count()}

    def close(self):
        try:
            self.connection.close()
        except Exception:
            pass


_STORE = None
_STORE_STATE = {"mode": "off", "path": None}
_MEMORY = {}


def configure_tm_cache(tm_cache="auto"):
    """配置 TM 缓存：`"auto"`（默认库位）/ `"off"` / 显式路径。

    配置变化时**清空进程内层并重建存储**，避免沿用上一次的预算或路径。
    """
    global _STORE
    if _STORE is not None:
        _STORE.close()
        _STORE = None
    _MEMORY.clear()
    mode = "auto" if tm_cache is None else str(tm_cache)
    if mode == "off":
        _STORE_STATE.update({"mode": "off", "path": None})
        return tm_cache_snapshot()
    path = default_tm_cache_path() if mode == "auto" else mode
    _STORE = TMStore(path)
    _STORE_STATE.update({"mode": mode, "path": path})
    return tm_cache_snapshot()


def tm_cache_enabled():
    return _STORE is not None


def tm_cache_snapshot():
    snapshot = {"tm_cache_mode": _STORE_STATE["mode"],
                "tm_cache_path": _STORE_STATE["path"],
                "memory_entries": len(_MEMORY)}
    if _STORE is not None:
        snapshot.update(_STORE.snapshot())
    return snapshot


def _cached_pair(pdb1, pdb2, usalign_path=None):
    """两层同键：进程内层 → SQLite → 计算（父进程路径）。

    关闭缓存时**两层都不用**（内存层也不能当后备，否则"关闭"名不副实）。
    """
    key = pair_key(pdb1, pdb2, usalign_path)
    if _STORE is None:
        tm, seqid = usalign_pair(pdb1, pdb2, usalign_path)
        return {"tm": tm, "seqid": seqid}
    cached = _MEMORY.get(key)
    if cached is not None:
        return cached
    cached = _STORE.get(key)
    if cached is not None:
        _MEMORY[key] = cached
        return cached
    tm, seqid = usalign_pair(pdb1, pdb2, usalign_path)
    value = {"tm": tm, "seqid": seqid}
    _MEMORY[key] = value
    _STORE.put(key, value)
    return value


def calculate_tm_score(pdb1, pdb2, usalign_path=None):
    """两结构的 TM-score（双向最小值）；失败抛 `USalignError`，合法 0.0 可缓存。"""
    return _cached_pair(pdb1, pdb2, usalign_path)["tm"]


def calculate_seqid(pdb1, pdb2, usalign_path=None):
    """USalign 对齐区间上的序列同一性（n_identical/n_aligned）。

    同源判定用它而非结构 TM：同一蛋白的拷贝即使被组装坏（结构 TM 很低），序列同一性仍≈1。
    与 `calculate_tm_score` 共用同一次 USalign 调用与同一份缓存。
    """
    return _cached_pair(pdb1, pdb2, usalign_path)["seqid"]


def _tm_worker(arg):
    """worker：只跑 USalign（不碰 SQLite），把成功/失败都原样带回父进程。"""
    p1, p2 = arg[0], arg[1]
    usalign_path = arg[2] if len(arg) > 2 else None
    try:
        tm, seqid = usalign_pair(p1, p2, usalign_path)
    except USalignError as exc:
        return {"ok": False, "p1": p1, "p2": p2, "error": str(exc)}
    return {"ok": True, "p1": p1, "p2": p2, "tm": tm, "seqid": seqid}


def prefill_tm_cache(pairs, context, usalign_path=None):
    """用运行级池预填 (pdb1, pdb2) 的 TM/SeqID 进缓存（**父进程写入**）。

    返回实际计算的对数；任一 worker 失败即抛 `USalignError`（不写入伪分数）。
    """
    todo, seen = [], set()
    for p1, p2 in pairs:
        key = pair_key(p1, p2, usalign_path)
        if _STORE is not None and (key in _MEMORY or _STORE.get(key) is not None):
            continue
        if key in seen:
            continue
        seen.add(key)
        todo.append((p1, p2, usalign_path))
    if not todo:
        return 0
    failures = []
    for result in context.map(_tm_worker, todo):
        if not result["ok"]:
            failures.append(result)
            continue
        key = pair_key(result["p1"], result["p2"], usalign_path)
        value = {"tm": result["tm"], "seqid": result["seqid"]}
        if _STORE is not None:
            _MEMORY[key] = value
            _STORE.put(key, value)
    if failures:
        first = failures[0]
        raise USalignError("%d/%d 对结构计算失败，例：%s vs %s\n%s"
                           % (len(failures), len(todo), first["p1"], first["p2"],
                              first["error"]))
    return len(todo)
