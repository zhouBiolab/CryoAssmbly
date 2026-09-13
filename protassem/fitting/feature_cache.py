"""按字节计费的有界 LRU 缓存（任务卡 T05/T07 共用）。

设计约定
--------
- 容量以**字节**计（不是条目数）；插入时用调用方给的 `size_of(value)` 计费。
- `capacity_bytes <= 0` 表示**关闭**：`get()` 恒 miss、`put()` 不存；调用方代码不需要分叉。
- 单条体积超过容量时，`put()` 返回 False（调用方正常使用该值，只是不缓存）。
- 不做并发保护：当前唯一使用者是 PARENet 常驻服务进程（`demo_mask._server_loop`，
  单进程顺序处理请求），并行 worker 不共享缓存。这是**边界事实**，不是可以靠加锁掩盖的设计；
  若将来跨进程共享，应在调用方引入显式同步或改为进程本地缓存。
- 值必须视为**只读**：`get()` 返回同一个对象，缓存不做深拷贝；调用方不得原地修改。

一行一个操作，参数显式传递，不新增隐式全局状态。
"""

from collections import OrderedDict
from dataclasses import dataclass

ENTRY_OVERHEAD_BYTES = 1024     # 每个条目的对象/字典开销（保守计入，避免账面偏乐观）


@dataclass
class CacheStats:
    """缓存计数（累计值）；`as_dict()` 可直接写进运行记录。"""

    name: str
    capacity_bytes: int
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    rejected_too_large: int = 0
    disabled_puts: int = 0
    entries: int = 0
    bytes: int = 0
    peak_bytes: int = 0

    def as_dict(self):
        data = {"cache": self.name, "capacity_bytes": self.capacity_bytes,
                "hits": self.hits, "misses": self.misses,
                "evictions": self.evictions,
                "rejected_too_large": self.rejected_too_large,
                "disabled_puts": self.disabled_puts,
                "entries": self.entries, "bytes": self.bytes,
                "peak_bytes": self.peak_bytes}
        total = self.hits + self.misses
        data["hit_rate"] = round(float(self.hits) / total, 6) if total else None
        return data


@dataclass
class _Entry:
    value: object
    size: int


class ByteLruCache:
    """字节计费的 LRU；`size_of(value)` 必须返回该条目的字节数。"""

    def __init__(self, capacity_bytes, size_of, name="cache"):
        if capacity_bytes < 0:
            raise ValueError("capacity_bytes 不能为负：%r" % (capacity_bytes,))
        if not callable(size_of):
            raise ValueError("size_of 必须是可调用对象")
        self.capacity_bytes = int(capacity_bytes)
        self.size_of = size_of
        self.name = name
        self.stats = CacheStats(name=name, capacity_bytes=self.capacity_bytes)
        self._entries = OrderedDict()
        self._bytes = 0

    @property
    def enabled(self):
        return self.capacity_bytes > 0

    @property
    def bytes(self):
        return self._bytes

    def __contains__(self, key):
        return key in self._entries

    def get(self, key):
        """命中返回条目并刷新 LRU 顺序；未命中返回 None。"""
        if key is None or not self.enabled:
            self.stats.misses += 1
            return None
        entry = self._entries.get(key)
        if entry is None:
            self.stats.misses += 1
            return None
        self._entries.move_to_end(key)
        self.stats.hits += 1
        return entry.value

    def put(self, key, value):
        """插入条目；返回是否真的存下（关闭/超容量/超预算时为 False）。"""
        if key is None:
            raise ValueError("缓存 key 不能为 None")
        if not self.enabled:
            self.stats.disabled_puts += 1
            return False
        size = int(self.size_of(value)) + ENTRY_OVERHEAD_BYTES
        if size > self.capacity_bytes:
            self.stats.rejected_too_large += 1
            return False
        if key in self._entries:
            self._bytes -= self._entries[key].size
            del self._entries[key]
        self._entries[key] = _Entry(value=value, size=size)
        self._bytes += size
        while self._bytes > self.capacity_bytes and self._entries:
            _, evicted = self._entries.popitem(last=False)
            self._bytes -= evicted.size
            self.stats.evictions += 1
        self._sync_stats()
        return True

    def clear(self):
        self._entries.clear()
        self._bytes = 0
        self._sync_stats()

    def _sync_stats(self):
        self.stats.entries = len(self._entries)
        self.stats.bytes = self._bytes
        self.stats.peak_bytes = max(self.stats.peak_bytes, self._bytes)

    def snapshot(self):
        self._sync_stats()
        return self.stats.as_dict()
