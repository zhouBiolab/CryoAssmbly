"""T05 有界 LRU 缓存契约测试（`protassem/fitting/feature_cache.py`）。

覆盖任务卡 T05 的"容量不超预算"与"0 = 关闭"，以及 LRU 语义、超预算条目不缓存、
字节计费与统计字段。
"""

import unittest

from protassem.fitting.feature_cache import (ENTRY_OVERHEAD_BYTES, ByteLruCache,
                                             CacheStats)

UNIT = 1000


def size_of(value):
    return len(value) * UNIT


class DisabledCacheTest(unittest.TestCase):
    def test_zero_capacity_is_disabled(self):
        cache = ByteLruCache(0, size_of, name="test")
        self.assertFalse(cache.enabled)
        self.assertFalse(cache.put("k", "abcd"))
        self.assertIsNone(cache.get("k"))
        self.assertEqual(1, cache.stats.disabled_puts)
        self.assertEqual(1, cache.stats.misses)
        self.assertEqual(0, cache.bytes)

    def test_negative_capacity_rejected(self):
        with self.assertRaises(ValueError):
            ByteLruCache(-1, size_of)

    def test_non_callable_size_of_rejected(self):
        with self.assertRaises(ValueError):
            ByteLruCache(100, 42)


class LruBehaviourTest(unittest.TestCase):
    def setUp(self):
        # 每条 1000 B + 1024 B 开销 = 2024 B；容量刚好放 2 条
        self.cache = ByteLruCache(2 * (UNIT + ENTRY_OVERHEAD_BYTES), size_of, name="test")

    def test_put_get_and_bytes(self):
        self.assertTrue(self.cache.put("a", "x" * 1))
        self.assertEqual("x" * 1, self.cache.get("a"))
        self.assertEqual(UNIT + ENTRY_OVERHEAD_BYTES, self.cache.bytes)
        self.assertIn("a", self.cache)
        self.assertEqual(1, self.cache.stats.hits)

    def test_eviction_is_least_recently_used(self):
        self.cache.put("a", "x")
        self.cache.put("b", "y")
        self.cache.get("a")            # a 变成最近使用
        self.cache.put("c", "z")       # 触发淘汰：应淘汰 b
        self.assertIn("a", self.cache)
        self.assertIn("c", self.cache)
        self.assertNotIn("b", self.cache)
        self.assertEqual(1, self.cache.stats.evictions)
        self.assertLessEqual(self.cache.bytes, self.cache.capacity_bytes)

    def test_oversized_entry_is_not_cached(self):
        big = "x" * 100
        self.assertFalse(self.cache.put("big", big))
        self.assertNotIn("big", self.cache)
        self.assertEqual(1, self.cache.stats.rejected_too_large)
        self.assertEqual(0, self.cache.bytes)

    def test_replacing_key_frees_old_size(self):
        self.cache.put("a", "x")
        self.cache.put("a", "yy")
        self.assertEqual(2 * UNIT + ENTRY_OVERHEAD_BYTES, self.cache.bytes)
        self.assertEqual(1, self.cache.stats.entries)

    def test_bytes_never_exceed_capacity(self):
        cache = ByteLruCache(5 * (UNIT + ENTRY_OVERHEAD_BYTES), size_of, name="test")
        for index in range(50):
            cache.put("key-%d" % index, "x" * (index % 4 + 1))
            self.assertLessEqual(cache.bytes, cache.capacity_bytes)
        self.assertLessEqual(cache.stats.entries, 5)
        self.assertGreater(cache.stats.evictions, 0)
        self.assertLessEqual(cache.stats.peak_bytes, cache.capacity_bytes)

    def test_clear_and_snapshot(self):
        self.cache.put("a", "x")
        snapshot = self.cache.snapshot()
        self.assertEqual(1, snapshot["entries"])
        self.assertEqual(UNIT + ENTRY_OVERHEAD_BYTES, snapshot["bytes"])
        self.assertEqual(self.cache.capacity_bytes, snapshot["capacity_bytes"])
        self.assertIsNone(snapshot["hit_rate"])
        self.cache.get("a")
        self.assertEqual(1.0, self.cache.snapshot()["hit_rate"])
        self.cache.clear()
        self.assertEqual(0, self.cache.snapshot()["entries"])
        self.assertEqual(0, self.cache.bytes)

    def test_none_key_rejected(self):
        with self.assertRaises(ValueError):
            self.cache.put(None, "x")
        self.assertIsNone(self.cache.get(None))
        self.assertEqual(1, self.cache.stats.misses)


class StatsTest(unittest.TestCase):
    def test_fields_present(self):
        stats = CacheStats(name="c", capacity_bytes=10).as_dict()
        for key in ("cache", "capacity_bytes", "hits", "misses", "evictions",
                    "rejected_too_large", "disabled_puts", "entries", "bytes",
                    "peak_bytes", "hit_rate"):
            self.assertIn(key, stats)


if __name__ == "__main__":
    unittest.main()
