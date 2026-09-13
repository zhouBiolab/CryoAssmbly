"""P5 TM 缓存测试：失败抛错、0.0 合法可缓存、两层同键、失效与对称性。"""

import os
import sqlite3
import stat
import tempfile
import unittest

from protassem.core import similarity
from protassem.core.similarity import (TMStore, USalignError, calculate_seqid,
                                       calculate_tm_score, configure_tm_cache,
                                       pair_key, prefill_tm_cache,
                                       tm_cache_snapshot)
from tests import fixtures

FAKE_USALIGN = """#!/bin/bash
# 假 USalign：行为由**脚本内容**决定（避免用环境变量绕开指纹），并记录调用。
log="{log}"
echo "$1 $2" >> "$log"
{mode}
cat <<EOF
TM-score=   {tm1} (normalized by length of Structure_1)
TM-score=   {tm2} (normalized by length of Structure_2)
Seq_ID=n_identical/n_aligned= {seqid}
EOF
"""


def _write_fake_usalign(path, mode="ok", tm=None, seqid="0.7500", version="v1"):
    """写一个假 USalign；`tm=None` 表示 TM 由两个输入文件的字节数决定（内容相关）。"""
    if mode == "exit":
        body, tm1, tm2 = 'echo "boom" >&2\nexit 3', "0.0", "0.0"
    elif mode == "garbage":
        body, tm1, tm2 = 'echo "no useful output"\nexit 0', "0.0", "0.0"
    else:
        body = ":"
        tm1 = '$(wc -c < "$1")' if tm is None else str(tm)
        tm2 = '$(wc -c < "$2")' if tm is None else str(tm)
    text = FAKE_USALIGN.format(log=os.environ["FAKE_USALIGN_LOG"], mode=body,
                               tm1=tm1, tm2=tm2, seqid=seqid)
    text = text.replace("\n", "\n# %s\n" % version, 1)      # 版本标记放第二行，保持 shebang 合法
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _expected_tm(structure_a, structure_b):
    return float(min(os.path.getsize(structure_a), os.path.getsize(structure_b)))


def _count_calls(log_path):
    if not os.path.exists(log_path):
        return 0
    with open(log_path, encoding="utf-8") as handle:
        return len([line for line in handle if line.strip()])


class _FakeContext:
    """最小 ExecutionContext 替身：顺序执行（P5 只要求父进程写入）。"""

    def map(self, func, items):
        return [func(item) for item in items]


class SimilarityCacheTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        self.log = os.path.join(self.dir, "calls.log")
        os.environ["FAKE_USALIGN_LOG"] = self.log
        self._old_env = os.environ.get("FAKE_USALIGN_LOG")
        self.addCleanup(self._restore_env)
        self.usalign = _write_fake_usalign(os.path.join(self.dir, "USalign"))
        self.structure_a = fixtures.make_chain_structure(
            os.path.join(self.dir, "a.pdb"), [("A", (0.0, 0.0, 0.0))], residues=6)
        self.structure_b = fixtures.make_chain_structure(
            os.path.join(self.dir, "b.pdb"), [("A", (2.0, 1.0, 0.0))], residues=6)
        self.expected = _expected_tm(self.structure_a, self.structure_b)
        self.db = os.path.join(self.dir, "cache", "tm.sqlite3")
        configure_tm_cache(self.db)
        self.addCleanup(configure_tm_cache, "off")

    def _restore_env(self):
        if self._old_env is None:
            os.environ.pop("FAKE_USALIGN_LOG", None)
        else:
            os.environ["FAKE_USALIGN_LOG"] = self._old_env

    def _rows(self):
        connection = sqlite3.connect(self.db)
        try:
            return connection.execute("SELECT COUNT(*) FROM tm_scores").fetchone()[0]
        finally:
            connection.close()

    def test_first_call_computes_second_uses_memory(self):
        first = calculate_tm_score(self.structure_a, self.structure_b, self.usalign)
        second = calculate_tm_score(self.structure_a, self.structure_b, self.usalign)
        self.assertEqual(first, second)
        self.assertEqual(_count_calls(self.log), 1)
        self.assertEqual(self._rows(), 1)

    def test_sqlite_layer_survives_memory_reset(self):
        calculate_tm_score(self.structure_a, self.structure_b, self.usalign)
        configure_tm_cache(self.db)          # 清内存层，保留磁盘层
        again = calculate_tm_score(self.structure_a, self.structure_b, self.usalign)
        self.assertEqual(_count_calls(self.log), 1)
        self.assertAlmostEqual(again, self.expected, places=6)

    def test_pair_order_is_symmetric(self):
        calculate_tm_score(self.structure_a, self.structure_b, self.usalign)
        reversed_value = calculate_tm_score(self.structure_b, self.structure_a,
                                            self.usalign)
        self.assertEqual(_count_calls(self.log), 1)
        self.assertAlmostEqual(reversed_value, self.expected, places=6)

    def test_tm_and_seqid_share_one_invocation(self):
        tm = calculate_tm_score(self.structure_a, self.structure_b, self.usalign)
        seqid = calculate_seqid(self.structure_a, self.structure_b, self.usalign)
        self.assertEqual(_count_calls(self.log), 1)
        self.assertAlmostEqual(seqid, 0.75, places=6)
        self.assertAlmostEqual(tm, self.expected, places=6)

    def test_zero_score_is_legal_and_cached(self):
        _write_fake_usalign(self.usalign, tm="0.0", version="v-zero")
        value = calculate_tm_score(self.structure_a, self.structure_b, self.usalign)
        self.assertEqual(value, 0.0)
        self.assertEqual(self._rows(), 1)
        self.assertEqual(calculate_tm_score(self.structure_a, self.structure_b,
                                            self.usalign), 0.0)
        self.assertEqual(_count_calls(self.log), 1)

    def test_failure_raises_and_is_not_cached(self):
        _write_fake_usalign(self.usalign, mode="exit", version="v-exit")
        with self.assertRaises(USalignError) as caught:
            calculate_tm_score(self.structure_a, self.structure_b, self.usalign)
        self.assertIn("退出码 3", str(caught.exception))
        self.assertEqual(self._rows(), 0)
        _write_fake_usalign(self.usalign, mode="garbage", version="v-garbage")
        with self.assertRaises(USalignError):
            calculate_tm_score(self.structure_a, self.structure_b, self.usalign)
        self.assertEqual(self._rows(), 0)

    def test_structure_content_change_invalidates(self):
        calculate_tm_score(self.structure_a, self.structure_b, self.usalign)
        fixtures.make_chain_structure(self.structure_b, [("A", (5.0, 0.0, 0.0))],
                                      residues=9)
        value = calculate_tm_score(self.structure_a, self.structure_b, self.usalign)
        self.assertEqual(_count_calls(self.log), 2)
        self.assertAlmostEqual(value, _expected_tm(self.structure_a, self.structure_b),
                               places=6)

    def test_usalign_binary_change_invalidates(self):
        calculate_tm_score(self.structure_a, self.structure_b, self.usalign)
        _write_fake_usalign(self.usalign, tm="0.42", version="v2")
        value = calculate_tm_score(self.structure_a, self.structure_b, self.usalign)
        self.assertEqual(_count_calls(self.log), 2)
        self.assertAlmostEqual(value, 0.42, places=6)

    def test_disabled_cache_still_computes(self):
        configure_tm_cache("off")
        self.assertFalse(similarity.tm_cache_enabled())
        calculate_tm_score(self.structure_a, self.structure_b, self.usalign)
        calculate_tm_score(self.structure_a, self.structure_b, self.usalign)
        self.assertEqual(_count_calls(self.log), 2)      # 无缓存 → 每次都算

    def test_prefill_writes_rows_and_dedupes(self):
        pairs = [(self.structure_a, self.structure_b),
                 (self.structure_b, self.structure_a)]
        count = prefill_tm_cache(pairs, _FakeContext(), self.usalign)
        self.assertEqual(count, 1)
        self.assertEqual(self._rows(), 1)
        self.assertEqual(_count_calls(self.log), 1)

    def test_prefill_raises_on_worker_failure(self):
        _write_fake_usalign(self.usalign, mode="exit", version="v-exit")
        with self.assertRaises(USalignError):
            similarity.prefill_tm_cache([(self.structure_a, self.structure_b)],
                                        _FakeContext(), self.usalign)
        self.assertEqual(self._rows(), 0)

    def test_pair_key_is_versioned_and_symmetric(self):
        key1 = pair_key(self.structure_a, self.structure_b, self.usalign)
        key2 = pair_key(self.structure_b, self.structure_a, self.usalign)
        self.assertEqual(key1, key2)
        self.assertTrue(key1.startswith(str(similarity.TM_CACHE_VERSION)))

    def test_store_distinguishes_missing_from_zero(self):
        store = TMStore(os.path.join(self.dir, "direct.sqlite3"))
        self.addCleanup(store.close)
        self.assertIsNone(store.get("missing"))
        store.put("zero", {"tm": 0.0, "seqid": 0.0})
        self.assertEqual(store.get("zero"), {"tm": 0.0, "seqid": 0.0})

    def test_default_path_uses_xdg(self):
        os.environ["XDG_CACHE_HOME"] = self.dir
        self.addCleanup(os.environ.pop, "XDG_CACHE_HOME", None)
        self.assertEqual(similarity.default_tm_cache_path(),
                         os.path.join(self.dir, "protassem", "tm.sqlite3"))
        snapshot = configure_tm_cache("auto")
        self.assertEqual(snapshot["tm_cache_mode"], "auto")


if __name__ == "__main__":
    unittest.main()
