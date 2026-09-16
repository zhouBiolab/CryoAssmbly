"""回归测试：`local_optimize` 调用点的签名绑定 + 独立入口的并行能力。

背景（本文件要拦住的缺陷）
--------------------------
P3（`773acf7`）把 `local_optimize` 的 `num_processes=1` 参数换成了 `context=None`，
但 `assembly/homo_chain_refine.py` 里三个调用点没跟着改，仍然传 `num_processes=1`：

- 该关键字不存在 → 调用时抛 `TypeError`；
- 三个调用点都在 `try/except Exception` 里，于是 Step 5 同源链精修**静默失效**
  （两个只写一行 stderr，`_fill_worker` 连 stderr 都不写）；
- 该路径需要 `--homo-chain-refine` 才走，`tests/` 下原本没有任何覆盖，所以一直没被发现。

另有第二处同类缺陷：`fitting/local_optimizer.py` 的 `__main__` 把 `a.num_processes`
当**位置**参数传，参数删除后不报错而是**错位**（字符串路径落进 `initial_cc`），
随后 `log.info(..., initial_cc)` 抛错并被函数内 `except` 兜成 `success=False`。

本文件用三种方式覆盖：

1. 静态：`ast` 扫全仓，逐个调用点 `inspect.signature(...).bind()`，不允许抛 `TypeError`；
2. 行为：`--num_processes 10` 与 `1` 都必须 `ok=True` 且产出文件存在；
3. 契约：确认该路径确实会调用 `local_optimize`（防止测试退化成空跑）。
"""

import ast
import glob
import inspect
import os
import subprocess
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from protassem.fitting.local_optimizer import local_optimize          # noqa: E402
from protassem.runtime.execution import ExecutionContext               # noqa: E402


def square_for_pool(value):
    """模块级函数：满足共享池「worker 必须是模块级可序列化对象」的契约。"""
    return value * value

TARGET_NAME = "local_optimize"
SKIP_DIRS = {".git", "__pycache__", "build", "dist", "pareconv_src",
             ".venv", "venv", "env"}

# tiny_case 提供极小 pdb+mrc+参数；缺失时跳过行为测试（不把绝对路径写进自研代码）
TINY_CASE_ENV = "PROTASSEM_TINY_CASE"
DEFAULT_TINY_CASE = "/xiangyux/claude_c_work/demo_reg_cases/tiny_case"


def _iter_python_files():
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in sorted(files):
            if name.endswith(".py"):
                yield os.path.join(root, name)


def _parse(path):
    """按 `utf-8-sig` 读取以避免 BOM：Python 3.8 的 `ast.parse` 处理不了带 BOM 的源码，
    `homo_chain_refine.py` 正好带 BOM，用普通 utf-8 读会让它被静默跳过（本测试曾因此漏检）。"""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
        return ast.parse(handle.read(), filename=path)


def _collect_call_sites():
    """返回 [(相对路径, 行号, 关键字参数名集合, 位置参数个数), ...]。"""
    sites = []
    parse_failures = []
    for path in _iter_python_files():
        try:
            tree = _parse(path)
        except SyntaxError as exc:
            parse_failures.append("%s: %s" % (os.path.relpath(path, PROJECT_ROOT), exc))
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else (
                func.attr if isinstance(func, ast.Attribute) else None)
            if name != TARGET_NAME:
                continue
            keywords = [kw.arg for kw in node.keywords if kw.arg is not None]
            sites.append((os.path.relpath(path, PROJECT_ROOT), node.lineno,
                          keywords, len(node.args)))
    _collect_call_sites.parse_failures = parse_failures
    return sites


def _tiny_case_dir():
    candidate = os.environ.get(TINY_CASE_ENV, DEFAULT_TINY_CASE)
    if not os.path.isdir(candidate):
        return None
    needed = ("chain_a.pdb", "tiny.mrc", "resolution.txt", "contour_level.txt")
    if not all(os.path.exists(os.path.join(candidate, name)) for name in needed):
        return None
    return candidate


class CallSiteSignatureTest(unittest.TestCase):
    """静态绑定检查：任何调用点都不得传 `local_optimize` 不接受的参数。"""

    def test_local_optimize_rejects_removed_num_processes(self):
        """先确认基线事实：`num_processes` 确实已不是该函数的参数。"""
        parameters = inspect.signature(local_optimize).parameters
        self.assertNotIn("num_processes", parameters)
        self.assertIn("context", parameters)

    def test_all_project_sources_parse(self):
        """自研代码必须全部可解析：否则上面的扫描会静默漏掉整个文件。"""
        _collect_call_sites()
        failures = getattr(_collect_call_sites, "parse_failures", [])
        self.assertEqual([], failures, "以下文件无法解析，调用点扫描会漏检：\n"
                         + "\n".join(failures))

    def test_no_call_site_has_unbindable_arguments(self):
        sites = _collect_call_sites()
        self.assertTrue(sites, "没有扫到任何 local_optimize 调用点，测试会退化成空跑")
        signature = inspect.signature(local_optimize)
        offenders = []
        for relative, lineno, keywords, positional in sites:
            args = [object()] * positional
            kwargs = {name: object() for name in keywords}
            try:
                signature.bind(*args, **kwargs)
            except TypeError as exc:
                offenders.append("%s:%d 传了 %s -> %s"
                                 % (relative, lineno, sorted(keywords), exc))
        self.assertEqual([], offenders,
                         "\n调用点与签名不匹配（运行时会被 except 静默吞掉）：\n"
                         + "\n".join(offenders))

    def test_homo_chain_refine_still_calls_local_optimize(self):
        """契约：Step 5 路径必须仍然携带这些调用点。

        缺陷的成因正是"改了签名、漏改这个文件"，若哪天这些调用点被整段删除，
        本测试应当失败并提醒维护者同步调整预期，而不是静默通过。
        """
        target = os.path.join(PROJECT_ROOT, "protassem", "assembly",
                              "homo_chain_refine.py")
        self.assertTrue(os.path.exists(target), target)
        relative = os.path.relpath(target, PROJECT_ROOT).replace(os.sep, "/")
        hits = [site for site in _collect_call_sites() if site[0] == relative]
        self.assertGreaterEqual(
            len(hits), 3,
            "homo_chain_refine.py 里的 local_optimize 调用点少于 3 处：%r" % (hits,))
        for _, lineno, keywords, _ in hits:
            self.assertIn("context", keywords,
                          "第 %d 行未显式给出 context：该路径运行在 Pool worker 内，"
                          "必须保持「不嵌套池」的语义" % lineno)


class MainEntryParallelTest(unittest.TestCase):
    """`python local_optimizer.py ... --num_processes N` 必须真的能跑（含并行）。"""

    def setUp(self):
        self.tiny = _tiny_case_dir()
        if self.tiny is None:
            self.skipTest("未找到 tiny_case（设 %s 指向含 pdb/mrc/参数的目录）"
                          % TINY_CASE_ENV)
        with open(os.path.join(self.tiny, "resolution.txt")) as handle:
            self.resolution = handle.read().strip()
        with open(os.path.join(self.tiny, "contour_level.txt")) as handle:
            self.contour = handle.read().strip()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _run_cli(self, workers):
        output = os.path.join(self._tmp.name, "opt_%d.pdb" % workers)
        command = [sys.executable, os.path.join(PROJECT_ROOT, "protassem", "fitting",
                                                "local_optimizer.py"),
                   os.path.join(self.tiny, "chain_a.pdb"),
                   os.path.join(self.tiny, "tiny.mrc"), output,
                   "--resolution", self.resolution, "--contour", self.contour,
                   "--num_processes", str(workers)]
        # 该入口按生产调用方式运行：从仓库根、把仓库根放进 PYTHONPATH
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [PROJECT_ROOT] + ([environment["PYTHONPATH"]]
                              if environment.get("PYTHONPATH") else []))
        completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True,
                                   text=True, timeout=900, env=environment)
        return completed, output

    def test_cli_serial_exits_zero_and_writes_output(self):
        completed, output = self._run_cli(1)
        self.assertEqual(0, completed.returncode,
                         "串行入口失败：\n%s\n%s" % (completed.stdout[-2000:],
                                                     completed.stderr[-2000:]))
        self.assertTrue(os.path.exists(output), output)

    def test_cli_parallel_exits_zero_and_writes_output(self):
        """并行入口：修复前这里会因错位参数静默 success=False（退出码 1）。"""
        completed, output = self._run_cli(10)
        self.assertEqual(0, completed.returncode,
                         "并行入口失败：\n%s\n%s" % (completed.stdout[-2000:],
                                                     completed.stderr[-2000:]))
        self.assertTrue(os.path.exists(output), output)


class ExecutionContextParallelTest(unittest.TestCase):
    """`ExecutionContext` 在 workers>1 时必须走池且结果与串行一致。"""

    def test_serial_branch_runs_in_process(self):
        """workers<=1 时不建池、结果按输入顺序返回（顺序归并语义）。"""
        calls = {"count": 0}

        def _square(value):
            calls["count"] += 1
            return value * value

        context = ExecutionContext(pool_workers=1)
        try:
            self.assertEqual([1, 4, 9], context.map(_square, [1, 2, 3]))
        finally:
            context.close()
        self.assertEqual(3, calls["count"])

    def test_parallel_map_with_module_level_function(self):
        """workers>1 时走真实进程池，且结果与串行一致（顺序归并）。"""
        context = ExecutionContext(pool_workers=10)
        try:
            self.assertEqual([1, 4, 9], list(context.map(square_for_pool, [1, 2, 3])))
        finally:
            context.close()


if __name__ == "__main__":
    unittest.main()
