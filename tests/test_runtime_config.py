"""P1 运行配置与线程控制测试（无 GPU）。

关键验证：
  1. JSON 覆盖默认值、未知键报错、CLI 优先级由 main.py 保证；
  2. `apply_thread_env()` 设置全部线程环境变量；
  3. **子进程实测**：在导入 numpy 之前应用 env，threadpoolctl 报告的 BLAS 线程数确实为 1；
  4. `main` 模块导入本身不加载 numpy/torch（两段式导入的契约）。
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from protassem.runtime.config import THREAD_ENV_VARS, RuntimeConfig

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_python(code):
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT,
                            capture_output=True, text=True)
    return result


class RuntimeConfigTest(unittest.TestCase):

    def test_defaults(self):
        config = RuntimeConfig()
        self.assertEqual(config.blas_threads, 1)
        self.assertEqual(config.seed, 7351)

    def test_json_overrides_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "runtime.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"blas_threads": 4, "seed": 11}, handle)
            config = RuntimeConfig.from_json(path)
        self.assertEqual(config.blas_threads, 4)
        self.assertEqual(config.seed, 11)

    def test_unknown_json_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "runtime.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"blas_threadz": 4}, handle)
            with self.assertRaises(ValueError) as ctx:
                RuntimeConfig.from_json(path)
        self.assertIn("unknown runtime config key", str(ctx.exception))

    def test_apply_thread_env_sets_every_variable(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            returned = RuntimeConfig(blas_threads=3).apply_thread_env()
            self.assertEqual(returned, "3")
            for name in THREAD_ENV_VARS:
                self.assertEqual(os.environ[name], "3")

    def test_apply_thread_env_never_sets_zero(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            RuntimeConfig(blas_threads=0).apply_thread_env()
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")

    def test_effective_threads_in_subprocess(self):
        """真实生效证据：导入 numpy 前设置 env → BLAS 线程数为 1。"""
        code = (
            "from protassem.runtime.config import RuntimeConfig\n"
            "RuntimeConfig(blas_threads=1).apply_thread_env()\n"
            "import numpy\n"
            "import threadpoolctl\n"
            "print(threadpoolctl.threadpool_info()[0]['num_threads'])\n"
        )
        result = _run_python(code)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "1")

    def test_main_module_import_does_not_load_numpy_or_torch(self):
        code = ("import sys\n"
                "import main\n"
                "print('numpy' in sys.modules or 'torch' in sys.modules)\n")
        result = _run_python(code)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
