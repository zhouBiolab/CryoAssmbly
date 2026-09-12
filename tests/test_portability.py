"""静态可移植性断言：自研代码不含服务器绝对路径、os.system 与 shell=True。

扫描范围为仓库内的自研 Python 代码：排除 `pareconv_src/`（vendored 训练代码，
本次不改）与 `tests/`、`test/`（测试与真实数据），排除 `.git/`、`__pycache__/`。
"""

import os
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIPPED_DIRS = {".git", "__pycache__", "pareconv_src", "tests", "test"}

FORBIDDEN = (
    ("/xiangyux/", "服务器绝对路径必须改为参数或仓库相对路径"),
    ("os.system(", "外部命令必须用 subprocess 的列表参数"),
    ("shell=True", "不要通过 shell 执行命令"),
)


def _iter_python_files():
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIPPED_DIRS]
        for name in sorted(files):
            if name.endswith(".py"):
                yield os.path.join(root, name)


class PortabilityTest(unittest.TestCase):

    def test_no_forbidden_literals_in_shipped_code(self):
        offenders = []
        for path in _iter_python_files():
            with open(path, encoding="utf-8", errors="replace") as handle:
                for lineno, line in enumerate(handle, 1):
                    for literal, reason in FORBIDDEN:
                        if literal in line:
                            offenders.append("%s:%d: %s (%s)"
                                             % (os.path.relpath(path, PROJECT_ROOT),
                                                lineno, literal, reason))
        self.assertEqual([], offenders, "\n" + "\n".join(offenders))


if __name__ == "__main__":
    unittest.main()
