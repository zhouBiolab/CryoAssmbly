"""建池成本微基准：fork（默认）vs spawn，各 3 次，10 worker。

spawn 必须从真实脚本文件运行：它会重新导入 __main__，stdin 脚本（<stdin>）会失败。
"""

import time

from protassem.runtime.pool import close_pool, open_pool


def square(value):
    return value * value


def main():
    for method in (None, "spawn"):
        times = []
        for _ in range(3):
            started = time.perf_counter()
            pool = open_pool(None, 10, "bench", method)
            created = time.perf_counter() - started
            pool.map(square, list(range(50)))
            close_pool(None, pool, "bench")
            times.append(created)
        print("%-14s 建池耗时(s): %s"
              % (method or "default(fork)", ", ".join("%.3f" % t for t in times)))


if __name__ == "__main__":
    main()
