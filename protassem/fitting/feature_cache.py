"""按字节计费的有界 LRU 缓存（兼容入口）。

实现已迁到 `protassem.runtime.byte_cache`（T05/T07 与老卡收口 P4 共用同一份）；
本模块保留同名导入，避免既有调用点与测试改动。
"""

from protassem.runtime.byte_cache import (ENTRY_OVERHEAD_BYTES, ByteLruCache,
                                          CacheStats, object_bytes)

__all__ = ["ByteLruCache", "CacheStats", "ENTRY_OVERHEAD_BYTES", "object_bytes"]
