"""Client for the persistent PARENet inference server (demo_mask.py --server).

The server is a single long-lived subprocess that loads the GPU model once and
serves many fitting requests. Keeping inference in a SEPARATE process means the
main (orchestration) process never initialises CUDA, so the fork-based parallel
CC / local-optimize pools stay safe.

Inter-process protocol (no blocking):
  - request  : one JSON line written to the server's stdin
  - pred out : pred_*.pdb files written to output_dir (monitored by the caller)
  - done     : server creates output_dir/_PARENET_DONE when a request finishes
  - stop     : caller creates output_dir/_PARENET_STOP to early-terminate
Server stdout/stderr go to a log file (never a PIPE), so the caller can never
block on a full pipe buffer.
"""

import os
import sys
import json
import atexit
import logging
import subprocess

from protassem.runtime.config import (DEFAULT_ALLOW_TF32, DEFAULT_ENCODING_CACHE_MB,
                                      DEFAULT_GEOMETRY_CACHE_MB, DEFAULT_INFERENCE_MODE,
                                      INFERENCE_MODES)

log = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_MASK_PATH = os.path.join(_HERE, "demo_mask.py")
DEMO_MASK_CWD = os.path.dirname(os.path.dirname(_HERE))

DONE_MARKER = "_PARENET_DONE"
STOP_MARKER = "_PARENET_STOP"

_SERVER = None
# 服务端配置：几何缓存容量（T05）、推理路径与 TF32 策略（T06）；
# 由 configure_* 在**启动服务前**显式设定（服务已在运行时只告警、不重启）
_GEOMETRY_CACHE_MB = DEFAULT_GEOMETRY_CACHE_MB
_INFERENCE_MODE = DEFAULT_INFERENCE_MODE
_ALLOW_TF32 = DEFAULT_ALLOW_TF32
_ENCODING_CACHE_MB = DEFAULT_ENCODING_CACHE_MB


def configure_geometry_cache(geometry_cache_mb):
    """设定服务端几何缓存容量（必须在第一次请求之前调用）。

    服务进程在启动时接收该值；若服务已经在运行，则保留其现有配置并记录警告——
    不做"悄悄重启服务"这种会打断在飞请求的事。
    """
    global _GEOMETRY_CACHE_MB
    value = int(geometry_cache_mb)
    if value < 0:
        raise ValueError("geometry_cache_mb 不能为负：%r" % (geometry_cache_mb,))
    if _SERVER is not None and _SERVER.poll() is None and value != _GEOMETRY_CACHE_MB:
        log.warning("PARENet server already running with geometry_cache_mb=%s; "
                    "keeping it (new value %s ignored)", _GEOMETRY_CACHE_MB, value)
        return _GEOMETRY_CACHE_MB
    _GEOMETRY_CACHE_MB = value
    return _GEOMETRY_CACHE_MB


def configure_allow_tf32(allow_tf32):
    """设定服务端 TF32 策略（None = 跟随推理模式；必须在第一次请求之前调用）。"""
    global _ALLOW_TF32
    value = None if allow_tf32 is None else bool(allow_tf32)
    if _SERVER is not None and _SERVER.poll() is None and value != _ALLOW_TF32:
        log.warning("PARENet server already running with allow_tf32=%s; "
                    "keeping it (new value %s ignored)", _ALLOW_TF32, value)
        return _ALLOW_TF32
    _ALLOW_TF32 = value
    return _ALLOW_TF32


def configure_encoding_cache(encoding_cache_mb):
    """设定服务端源编码缓存预算（MiB，T07；0 = 关闭；必须在第一次请求之前调用）。"""
    global _ENCODING_CACHE_MB
    value = int(encoding_cache_mb)
    if value < 0:
        raise ValueError("encoding_cache_mb 不能为负：%r" % (encoding_cache_mb,))
    if _SERVER is not None and _SERVER.poll() is None and value != _ENCODING_CACHE_MB:
        log.warning("PARENet server already running with encoding_cache_mb=%s; "
                    "keeping it (new value %s ignored)", _ENCODING_CACHE_MB, value)
        return _ENCODING_CACHE_MB
    _ENCODING_CACHE_MB = value
    return _ENCODING_CACHE_MB


def configure_inference_mode(inference_mode):
    """设定服务端推理路径（joint/split；必须在第一次请求之前调用）。"""
    global _INFERENCE_MODE
    if inference_mode not in INFERENCE_MODES:
        raise ValueError("未知的 inference_mode: %r" % (inference_mode,))
    if _SERVER is not None and _SERVER.poll() is None and inference_mode != _INFERENCE_MODE:
        log.warning("PARENet server already running with inference_mode=%s; "
                    "keeping it (new value %s ignored)", _INFERENCE_MODE, inference_mode)
        return _INFERENCE_MODE
    _INFERENCE_MODE = inference_mode
    return _INFERENCE_MODE


def get_server():
    """Lazily start the persistent PARENet server subprocess (singleton)."""
    global _SERVER
    if _SERVER is not None and _SERVER.poll() is None:
        return _SERVER

    log_path = os.path.join(DEMO_MASK_CWD, "parenet_server.log")
    logf = open(log_path, "a", buffering=1)
    command = [sys.executable, DEMO_MASK_PATH, "--server",
               "--geometry-cache-mb", str(_GEOMETRY_CACHE_MB),
               "--encoding-cache-mb", str(_ENCODING_CACHE_MB),
               "--inference-mode", _INFERENCE_MODE]
    if _ALLOW_TF32 is True:
        command.append("--allow-tf32")
    elif _ALLOW_TF32 is False:
        command.append("--no-allow-tf32")
    _SERVER = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=logf, stderr=logf,
        text=True, cwd=DEMO_MASK_CWD)
    atexit.register(shutdown_server)
    log.info("PARENet server started (pid=%d, geometry_cache_mb=%s, encoding_cache_mb=%s, "
             "inference_mode=%s, allow_tf32=%s, log=%s)",
             _SERVER.pid, _GEOMETRY_CACHE_MB, _ENCODING_CACHE_MB, _INFERENCE_MODE,
             _ALLOW_TF32, log_path)
    return _SERVER


def shutdown_server():
    """Stop the persistent server (close stdin -> EOF -> server loop ends)."""
    global _SERVER
    if _SERVER is None:
        return
    if _SERVER.poll() is None:
        try:
            _SERVER.stdin.close()
            _SERVER.wait(timeout=10)
        except Exception:
            try:
                _SERVER.terminate()
            except Exception:
                pass
    _SERVER = None


class ParenetRequest:
    """Handle for one in-flight request; mimics a Popen (poll/terminate)."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.done_file = os.path.join(output_dir, DONE_MARKER)
        self.stop_file = os.path.join(output_dir, STOP_MARKER)

    def poll(self):
        """None while running, 0 once the server signalled completion."""
        return 0 if os.path.exists(self.done_file) else None

    def terminate(self):
        """Ask the server to stop the current request early (file signal)."""
        try:
            open(self.stop_file, "w").close()
        except Exception:
            pass

    # subprocess-compatible no-ops
    def wait(self, timeout=None):
        return self.poll()

    def kill(self):
        self.terminate()


def start_request(target, source, chain_pdb, output_dir,
                  use_mask=True, configs="all",
                  mask_radius_factor=1.35, min_point_distance_factor=0.32,
                  inference_mode=None):
    """Send one fitting request to the persistent server; return a handle.

    inference_mode：覆盖服务端默认推理路径（None = 用服务端设定）；仅用于对照实验。
    """
    os.makedirs(output_dir, exist_ok=True)
    # clear stale markers from any previous run in this dir
    for m in (DONE_MARKER, STOP_MARKER):
        fp = os.path.join(output_dir, m)
        if os.path.exists(fp):
            os.remove(fp)

    server = get_server()
    request = {
        "target": str(target),
        "source": str(source),
        "chain_pdb": str(chain_pdb) if chain_pdb else None,
        "output_dir": str(output_dir),
        "use_mask": use_mask,
        "configs": configs,
        "mask_radius_factor": mask_radius_factor,
        "min_point_distance_factor": min_point_distance_factor,
    }
    # inference_mode 为 None 表示"用服务端设定"：此时**不写该键**（写 None 会被服务端当成非法值）
    if inference_mode is not None:
        request["inference_mode"] = inference_mode
    server.stdin.write(json.dumps(request) + "\n")
    server.stdin.flush()
    log.info("PARENet request: %s", os.path.basename(str(source)))
    return ParenetRequest(output_dir)
