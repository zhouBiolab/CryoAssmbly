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

log = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_MASK_PATH = os.path.join(_HERE, "demo_mask.py")
DEMO_MASK_CWD = os.path.dirname(os.path.dirname(_HERE))

DONE_MARKER = "_PARENET_DONE"
STOP_MARKER = "_PARENET_STOP"

_SERVER = None


def get_server():
    """Lazily start the persistent PARENet server subprocess (singleton)."""
    global _SERVER
    if _SERVER is not None and _SERVER.poll() is None:
        return _SERVER

    log_path = os.path.join(DEMO_MASK_CWD, "parenet_server.log")
    logf = open(log_path, "a", buffering=1)
    _SERVER = subprocess.Popen(
        [sys.executable, DEMO_MASK_PATH, "--server"],
        stdin=subprocess.PIPE, stdout=logf, stderr=logf,
        text=True, cwd=DEMO_MASK_CWD)
    atexit.register(shutdown_server)
    log.info("PARENet server started (pid=%d, log=%s)", _SERVER.pid, log_path)
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
                  mask_radius_factor=1.35, min_point_distance_factor=0.32):
    """Send one fitting request to the persistent server; return a handle."""
    os.makedirs(output_dir, exist_ok=True)
    # clear stale markers from any previous run in this dir
    for m in (DONE_MARKER, STOP_MARKER):
        fp = os.path.join(output_dir, m)
        if os.path.exists(fp):
            os.remove(fp)

    server = get_server()
    req = json.dumps({
        "target": str(target),
        "source": str(source),
        "chain_pdb": str(chain_pdb) if chain_pdb else None,
        "output_dir": str(output_dir),
        "use_mask": use_mask,
        "configs": configs,
        "mask_radius_factor": mask_radius_factor,
        "min_point_distance_factor": min_point_distance_factor,
    })
    server.stdin.write(req + "\n")
    server.stdin.flush()
    log.info("PARENet request: %s", os.path.basename(str(source)))
    return ParenetRequest(output_dir)
