"""固定掩码配准微基准（任务卡 T00）。

只做固定配准，不从头运行装配；产物写到独立目录。

用法：
    # 1) 从一次已完成的装配输出里挑源与至少三个不同大小的目标掩码，生成 manifest
    python tools/benchmark_registration.py manifest --run-dir <输出目录> --out tests/cases/registration_manifest.json

    # 2) 用同一 manifest 重放（可重复多次），产物写到独立目录
    python tools/benchmark_registration.py run --manifest tests/cases/registration_manifest.json \
        --out-dir /path/to/bench_out --repeat 1

manifest 记录：输入 hash 与点数、参数、顺序、seed，以及依赖来源（HEAD/dirty、Python/Torch/CUDA、
权重 hash、pareconv 与 CUDA 扩展的实际导入路径及与仓库副本的对照）。
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import platform
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_VERSION = 1
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from protassem.runtime.config import (DEFAULT_ALLOW_TF32, DEFAULT_INFERENCE_MODE,
                                      INFERENCE_MODES)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def point_count(txt_path):
    """用统一读写模块数点（与主流程同一实现）。"""
    sys.path.insert(0, PROJECT_ROOT)
    from protassem.core.points_txt import read_point_cloud
    return len(read_point_cloud(txt_path).points)


def git_state():
    head = subprocess.run(["git", "-C", PROJECT_ROOT, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", PROJECT_ROOT, "status", "--porcelain"],
                           capture_output=True, text=True).stdout.strip()
    return {"head": head, "dirty": bool(dirty),
            "dirty_entries": dirty.splitlines() if dirty else []}


def dependency_provenance():
    """记录实际加载的依赖来源，并与仓库副本对照（任务卡 T00 风险项）。"""
    import torch
    import pareconv
    import pointops_cuda

    weights = os.path.join(PROJECT_ROOT, "protassem", "fitting", "parenet",
                           "weights", "epoch-18.pth.tar")
    repo_copy = os.path.join(PROJECT_ROOT, "protassem", "fitting", "pareconv_src",
                             "pareconv", "__init__.py")
    return {
        "git": git_state(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "weights_path": weights,
        "weights_sha256": sha256(weights),
        "pareconv_imported_from": os.path.abspath(pareconv.__file__),
        "pareconv_repo_copy": repo_copy,
        "pareconv_uses_repo_copy": os.path.abspath(pareconv.__file__).startswith(
            os.path.join(PROJECT_ROOT, "protassem", "fitting", "pareconv_src")),
        "cuda_extension_imported_from": os.path.abspath(pointops_cuda.__file__),
    }


def build_manifest(run_dir, target_count):
    """从已完成的装配输出里挑 1 个源 + N 个不同大小的目标掩码。"""
    src_dir = os.path.join(run_dir, "sampled_sources")
    sources = sorted(f for f in os.listdir(src_dir) if f.endswith(".txt"))
    if not sources:
        raise FileNotFoundError("no source txt in %s" % src_dir)
    source_txt = os.path.join(src_dir, sources[0])
    source_pdb = os.path.join(src_dir, sources[0].replace("_2.00.txt", ".pdb"))
    if not os.path.exists(source_pdb):
        raise FileNotFoundError("source PDB next to %s not found" % source_txt)

    mask_paths = sorted(
        os.path.join(run_dir, "assembly", "work", name, "filtered.txt")
        for name in os.listdir(os.path.join(run_dir, "assembly", "work"))
        if name.startswith("mask_"))
    masks = [(path, point_count(path)) for path in mask_paths if os.path.exists(path)]
    masks.sort(key=lambda item: -item[1])
    if len(masks) < target_count:
        raise ValueError("need %d masks, found %d" % (target_count, len(masks)))

    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_dir": os.path.abspath(run_dir),
        "seed": 100000,
        "params": {"configs": "all", "use_mask": False,
                   "mask_radius_factor": 1.35, "min_coverage": 0.15,
                   "min_point_distance_factor": 0.32},
        "source": {"txt": os.path.abspath(source_txt),
                   "pdb": os.path.abspath(source_pdb),
                   "sha256": sha256(source_txt),
                   "point_count": point_count(source_txt)},
        "targets": [{"txt": os.path.abspath(path), "sha256": sha256(path),
                     "point_count": count, "order": index}
                    for index, (path, count) in enumerate(masks[:target_count])],
        "dependencies": dependency_provenance(),
    }


def summarize_predictions(pair_dir):
    """从写出的 pred_*.pdb 解析结果（run_inference 无返回值，命名携带 overlap）。"""
    files = sorted(glob.glob(os.path.join(pair_dir, "pred_*.pdb")))
    overlaps = []
    for path in files:
        match = re.search(r"_(\d+\.\d+)_", os.path.basename(path))
        if match:
            overlaps.append(float(match.group(1)))
    return files, overlaps


def run_manifest(manifest_path, out_dir, repeat, geometry_cache_mb=None,
                 inference_mode=DEFAULT_INFERENCE_MODE, allow_tf32=DEFAULT_ALLOW_TF32):
    """按 manifest 重放固定配准（不生成掩码、不跑装配）。

    geometry_cache_mb：T05 单侧几何缓存容量（MiB）；0 = 关闭，None = 用运行配置默认值。
    inference_mode  ：T06 推理路径（joint 默认 = 联合布局 + forward；split = 单侧编码 + 双侧配准）。
    allow_tf32      ：TF32 策略；None = 跟随 inference_mode（split 强制关闭）。
    """
    from protassem.fitting.cloud_encoding import GeometryCache
    from protassem.fitting.demo_mask import run_inference
    from protassem.runtime.config import (DEFAULT_GEOMETRY_CACHE_MB, apply_tf32_policy,
                                          effective_allow_tf32)
    from protassem.runtime.metrics import Metrics

    resolved_tf32 = effective_allow_tf32(inference_mode, allow_tf32)
    policy = apply_tf32_policy(resolved_tf32)
    print("inference_mode=%s allow_tf32=%s -> %s" % (inference_mode, allow_tf32, policy))
    if geometry_cache_mb is None:
        geometry_cache_mb = DEFAULT_GEOMETRY_CACHE_MB
    geometry_cache = GeometryCache(int(geometry_cache_mb) * 1024 * 1024) \
        if geometry_cache_mb else None

    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)

    os.makedirs(out_dir, exist_ok=True)
    metrics = Metrics(output_dir=os.path.join(out_dir, "metrics"),
                      run_id="t00_" + datetime.datetime.now().strftime("%H%M%S"))
    weights = manifest["dependencies"]["weights_path"]
    records = []
    for repeat_index in range(repeat):
        for target in manifest["targets"]:
            pair_dir = os.path.join(out_dir, "target_%d_run_%d"
                                    % (target["order"], repeat_index + 1))
            os.makedirs(pair_dir, exist_ok=True)
            started = time.perf_counter()
            with metrics.stage("registration", target_order=target["order"],
                               repeat=repeat_index + 1):
                run_inference(
                    target=target["txt"], source=manifest["source"]["txt"],
                    chain_pdb=manifest["source"]["pdb"], output_dir=pair_dir,
                    weights=weights, use_mask=False,
                    configs=manifest["params"]["configs"], seed=manifest["seed"],
                    geometry_cache=geometry_cache, allow_tf32=resolved_tf32,
                    inference_mode=inference_mode)
            elapsed = time.perf_counter() - started
            pred_files, overlaps = summarize_predictions(pair_dir)
            records.append({
                "target_order": target["order"], "repeat": repeat_index + 1,
                "target_points": target["point_count"],
                "source_points": manifest["source"]["point_count"],
                "configs": manifest["params"]["configs"],
                "wall_s": round(elapsed, 3),
                "predictions": len(pred_files),
                "overlaps": sorted(round(value, 6) for value in overlaps),
                "best_overlap": max(overlaps) if overlaps else None,
            })
            print("target %d run %d: %.2f s, %d predictions, best overlap=%s"
                  % (target["order"], repeat_index + 1, elapsed, len(pred_files),
                     max(overlaps) if overlaps else None))
    summary_path = metrics.write_summary()
    cache_stats = geometry_cache.snapshot() if geometry_cache is not None else None
    report = {"manifest": os.path.abspath(manifest_path), "out_dir": os.path.abspath(out_dir),
              "repeat": repeat, "records": records,
              "geometry_cache_mb": geometry_cache_mb, "geometry_cache": cache_stats,
              "inference_mode": inference_mode, "allow_tf32": resolved_tf32,
              "tf32_policy": policy,
              "metrics_summary": summary_path}
    with open(os.path.join(out_dir, "t00_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="固定掩码配准微基准（T00）")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("manifest", help="生成 manifest")
    build.add_argument("--run-dir", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--targets", type=int, default=3)

    run = sub.add_parser("run", help="按 manifest 重放")
    run.add_argument("--manifest", required=True)
    run.add_argument("--out-dir", required=True)
    run.add_argument("--repeat", type=int, default=1)
    run.add_argument("--geometry-cache-mb", type=int, default=None,
                     help="单侧几何缓存容量（MiB，0 = 关闭；默认取运行配置默认值）")
    run.add_argument("--inference-mode", choices=INFERENCE_MODES,
                     default=DEFAULT_INFERENCE_MODE,
                     help="joint（默认，联合布局 + forward）或 split（单侧编码 + 双侧配准）")
    run.add_argument("--allow-tf32", dest="allow_tf32", action="store_true", default=None,
                     help="允许 TF32（默认跟随 inference_mode；split 不允许）")
    run.add_argument("--no-allow-tf32", dest="allow_tf32", action="store_false",
                     help="关闭 TF32（数值与形状无关；split 必须）")

    args = parser.parse_args(argv)
    if args.command == "manifest":
        manifest = build_manifest(args.run_dir, args.targets)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        print("manifest written: %s" % args.out)
        print("source: %s (%d points)" % (manifest["source"]["txt"],
                                          manifest["source"]["point_count"]))
        for target in manifest["targets"]:
            print("target %d: %s (%d points)" % (target["order"], target["txt"],
                                                 target["point_count"]))
        deps = manifest["dependencies"]
        print("pareconv imported from: %s" % deps["pareconv_imported_from"])
        print("pareconv uses repo copy: %s" % deps["pareconv_uses_repo_copy"])
        print("cuda ext: %s" % deps["cuda_extension_imported_from"])
        return 0

    report = run_manifest(args.manifest, args.out_dir, args.repeat,
                          geometry_cache_mb=args.geometry_cache_mb,
                          inference_mode=args.inference_mode,
                          allow_tf32=args.allow_tf32)
    print("report written: %s" % os.path.join(args.out_dir, "t00_report.json"))
    print("geometry cache: %s" % (report["geometry_cache"] or "关闭"))
    print("inference_mode: %s ; allow_tf32: %s"
          % (report["inference_mode"], report["allow_tf32"]))
    for record in report["records"]:
        print("  target %d run %d: %.2f s, %d predictions, best overlap=%s"
              % (record["target_order"], record["repeat"], record["wall_s"],
                 record["predictions"], record["best_overlap"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
