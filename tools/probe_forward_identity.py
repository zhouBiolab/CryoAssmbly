"""前向输出逐字段取证（任务卡 T03/T10 的等价性工具）。

用途：对固定输入跑一次 PARENet 前向，把 `output_dict` 的**全部字段**按位落盘（`.npz` +
字段清单 JSON），供两棵树/两个提交之间做逐位与容差比对（统一验收：特征诊断 atol=1e-6、
rtol=1e-5；位姿与候选身份要求一致）。

用法（在"被测树"的 PYTHONPATH 下运行；工具本身与所在树无关）：

    PYTHONPATH=<tree> python <任意路径>/tools/probe_forward_identity.py \
        --manifest tests/cases/registration_manifest.json \
        --target-order 0 --config-id 0 --sampling voxel --out-dir <dir>

脚本只做一次固定配准的前向，不写 PDB、不改随机状态之外的任何全局状态；
调用 `model(data_dict)`（不传 output_fields），因此新旧两棵树都能直接跑。
"""

import argparse
import hashlib
import json
import os
import random
import time

import numpy as np


def prepare_input(source_txt, target_txt, config_id, sampling, cfg):
    """复刻 demo_mask.process_single_pair 的输入准备（中心化 → scale → collate）。"""
    from protassem.fitting.demo_mask import preprocess_point_cloud_data
    from pareconv.utils.data_mask import registration_collate_fn_stack_mode

    src_data = preprocess_point_cloud_data(source_txt, point_limit=70000)
    tgt_data = preprocess_point_cloud_data(target_txt, point_limit=70000)
    c_ref = tgt_data.centroid
    c_src = src_data.centroid
    ref_norm = tgt_data.points.copy()
    src_norm = src_data.points.copy()
    ref_norm[:, :3] -= c_ref
    src_norm[:, :3] -= c_src
    scale = max(np.linalg.norm(ref_norm, axis=1).max(),
                np.linalg.norm(src_norm, axis=1).max()).astype(np.float32)
    data_dict = {
        "ref_points": ref_norm.astype(np.float32),
        "src_points": src_norm.astype(np.float32),
        "ref_feats": np.ones((ref_norm.shape[0], 1), dtype=np.float32),
        "src_feats": np.ones((src_norm.shape[0], 1), dtype=np.float32),
        "transform": np.eye(4, dtype=np.float32),
        "scale": scale,
    }
    data_dict = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size,
        cfg.backbone.num_neighbors, cfg.backbone.subsample_ratio)
    return data_dict, c_src, c_ref, scale


def tensor_record(value):
    """把张量/数组转成可落盘的 numpy 数组；其它类型返回 None。"""
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="前向输出逐字段取证（T03/T10）")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--target-order", type=int, default=0)
    # config_id / sampling 在本流程里只是标签（不改变送入模型的数据，见 T02 print），
    # 这里照原样记录，便于与基准产物对齐。
    parser.add_argument("--config-id", type=int, default=0)
    parser.add_argument("--sampling", default="voxel")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    import torch
    from protassem.fitting.parenet.config import make_cfg
    from protassem.fitting.parenet.model import create_model
    import protassem.fitting.parenet.model as model_module

    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    target = [item for item in manifest["targets"]
              if item["order"] == args.target_order][0]

    random.seed(manifest["seed"])
    np.random.seed(manifest["seed"])
    torch.manual_seed(manifest["seed"])

    cfg = make_cfg()
    model = create_model(cfg).cuda()
    state = torch.load(manifest["dependencies"]["weights_path"])
    model.load_state_dict(state["model"])
    model.eval()

    data_dict, c_src, c_ref, scale = prepare_input(
        manifest["source"]["txt"], target["txt"], args.config_id, args.sampling, cfg)
    from pareconv.utils.data_mask import precompute_neibors
    from pareconv.utils.torch import to_cuda
    data_dict = to_cuda(data_dict)
    data_dict.update(precompute_neibors(
        data_dict["points"], data_dict["lengths"],
        cfg.backbone.num_stages, cfg.backbone.num_neighbors))
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        output_dict = model(data_dict)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    os.makedirs(args.out_dir, exist_ok=True)
    arrays = {}
    fields = []
    for name in sorted(output_dict):
        array = tensor_record(output_dict[name])
        if array is None:
            fields.append({"name": name, "kind": type(output_dict[name]).__name__,
                           "value": repr(output_dict[name])})
            continue
        digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        arrays["out__" + name] = array
        fields.append({"name": name, "kind": "tensor", "shape": list(array.shape),
                       "dtype": str(array.dtype), "sha256": digest})

    arrays["input__transform"] = data_dict["transform"].detach().cpu().numpy()
    arrays["input__scale"] = np.asarray([float(scale)], dtype=np.float32)
    arrays["meta__centroid_src"] = np.asarray(c_src, dtype=np.float32)
    arrays["meta__centroid_ref"] = np.asarray(c_ref, dtype=np.float32)

    np.savez(os.path.join(args.out_dir, "forward_outputs.npz"), **arrays)
    summary = {
        "manifest": os.path.abspath(args.manifest),
        "target_txt": target["txt"],
        "target_sha256": target["sha256"],
        "config_id": args.config_id,
        "sampling": args.sampling,
        "seed": manifest["seed"],
        "forward_seconds": round(elapsed, 4),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "model_module": os.path.abspath(model_module.__file__),
        "weights_sha256": manifest["dependencies"]["weights_sha256"],
        "fields": fields,
    }
    with open(os.path.join(args.out_dir, "forward_outputs.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print("wrote %s (%d fields, forward %.4f s)" % (args.out_dir, len(fields), elapsed))
    print("model module: %s" % summary["model_module"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
