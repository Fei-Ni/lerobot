#!/usr/bin/env python3
"""
PI0.5 离线 fit-check（lerobot-native，单脚本可跑）

对训完的 pi05 ckpt：
  - 载 LeRobotDataset 里指定 episode
  - deployment-style replan（每 execute_horizon 步重新 predict 一次 chunk）
  - 收集 GT vs predicted action，画每个 dim 一行 subplot 的拟合曲线
  - 落 JSON（schema 与 external_predictor_eval.py 兼容）

跑法：
  python train4arx/fit_check_pi05.py \
    --checkpoint /home/n84416302/lerobot/results/pi05_prgvla_sort_the_table_leftcam_v1_4gpu_b16_c60_s80k_restart_20260501/checkpoints/020000/pretrained_model \
    --dataset-root /home/n84416302/dataset/prgvla_sort_the_table_leftcam_v1 \
    --task-prompt "sort the table" \
    --task-slug   sort_table \
    --output-dir  /home/n84416302/lerobot/results/fit_check/sort_table_step20k

要点：
  - norm 空间用 **q01/q99**（和训练时 normalization_mapping={"ACTION":"QUANTILES",...} 完全一致），
    所以画出来的 norm 曲线就是模型实际看到的 [-1, 1] 空间。raw 单位的曲线另外再画一张。
  - chunk_size / n_action_steps / image_resolution 全部从 ckpt config 自动读，不要写死。
  - state、image keys 也从 config.input_features 自动取。
  - 默认对 dataset 里**所有 episode** 跑；想加速可以传 --num-episodes 截断。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


# ---------------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="ckpt dir 路径，要含 config.json + model.safetensors（即训练落盘的 pretrained_model/）")
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="LeRobot 数据集本地路径（要含 meta/ data/ videos/）")
    p.add_argument("--task-prompt", type=str, required=True,
                   help='任务文本，照训练原文，如 "sort the table" / "stack the cups"')
    p.add_argument("--task-slug", type=str, required=True,
                   help="输出文件名前缀和 JSON 标签，如 sort_table / stack_cups")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--episode-indices", type=int, nargs="+", default=None,
                   help="要评的 episode index 列表；缺省全跑")
    p.add_argument("--num-episodes", type=int, default=None,
                   help="如果不显式指定 --episode-indices，取前 N 个 episode；缺省全跑")
    p.add_argument("--execute-horizon", type=int, default=10,
                   help="replan 间隔（每 H 步重新调一次 model）")
    p.add_argument("--max-episode-steps", type=int, default=240,
                   help="单 episode 最多评多少步；防止超长 episode 拖慢")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    return p.parse_args()


# ---------------------------------------------------------------------------- #
# Loading
# ---------------------------------------------------------------------------- #


def load_policy(ckpt_dir: Path, device: str, dtype: str):
    """Load PI05 policy from a saved pretrained_model/ dir."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05 import PI05Policy

    config = PreTrainedConfig.from_pretrained(str(ckpt_dir))
    config.device = device
    config.dtype = dtype

    policy = PI05Policy.from_pretrained(str(ckpt_dir), config=config)
    policy.eval()
    return policy, config


def load_dataset(root: Path):
    """Load LeRobotDataset (handles both old and new import paths)."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(repo_id=str(root), root=str(root))


def get_action_dim(config) -> int:
    from lerobot.utils.constants import ACTION
    return int(config.output_features[ACTION].shape[0])


def get_action_quantiles_from_stats(stats: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pull action q01/q99 from dataset stats — same normalization training used (QUANTILES)."""
    a = stats["action"]
    if "q01" not in a or "q99" not in a:
        return None, None
    lo = np.asarray(a["q01"], dtype=np.float32).reshape(-1)
    hi = np.asarray(a["q99"], dtype=np.float32).reshape(-1)
    return lo, hi


def normalize_quantiles(raw: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Map raw → [-1, 1] using q01/q99 (matches QUANTILES normalization in training)."""
    rng = np.where(hi != lo, hi - lo, 1.0)
    return 2.0 * (raw - lo) / rng - 1.0


def summarize_errors(gt: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - gt
    abs_err = np.abs(err)
    sq_err = err * err
    return {
        "mae_mean": float(abs_err.mean()),
        "rmse_mean": float(math.sqrt(float(sq_err.mean()))),
        "mae_per_dim": abs_err.mean(axis=0).astype(float).tolist(),
        "rmse_per_dim": np.sqrt(sq_err.mean(axis=0)).astype(float).tolist(),
    }


# ---------------------------------------------------------------------------- #
# Inference
# ---------------------------------------------------------------------------- #


def build_batch(sample: dict, task_prompt: str, image_keys: list[str], state_key: str | None,
                device: str) -> dict:
    """Wrap a single LeRobotDataset sample into a B=1 batch the preprocessor expects."""
    batch = {"task": [task_prompt]}
    for k in image_keys:
        if k not in sample:
            raise KeyError(f"image key {k!r} missing from dataset sample. keys present: {list(sample)}")
        v = sample[k]
        if not torch.is_tensor(v):
            v = torch.as_tensor(v)
        batch[k] = v.unsqueeze(0).to(device)
    if state_key is not None:
        if state_key not in sample:
            raise KeyError(f"state key {state_key!r} missing from dataset sample.")
        v = sample[state_key]
        if not torch.is_tensor(v):
            v = torch.as_tensor(v)
        batch[state_key] = v.unsqueeze(0).to(device).float()
    return batch


@torch.no_grad()
def predict_chunk(policy, preprocessor, postprocessor, batch: dict) -> np.ndarray:
    """Run preprocessor → predict_action_chunk → postprocessor; return [chunk, action_dim] raw."""
    batch = preprocessor(batch)
    actions_norm = policy.predict_action_chunk(batch)        # [1, chunk, action_dim] in norm space
    # Postprocessor un-normalizes back to raw joint units.
    actions_raw = postprocessor(actions_norm)
    if torch.is_tensor(actions_raw):
        actions_raw = actions_raw.detach().cpu().numpy()
    arr = np.asarray(actions_raw, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise RuntimeError(f"expected predicted chunk shape [chunk, dim], got {arr.shape}")
    return arr


# ---------------------------------------------------------------------------- #
# Plotting
# ---------------------------------------------------------------------------- #


def render_episode_plot(figure_path: Path, *, steps: np.ndarray, gt_actions: np.ndarray,
                        pred_actions: np.ndarray, action_names: list[str],
                        title: str, y_label_suffix: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_dims = gt_actions.shape[1]
    fig, axes = plt.subplots(num_dims, 1, figsize=(14, max(4, 2.1 * num_dims)),
                             sharex=True, dpi=160)
    if num_dims == 1:
        axes = [axes]
    for dim_idx, ax in enumerate(axes):
        ax.plot(steps, gt_actions[:, dim_idx], label="gt", linewidth=1.2)
        ax.plot(steps, pred_actions[:, dim_idx], label="pred", linewidth=1.0)
        name = action_names[dim_idx] if dim_idx < len(action_names) else f"dim_{dim_idx}"
        ax.set_ylabel(f"{name}\n{y_label_suffix}", fontsize=8)
        ax.grid(True, alpha=0.25)
        if dim_idx == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("episode step")
    fig.suptitle(title)
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path)
    plt.close(fig)


# ---------------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------------- #


def main() -> int:
    args = parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- load dataset first so we can build processors with its stats
    print(f"[INFO] loading dataset from {args.dataset_root} ...", flush=True)
    ds = load_dataset(args.dataset_root)

    # --- load policy
    print(f"[INFO] loading checkpoint from {args.checkpoint} ...", flush=True)
    policy, config = load_policy(args.checkpoint, args.device, args.dtype)

    action_dim = get_action_dim(config)
    state_key = "observation.state" if "observation.state" in config.input_features else None
    image_keys = [k for k in config.input_features if k.startswith("observation.images.")]
    if not image_keys:
        print(f"[WARN] no image features in config.input_features={list(config.input_features)}",
              flush=True)
    chunk_size = int(config.chunk_size)
    print(f"[INFO] action_dim={action_dim} state_key={state_key} image_keys={image_keys} "
          f"chunk_size={chunk_size}", flush=True)

    # --- processors (use dataset stats to match training-time normalization)
    from lerobot.policies.pi05 import make_pi05_pre_post_processors
    preprocessor, postprocessor = make_pi05_pre_post_processors(
        config=config, dataset_stats=ds.meta.stats,
    )

    # --- pick episodes
    total_eps = len(ds.meta.episodes)
    if args.episode_indices is not None:
        episode_indices = list(map(int, args.episode_indices))
    elif args.num_episodes is not None:
        episode_indices = list(range(min(int(args.num_episodes), total_eps)))
    else:
        episode_indices = list(range(total_eps))
    print(f"[INFO] episodes total={total_eps}, evaluating={len(episode_indices)} "
          f"({episode_indices[:6]}{'...' if len(episode_indices) > 6 else ''})", flush=True)

    # --- action q01/q99 for norm-space metric (matches training QUANTILES normalization)
    lo, hi = get_action_quantiles_from_stats(ds.meta.stats)
    if lo is None or hi is None or lo.shape != (action_dim,) or hi.shape != (action_dim,):
        print(f"[WARN] stats action q01/q99 missing or shape mismatch; "
              "norm-space numbers will be skipped (raw is unaffected)", flush=True)
        lo, hi = None, None

    action_names = [f"joint_{i}" for i in range(action_dim - 1)] + ["gripper"]

    # --- run
    aggregate_norm_gt: list[np.ndarray] = []
    aggregate_norm_pred: list[np.ndarray] = []
    aggregate_raw_gt: list[np.ndarray] = []
    aggregate_raw_pred: list[np.ndarray] = []
    per_episode_summary: list[dict] = []
    plot_paths: list[str] = []

    for trajectory_id in episode_indices:
        ep_meta = ds.meta.episodes[int(trajectory_id)]
        ep_length = int(ep_meta["length"])
        total_steps = min(ep_length, int(args.max_episode_steps))
        from_idx = int(ep_meta["dataset_from_index"])

        policy.reset()  # clear action queue / per-episode state
        records: list[dict] = []

        step_idx = 0
        plan_count = 0
        while step_idx < total_steps:
            sample = ds[from_idx + step_idx]
            batch = build_batch(sample, args.task_prompt, image_keys, state_key, args.device)
            chunk_raw = predict_chunk(policy, preprocessor, postprocessor, batch)  # [chunk, dim]
            plan_count += 1
            if chunk_raw.shape[1] != action_dim:
                raise RuntimeError(f"predicted chunk dim {chunk_raw.shape[1]} != action_dim {action_dim}")

            execute_count = min(int(args.execute_horizon), chunk_raw.shape[0],
                                total_steps - step_idx)
            if execute_count <= 0:
                raise RuntimeError(f"step={step_idx} 处无可执行 action")
            for local_idx in range(execute_count):
                ds_step = step_idx + local_idx
                gt_sample = sample if local_idx == 0 else ds[from_idx + ds_step]
                gt_action = np.asarray(gt_sample["action"], dtype=np.float32).reshape(-1)
                if gt_action.shape[0] != action_dim:
                    raise RuntimeError(f"GT action shape={gt_action.shape}, expected ({action_dim},)")
                records.append({
                    "trajectory_id": int(trajectory_id),
                    "step": int(ds_step),
                    "plan_origin_step": int(step_idx),
                    "chunk_offset": int(local_idx),
                    "task": args.task_prompt,
                    "gt_action_raw": gt_action.tolist(),
                    "pred_action_raw": chunk_raw[local_idx].astype(float).tolist(),
                })
            step_idx += execute_count

        steps = np.asarray([r["step"] for r in records], dtype=np.int64)
        gt_raw = np.asarray([r["gt_action_raw"] for r in records], dtype=np.float32)
        pred_raw = np.asarray([r["pred_action_raw"] for r in records], dtype=np.float32)

        aggregate_raw_gt.append(gt_raw)
        aggregate_raw_pred.append(pred_raw)
        raw_metrics = summarize_errors(gt_raw, pred_raw)

        if lo is not None and hi is not None:
            gt_norm = normalize_quantiles(gt_raw, lo, hi)
            pred_norm = normalize_quantiles(pred_raw, lo, hi)
            aggregate_norm_gt.append(gt_norm)
            aggregate_norm_pred.append(pred_norm)
            norm_metrics = summarize_errors(gt_norm, pred_norm)
        else:
            gt_norm = pred_norm = None
            norm_metrics = None

        title_base = (f"{args.task_slug} | pi05 fit-check | ep{trajectory_id:03d} | "
                      f"replan={args.execute_horizon} | plans={plan_count}")
        raw_path = output_dir / f"episode_{trajectory_id:03d}_{args.task_slug}_action_dims_raw.png"
        render_episode_plot(
            raw_path, steps=steps, gt_actions=gt_raw, pred_actions=pred_raw,
            action_names=action_names,
            title=f"{title_base}\nraw MAE={raw_metrics['mae_mean']:.5f}, RMSE={raw_metrics['rmse_mean']:.5f}",
            y_label_suffix="raw",
        )
        plot_paths.append(str(raw_path))
        if norm_metrics is not None:
            norm_path = output_dir / f"episode_{trajectory_id:03d}_{args.task_slug}_action_dims_normalized.png"
            render_episode_plot(
                norm_path, steps=steps, gt_actions=gt_norm, pred_actions=pred_norm,
                action_names=action_names,
                title=f"{title_base}\nnorm MAE={norm_metrics['mae_mean']:.5f}, RMSE={norm_metrics['rmse_mean']:.5f}",
                y_label_suffix="norm",
            )
            plot_paths.append(str(norm_path))

        per_episode_summary.append({
            "trajectory_id": int(trajectory_id),
            "task": args.task_prompt,
            "task_slug": args.task_slug,
            "total_steps": int(total_steps),
            "plans": int(plan_count),
            "raw": raw_metrics,
            "normalized": norm_metrics,
            "raw_figure": str(raw_path),
            "normalized_figure": str(norm_path) if norm_metrics is not None else None,
        })
        print(f"[OK] ep{trajectory_id:03d}  raw_MAE={raw_metrics['mae_mean']:.5f}"
              + (f"  norm_MAE={norm_metrics['mae_mean']:.5f}" if norm_metrics is not None else ""),
              flush=True)

    all_raw_gt = np.concatenate(aggregate_raw_gt, axis=0)
    all_raw_pred = np.concatenate(aggregate_raw_pred, axis=0)
    aggregate = {
        "num_episodes": len(per_episode_summary),
        "num_records": int(all_raw_gt.shape[0]),
        "raw": summarize_errors(all_raw_gt, all_raw_pred),
    }
    if aggregate_norm_gt:
        all_norm_gt = np.concatenate(aggregate_norm_gt, axis=0)
        all_norm_pred = np.concatenate(aggregate_norm_pred, axis=0)
        aggregate["normalized"] = summarize_errors(all_norm_gt, all_norm_pred)
    else:
        aggregate["normalized"] = None

    summary = {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "harness": "fit_check_pi05.py (lerobot-native)",
            "task_slug": args.task_slug,
            "task_prompt": args.task_prompt,
            "checkpoint": str(args.checkpoint),
            "dataset_root": str(args.dataset_root),
            "selected_trajectory_ids": episode_indices,
            "execute_horizon": int(args.execute_horizon),
            "max_episode_steps": int(args.max_episode_steps),
            "chunk_size": chunk_size,
            "action_dim": action_dim,
            "action_names": action_names,
            "state_key": state_key,
            "image_keys": image_keys,
            "device": args.device,
            "dtype": args.dtype,
            "norm_space": "q01/q99 from dataset stats — matches training-time "
                          "QUANTILES normalization (训推一致). raw MAE is in joint units.",
        },
        "aggregate": aggregate,
        "episodes": per_episode_summary,
        "plots": plot_paths,
    }
    summary_path = output_dir / "action_fit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[OK] aggregate raw_MAE={aggregate['raw']['mae_mean']:.5f}", flush=True)
    if aggregate.get("normalized") is not None:
        print(f"[OK] aggregate norm_MAE={aggregate['normalized']['mae_mean']:.5f}", flush=True)
    print(f"[OK] wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
