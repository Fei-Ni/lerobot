#!/usr/bin/env python3
"""
跨 codebase 的 prgvla 离线 fit-check harness（model-agnostic）

用途：
  让外部 codebase（如 Pi0）在 prgvla 同一份数据上跑出和我们这边
  examples/REAL/prgvla/render_prgvla_action_fit_curves.py 完全可比的：
    - per-episode 8-dim action 拟合曲线（GT vs pred）
    - aggregate raw / normalized MAE + RMSE
    - per-dim MAE / RMSE
    - JSON schema 与我们一致

你（外部 codebase 的人）需要做的事：
  1. 把下面 my_predict() 替换成调你自己模型的实现，返回 raw 关节单位的
     [H, 8] action chunk（H 是你模型的 chunk 长度，>= execute_horizon
     即可；返回时第一行是当前步的 action，往后是未来 action）
  2. 跑：
       python external_predictor_eval.py \
         --dataset-root <PATH>/prgvla_sort_table_310left_nonbad_starvla \
         --action-stats <PATH>/dataset_statistics.json \
         --task-prompt "sort the table" \
         --task-slug   sort_table \
         --output-dir  <OUT>
  3. 输出：<OUT>/action_fit_summary.json + 12 张 per-episode PNG

依赖：lerobot, pillow, numpy, matplotlib, omegaconf 不需要
（这个脚本完全不依赖 starVLA repo）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
# 跟我们这边完全一致的固定评价口径（不要改）
# --------------------------------------------------------------------------- #

EPISODES_BY_TASK: dict[str, list[int]] = {
    "sort_table": [0, 2, 5, 7, 10, 11, 15, 17, 19, 22, 25, 27],
    "stack_cups": [0, 2, 5, 7, 10, 11, 15, 17, 19, 22, 25, 28],
}
EXECUTE_HORIZON = 10            # replan 间隔（不是模型 chunk 长度）
MAX_EPISODE_STEPS = 240         # 每个 episode 最多评 240 步
IMAGE_SIZE = (224, 224)         # 模型输入图像尺寸
ACTION_DIM = 8                  # joint_0..joint_6 + gripper
ACTION_NAMES = [
    "joint_0", "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6", "gripper",
]
PRIMARY_VIDEO_KEY = "observation.images.primary_image"   # 训练时用的视图

# --------------------------------------------------------------------------- #
# 你（外部 codebase 的人）要替换的部分
# --------------------------------------------------------------------------- #

def my_predict(
    image: Image.Image,
    task: str,
    *,
    first_query: bool,
    step: int,
) -> np.ndarray:
    """
    你的模型推理入口。返回 shape [H, 8] 的 numpy float32，**raw 关节单位**
    （不是 [-1, 1] 归一化空间）。

    参数：
      image          PIL.Image，已经 resize 到 224×224 RGB
      task           任务文本，如 "sort the table" / "stack the cups"
      first_query    新 episode 的第一帧时为 True，episode 中其他帧 False
                       （如果你的模型有 per-episode 状态需要 reset，用这个标志位）
      step           episode 内的步数计数（首查询 0，之后 +EXECUTE_HORIZON）

    要求：
      - 返回 shape (H, 8)，H >= EXECUTE_HORIZON（10）即可
      - 返回值已经反归一化到原始关节单位，**不要**返回 [-1, 1] 的值
      - 顺序：第 0 行是"当前步"的 action，第 i 行是"未来第 i 步"的 action
    """
    raise NotImplementedError(
        "请实现你的模型推理。返回 shape (H, 8) 的 raw 关节单位 action chunk。\n"
        "示例骨架：\n"
        "    if first_query:\n"
        "        self.episode_state = reset_state()\n"
        "    inputs = preprocess(image, task)\n"
        "    chunk_norm = self.model(inputs)        # [H, 8] in [-1, 1]\n"
        "    chunk_raw = (chunk_norm + 1) / 2 * (action_max - action_min) + action_min\n"
        "    return chunk_raw.astype(np.float32)\n"
    )


# --------------------------------------------------------------------------- #
# 以下是 harness 实现（model-agnostic，不要改）
# --------------------------------------------------------------------------- #

def load_action_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """从 dataset_statistics.json 读 action.joints 的 min / max。"""
    s = json.loads(path.read_text())
    if len(s) != 1:
        raise ValueError(f"dataset_statistics.json 应该只有一个 unnorm_key，发现：{list(s.keys())}")
    [(_unnorm_key, body)] = s.items()
    a = body["action"]["joints"]
    lo = np.asarray(a["min"], dtype=np.float32)
    hi = np.asarray(a["max"], dtype=np.float32)
    if lo.shape != (ACTION_DIM,) or hi.shape != (ACTION_DIM,):
        raise ValueError(f"action.joints min/max 维度应为 ({ACTION_DIM},)，得到 lo={lo.shape} hi={hi.shape}")
    return lo, hi


def normalize_min_max(raw: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """raw → [-1, 1]"""
    rng = np.where(hi != lo, hi - lo, 1.0)
    return 2.0 * (raw - lo) / rng - 1.0


def summarize_errors(gt: np.ndarray, pred: np.ndarray) -> dict:
    """返回 mae_mean / rmse_mean / mae_per_dim / rmse_per_dim，schema 与 starVLA 端一致。"""
    err = pred - gt
    abs_err = np.abs(err)
    sq_err = err * err
    return {
        "mae_mean": float(abs_err.mean()),
        "rmse_mean": float(math.sqrt(float(sq_err.mean()))),
        "mae_per_dim": abs_err.mean(axis=0).astype(float).tolist(),
        "rmse_per_dim": np.sqrt(sq_err.mean(axis=0)).astype(float).tolist(),
    }


def render_episode_plot(
    figure_path: Path,
    *,
    steps: np.ndarray,
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    title: str,
    y_label_suffix: str,
) -> None:
    """8 个 subplot，每个 subplot 一个 dim，GT vs pred 曲线。与 starVLA 端格式一致。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_dims = gt_actions.shape[1]
    fig, axes = plt.subplots(num_dims, 1, figsize=(14, max(4, 2.1 * num_dims)), sharex=True, dpi=160)
    if num_dims == 1:
        axes = [axes]
    for dim_idx, ax in enumerate(axes):
        ax.plot(steps, gt_actions[:, dim_idx], label="gt", linewidth=1.2)
        ax.plot(steps, pred_actions[:, dim_idx], label="pred", linewidth=1.0)
        ax.set_ylabel(f"{ACTION_NAMES[dim_idx]}\n{y_label_suffix}", fontsize=8)
        ax.grid(True, alpha=0.25)
        if dim_idx == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("episode step")
    fig.suptitle(title)
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path)
    plt.close(fig)


def evaluate(
    *,
    predict_fn: Callable[[Image.Image, str], np.ndarray],
    dataset_root: Path,
    action_stats_path: Path,
    task_prompt: str,
    task_slug: str,
    output_dir: Path,
    episode_indices: Iterable[int] | None = None,
    execute_horizon: int = EXECUTE_HORIZON,
    max_episode_steps: int = MAX_EPISODE_STEPS,
) -> dict:
    """
    主入口。返回 summary dict（也写到 output_dir/action_fit_summary.json）。
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if episode_indices is None:
        if task_slug not in EPISODES_BY_TASK:
            raise ValueError(f"未知 task_slug={task_slug}；支持 {list(EPISODES_BY_TASK)}")
        episode_indices = EPISODES_BY_TASK[task_slug]
    episode_indices = list(map(int, episode_indices))

    lo, hi = load_action_stats(Path(action_stats_path))
    print(f"[INFO] action.joints lo={lo.tolist()}", flush=True)
    print(f"[INFO] action.joints hi={hi.tolist()}", flush=True)

    # 用 LeRobotDataset 读取数据
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # newer lerobot
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # alt path
    ds = LeRobotDataset(repo_id=str(dataset_root), root=str(dataset_root), local_files_only=True)

    aggregate_norm_gt: list[np.ndarray] = []
    aggregate_norm_pred: list[np.ndarray] = []
    aggregate_raw_gt: list[np.ndarray] = []
    aggregate_raw_pred: list[np.ndarray] = []
    per_episode_summary: list[dict] = []
    plot_paths: list[str] = []

    for trajectory_id in episode_indices:
        ep_meta = ds.meta.episodes[trajectory_id]
        ep_length = int(ep_meta["length"])
        total_steps = min(ep_length, int(max_episode_steps))
        from_idx = int(ds.episode_data_index["from"][trajectory_id])
        # to_idx = int(ds.episode_data_index["to"][trajectory_id])  # 未用

        records = []
        first_query = True
        step_idx = 0
        while step_idx < total_steps:
            sample_idx = from_idx + step_idx
            sample = ds[sample_idx]
            # image: lerobot 默认是 (C, H, W) tensor float32 [0, 1]，需要转 PIL
            img = sample[PRIMARY_VIDEO_KEY]
            if hasattr(img, "numpy"):
                img = img.numpy()
            arr = np.asarray(img)
            if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            pil = Image.fromarray(arr).convert("RGB").resize(IMAGE_SIZE)

            pred_chunk_raw = predict_fn(
                pil, task_prompt,
                first_query=first_query, step=int(step_idx),
            )
            first_query = False
            pred_chunk_raw = np.asarray(pred_chunk_raw, dtype=np.float32)
            if pred_chunk_raw.ndim != 2 or pred_chunk_raw.shape[1] != ACTION_DIM:
                raise RuntimeError(
                    f"predict_fn 返回 shape 应为 [H, {ACTION_DIM}]，得到 {pred_chunk_raw.shape}"
                )
            chunk_h = pred_chunk_raw.shape[0]
            execute_count = min(int(execute_horizon), chunk_h, total_steps - step_idx)
            if execute_count <= 0:
                raise RuntimeError(f"step={step_idx} 处无可执行 action")
            for local_idx in range(execute_count):
                ds_step = step_idx + local_idx
                gt_sample = sample if local_idx == 0 else ds[from_idx + ds_step]
                gt_action = np.asarray(gt_sample["action"], dtype=np.float32)
                if gt_action.shape != (ACTION_DIM,):
                    raise RuntimeError(f"GT action shape={gt_action.shape}, 期望 ({ACTION_DIM},)")
                pred_action_raw = pred_chunk_raw[local_idx]
                records.append({
                    "trajectory_id": int(trajectory_id),
                    "step": int(ds_step),
                    "plan_origin_step": int(step_idx),
                    "chunk_offset": int(local_idx),
                    "task": task_prompt,
                    "gt_action_raw": gt_action.tolist(),
                    "pred_action_raw": pred_action_raw.astype(float).tolist(),
                })
            step_idx += execute_count

        # 这一条 episode 的指标
        steps = np.asarray([r["step"] for r in records], dtype=np.int64)
        gt_raw = np.asarray([r["gt_action_raw"] for r in records], dtype=np.float32)
        pred_raw = np.asarray([r["pred_action_raw"] for r in records], dtype=np.float32)
        gt_norm = normalize_min_max(gt_raw, lo, hi)
        pred_norm = normalize_min_max(pred_raw, lo, hi)

        aggregate_raw_gt.append(gt_raw)
        aggregate_raw_pred.append(pred_raw)
        aggregate_norm_gt.append(gt_norm)
        aggregate_norm_pred.append(pred_norm)

        norm_metrics = summarize_errors(gt_norm, pred_norm)
        raw_metrics = summarize_errors(gt_raw, pred_raw)
        title_base = (
            f"{task_slug} | external predictor | ep{trajectory_id:03d} | "
            f"replan={execute_horizon}"
        )
        norm_path = output_dir / f"episode_{trajectory_id:03d}_{task_slug}_action_dims_normalized.png"
        raw_path = output_dir / f"episode_{trajectory_id:03d}_{task_slug}_action_dims_raw.png"
        render_episode_plot(
            norm_path, steps=steps, gt_actions=gt_norm, pred_actions=pred_norm,
            title=f"{title_base}\nnormalized MAE={norm_metrics['mae_mean']:.5f}, RMSE={norm_metrics['rmse_mean']:.5f}",
            y_label_suffix="norm",
        )
        render_episode_plot(
            raw_path, steps=steps, gt_actions=gt_raw, pred_actions=pred_raw,
            title=f"{title_base}\nraw MAE={raw_metrics['mae_mean']:.5f}, RMSE={raw_metrics['rmse_mean']:.5f}",
            y_label_suffix="raw",
        )
        plot_paths += [str(norm_path), str(raw_path)]
        per_episode_summary.append({
            "trajectory_id": int(trajectory_id),
            "task": task_prompt,
            "task_slug": task_slug,
            "total_steps": int(total_steps),
            "normalized": norm_metrics,
            "raw": raw_metrics,
            "normalized_figure": str(norm_path),
            "raw_figure": str(raw_path),
        })
        print(
            f"[OK] ep{trajectory_id:03d}  raw_MAE={raw_metrics['mae_mean']:.5f}  norm_MAE={norm_metrics['mae_mean']:.5f}",
            flush=True,
        )

    all_norm_gt = np.concatenate(aggregate_norm_gt, axis=0)
    all_norm_pred = np.concatenate(aggregate_norm_pred, axis=0)
    all_raw_gt = np.concatenate(aggregate_raw_gt, axis=0)
    all_raw_pred = np.concatenate(aggregate_raw_pred, axis=0)

    summary = {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "harness": "external_predictor_eval.py",
            "task_slug": task_slug,
            "task_prompt": task_prompt,
            "dataset_root": str(dataset_root),
            "action_stats_path": str(action_stats_path),
            "action_min": lo.astype(float).tolist(),
            "action_max": hi.astype(float).tolist(),
            "action_names": ACTION_NAMES,
            "execute_horizon": int(execute_horizon),
            "max_episode_steps": int(max_episode_steps),
            "image_size": list(IMAGE_SIZE),
            "primary_video_key": PRIMARY_VIDEO_KEY,
            "selected_trajectory_ids": episode_indices,
        },
        "aggregate": {
            "num_episodes": len(per_episode_summary),
            "num_records": int(all_norm_gt.shape[0]),
            "normalized": summarize_errors(all_norm_gt, all_norm_pred),
            "raw": summarize_errors(all_raw_gt, all_raw_pred),
        },
        "episodes": per_episode_summary,
        "plots": plot_paths,
    }
    summary_path = output_dir / "action_fit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[OK] aggregate raw_MAE={summary['aggregate']['raw']['mae_mean']:.5f}", flush=True)
    print(f"[OK] aggregate norm_MAE={summary['aggregate']['normalized']['mae_mean']:.5f}", flush=True)
    print(f"[OK] wrote {summary_path}", flush=True)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="prgvla 任务的 LeRobot 数据集本地路径（要含 meta/ data/ videos/）")
    p.add_argument("--action-stats", type=Path, required=True,
                   help="dataset_statistics.json 路径，用我们 HF release 里任一变体里的版本以保证跨 codebase 公平")
    p.add_argument("--task-prompt", type=str, required=True,
                   help='任务文本，照训练原文：sort_table 用 "sort the table"，stack_cups 用 "stack the cups"')
    p.add_argument("--task-slug", type=str, required=True, choices=list(EPISODES_BY_TASK),
                   help="决定用哪 12 个 episode")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--execute-horizon", type=int, default=EXECUTE_HORIZON)
    p.add_argument("--max-episode-steps", type=int, default=MAX_EPISODE_STEPS)
    args = p.parse_args()

    return 0 if evaluate(
        predict_fn=my_predict,
        dataset_root=args.dataset_root,
        action_stats_path=args.action_stats,
        task_prompt=args.task_prompt,
        task_slug=args.task_slug,
        output_dir=args.output_dir,
        execute_horizon=args.execute_horizon,
        max_episode_steps=args.max_episode_steps,
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
