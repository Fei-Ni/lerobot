# 跨 codebase 复刻 prgvla 离线 fit-check（给 Pi0 / 其他 VLA 的同学）

本指南说明：你们在自己的 codebase（如 Pi0）上用 prgvla 同一份数据训完模型后，怎么用我们这边的同款评价方式跑出**与 starVLA 端可直接对比**的指标和曲线图。

## 0. 一句话说明

复制两个文件 + 实现一个函数（你的模型推理）+ 跑一行命令 → 拿到与我们这边
`action_fit_summary.json` schema 完全一致的结果（含 12 张 per-episode 8-dim 拟合曲线 + raw / norm MAE）。

## 1. 你需要从我们这边拷过去的两样东西

### a) `external_predictor_eval.py`（评价 harness，model-agnostic）

文件位置：`deployment/model_server/external_predictor_eval.py`（在 starVLA repo 里）

这个文件**不依赖 starVLA**，只用 `lerobot, numpy, pillow, matplotlib`。直接放到你们 codebase 任意目录即可。

### b) `dataset_statistics.json`（action min/max，确保归一化口径一致）

直接从我们的 HF release 拿任一变体里的版本，**两个 codebase 用同一个 stats 文件，归一化空间才能严格对齐**：

```bash
huggingface-cli download spikefly/starvla-prgvla-310left-nonbad-chunk60 \
  --include "sort_table_baseline_60k/dataset_statistics.json" \
  --local-dir <YOUR_LOCAL>/prgvla_stats
# 文件落到：<YOUR_LOCAL>/prgvla_stats/sort_table_baseline_60k/dataset_statistics.json
```

> 同一个 task 的 baseline 和 dinoquery 用的 stats 是相同的（都来自同一份训练数据），任选一个变体的即可。

## 2. 评价数据集

你们既然在 prgvla 同一份数据上训了 Pi0，本地应该已经有 LeRobot 格式的转换数据集（即我们这边的
`prgvla_sort_table_310left_nonbad_starvla` / `prgvla_stack_cups_310left_nonbad_starvla`）。
harness 直接用 `LeRobotDataset` 读它。

如果你们没有这个 LeRobot-format 转换数据集（只有原始 HDF5），先做转换；我们这边的转换脚本在 starVLA repo
`examples/REAL/prgvla/convert_prgvla_raw_to_starvla.py`，可以参考。

## 3. 环境依赖

```bash
pip install "lerobot>=0.1" pillow numpy matplotlib
```

不需要 starVLA、Qwen3-VL、DINOv2 这些——评价 harness 只调你的 predict 函数 + 标准 LeRobot 数据访问。

## 4. 你要实现的部分（一个函数）

打开 `external_predictor_eval.py`，找到 `def my_predict(...)`，把它替换成调你自己 Pi0 模型的实现：

```python
def my_predict(image, task, *, first_query, step):
    """
    输入:
      image       PIL.Image，已 resize 到 224×224 RGB
      task        "sort the table" 或 "stack the cups"（必须照训练原文）
      first_query 新 episode 的第一帧时为 True
      step        episode 内的步数（首查询 0，之后递增 EXECUTE_HORIZON=10）

    返回:
      np.ndarray, shape (H, 8), dtype float32, raw 关节单位（不是归一化空间）
      H 是你的模型 chunk 长度，>= 10 即可；第 0 行是当前步，第 i 行是未来第 i 步
    """
    # 例：
    # if first_query:
    #     self.episode_state = reset()
    # chunk_norm = self.pi0_model.predict(image, task)   # [H, 8] in [-1, 1]
    # chunk_raw  = unnormalize_with_your_own_stats(chunk_norm)
    # return chunk_raw.astype(np.float32)
```

**要点**：

- 返回 raw 关节单位（你自己的反归一化口径，用你训练时的 min/max）。harness 内部会用我们 HF 的 stats
  对你的 raw output 重做一次归一化得到 norm 空间数字，确保两边可对比。
- 如果 H >= 10，harness 只取前 10 步执行（与我们的 `execute_horizon=10` 完全对齐）。
- `first_query=True` 是在每个 episode 开头唤起一次，方便你 reset 内部状态（如 history buffer）。

## 5. 跑

```bash
# sort_table 任务
python external_predictor_eval.py \
  --dataset-root  <YOUR>/prgvla_sort_table_310left_nonbad_starvla \
  --action-stats  <YOUR_LOCAL>/prgvla_stats/sort_table_baseline_60k/dataset_statistics.json \
  --task-prompt   "sort the table" \
  --task-slug     sort_table \
  --output-dir    <OUT>/pi0_sort_table_eval

# stack_cups 任务
python external_predictor_eval.py \
  --dataset-root  <YOUR>/prgvla_stack_cups_310left_nonbad_starvla \
  --action-stats  <YOUR_LOCAL>/prgvla_stats/sort_table_baseline_60k/dataset_statistics.json \
  --task-prompt   "stack the cups" \
  --task-slug     stack_cups \
  --output-dir    <OUT>/pi0_stack_cups_eval
```

跑完每个 task 大概 5-10 分钟（12 个 episode × 每 episode ~24 个 plan 步）。

## 6. 输出

两个 task 各产出：

```
<OUT>/pi0_<task>_eval/
  action_fit_summary.json                                   # 顶层指标
  episode_000_<task>_action_dims_normalized.png             # 8-dim 曲线（norm 空间）
  episode_000_<task>_action_dims_raw.png                    # 8-dim 曲线（raw 关节空间）
  ... (× 12 episodes)
```

`action_fit_summary.json` 的 schema 跟我们这边一模一样，关键字段：

```json
{
  "metadata": {...},
  "aggregate": {
    "num_episodes": 12,
    "num_records": ~2700,
    "normalized": {"mae_mean": 0.xxx, "rmse_mean": 0.xxx, "mae_per_dim": [...], "rmse_per_dim": [...]},
    "raw":        {"mae_mean": 0.xxx, "rmse_mean": 0.xxx, "mae_per_dim": [...], "rmse_per_dim": [...]}
  },
  "episodes": [{...}, {...}, ...]
}
```

## 7. 我们这边的参考数字（用来做对比的 baseline）

同样 12 episode、`execute_horizon=10`、`do_sample=False`、deployment-style replan 跑出的：

| Task | 模型 | 步数 | raw MAE | norm MAE |
|---|---|---:|---:|---:|
| sort_table | starVLA QwenGR00T baseline | 60k | 0.1107 | 0.1476 |
| sort_table | starVLA ProgressVLA dinoquery vitg14 | 40k | **0.1034** | **0.1373** |
| stack_cups | starVLA QwenGR00T baseline | 60k | 0.0792 | 0.1230 |
| stack_cups | starVLA ProgressVLA dinoquery vitg14 | 40k | **0.0731** | **0.1135** |

> 如果 Pi0 跑出来 raw MAE 比我们这边好，说明 Pi0 在该数据上拟合得更好；反之亦然。两边的 raw 单位完全一致（都是同一份 prgvla 数据的关节角 + gripper 位置），可以直接比大小。

## 8. 公平性 checklist（确保两边可比）

要拿到一个 fair 的对比，请确认：

- [ ] **数据集**：用同一份 LeRobot 转换 prgvla 数据集（episode 索引、step 顺序完全一致）
- [ ] **task 文本**：照训练原文，不要改写。`sort the table` / `stack the cups` 两个**不要改大小写或加字符**
- [ ] **图像视图**：都用 `observation.images.primary_image`（即原始 prgvla 的静态左相机 `310222076420`）
- [ ] **图像尺寸**：harness 里强制 resize 到 224×224；如果你们模型用别的尺寸，确保你们 predict 内部自己处理（harness 传过来的是 224×224）
- [ ] **action 顺序**：8 维顺序固定 `joint_0..joint_6, gripper`（与我们 `dataset_statistics.json` 保持一致）
- [ ] **action 反归一化口径**：你们用 **自己训练时的 min/max** 把模型输出反归一化回 raw；harness 用我们 HF release 的 stats 把 raw 再归一化到 norm 空间——这一步保证 norm 空间口径相同
- [ ] **execute_horizon**：固定 10，不要改成 1 / 60；这一步影响 replan 频率，从而影响整体 MAE
- [ ] **`do_sample=False`**：如果你们模型是 stochastic 的（如 diffusion），用 deterministic / fixed seed 模式；否则跑两次会得到不同结果，影响可重复性

## 9. 输出 / 排错检查

- 如果 raw MAE 远超 0.3：检查 task 文本是否一致、图像视图是否对、action 顺序是否对（最常见是 gripper 在第 0 维而不是第 7 维）
- 如果 norm MAE 远比 raw MAE 大很多（差几倍）：八成是 action min/max 路径错了或者文件版本不对——确认 stats JSON 的 `action.joints.min/max` 长度都是 8
- 如果 LeRobot 读图像失败：检查是不是 LeRobot 版本太老（`pip install -U lerobot`）；harness 里有 `(C, H, W) <-> (H, W, C)` 自动转
- 如果某个 episode 报"无可执行 action"：你的 predict 返回的 chunk_h < 10；要么补长 chunk，要么调小 `--execute-horizon`（但这就和我们这边不可比了）

## 10. 文件清单（拷到你们集群上）

只需要：

```
external_predictor_eval.py                # harness（model-agnostic）
dataset_statistics.json                    # action min/max（从我们 HF 拿）
EXTERNAL_EVAL_GUIDE.md                     # 这份文档（可选）
```

不需要带 starVLA 任何模型代码，也不需要带 Qwen3-VL / DINOv2。
