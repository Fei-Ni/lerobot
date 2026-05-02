# PRGVLA -> PI05 Notes

这套文件是给 `zhuoKCL/prgvla` 额外微调 `pi05` 用的，按你的要求做了这些约定：

- 拆成两个任务分别训练：
  - `sorting` -> `sort the table`
  - `stack` -> `stack the cups`
- 只用单相机：
  - `310222076420_left.mp4`
- `pi05` 训练配置和现在仓库里的 ARX 配置基本一致
- `chunk_size=60`
- `n_action_steps=60`
- `steps=80000`

## 1. 原始数据结构

`prgvla` 不是 LeRobot v3。

它在 Hugging Face 上的结构是：

```text
sorting/<episode_time>/trajectory.h5
sorting/<episode_time>/recordings/310222076420_left.mp4
stack/<episode_time>/trajectory.h5
stack/<episode_time>/recordings/310222076420_left.mp4
```

当前检查到的 episode 数量是：

- `sorting`: 36
- `stack`: 30

采样检查结果：

- 单相机视频分辨率：`640x480`
- 视频帧率：`10 FPS`
- `trajectory.h5` 的长度和视频帧数一致
- 机器人状态/动作是单臂 `7 joints + gripper`

## 2. 转换脚本

转换脚本：

- `train4arx/convert_prgvla_to_lerobot.py`

它会：

1. 从 `zhuoKCL/prgvla` 下载选定任务下的：
   - `trajectory.h5`
   - `recordings/310222076420_left.mp4`
2. 读取：
   - `observation/robot_state/joint_positions`
   - `observation/robot_state/gripper_position`
   - `action/joint_position`
   - `action/gripper_position`
3. 生成 LeRobot v3 数据集：
   - `observation.state`: 8 维
   - `action`: 8 维
   - `observation.images.camera_left`: 单相机视频
4. 把任务文本固定成：
   - `sort the table`
   - `stack the cups`

注意：

- 这里**不使用**原始 h5 里的 `current_task` 字段，因为采样看起来它并不能可靠反映这两个任务。

## 3. 转换命令

先进入环境：

```bash
conda activate lerobot-new
cd /home/n84416302/lerobot
```

转换 `sorting`：

```bash
python train4arx/convert_prgvla_to_lerobot.py --task sorting
```

转换 `stack`：

```bash
python train4arx/convert_prgvla_to_lerobot.py --task stack
```

一次性都转：

```bash
python train4arx/convert_prgvla_to_lerobot.py --task all
```

默认输出目录：

- `/home/n84416302/dataset/prgvla_sort_the_table_leftcam_v1`
- `/home/n84416302/dataset/prgvla_stack_the_cups_leftcam_v1`

如果只是先做 smoke test，可以限制 episode 数：

```bash
python train4arx/convert_prgvla_to_lerobot.py --task sorting --max-episodes 1
```

## 4. 训练脚本

两个训练脚本：

- `train4arx/train_pi05_prgvla_sort_the_table_leftcam_v1.sh`
- `train4arx/train_pi05_prgvla_stack_the_cups_leftcam_v1.sh`

两个 submit 脚本：

- `submit_train_pi05_prgvla_sort_the_table_leftcam_v1.sh`
- `submit_train_pi05_prgvla_stack_the_cups_leftcam_v1.sh`

这两套脚本默认：

- 4 GPU
- `bf16`
- `batch_size=16`
- `chunk_size=60`
- `n_action_steps=60`
- `steps=80000`
- 只吃：
  - `observation.state`
  - `observation.images.camera_left`

## 5. 训练前需要的环境变量

`pi05` 仍然需要能访问 `google/paligemma-3b-pt-224` 的 token：

```bash
export HF_TOKEN=你的_hf_token
```

这次新脚本默认把 wandb 关掉了：

```bash
--wandb.enable=false
```

所以不要求 `WANDB_API_KEY`。

## 6. 训练命令

直接跑：

```bash
bash train4arx/train_pi05_prgvla_sort_the_table_leftcam_v1.sh
bash train4arx/train_pi05_prgvla_stack_the_cups_leftcam_v1.sh
```

或者走 slurm：

```bash
sbatch submit_train_pi05_prgvla_sort_the_table_leftcam_v1.sh
sbatch submit_train_pi05_prgvla_stack_the_cups_leftcam_v1.sh
```

## 7. 运行进展 / 故障记录

### 7.1 第一次提交（2026-05-01 00:48，jobs 64548 / 64549）

- 输出目录：`results/pi05_prgvla_{stack_the_cups,sort_the_table}_leftcam_v1_4gpu_b16_c60_s80k`（不带 `_restart_` 后缀）。
- 两个 job 都跑到第一次 `save_checkpoint` 时挂掉（11:12 左右），根因相同：

  ```
  safetensors_rust.SafetensorError: Error while serializing:
  I/O error: Disk quota exceeded (os error 122)
  ```

- 这两个 results 目录里只剩空的 `checkpoints/`，没有 step 落盘。可清理。
- 磁盘配额问题已在 2026-05-02 之前由用户单独解决。

### 7.2 第二次提交（2026-05-01 23:15，jobs 64703 / 64704）

submit 脚本改成 restart 版（输出目录加后缀 `_restart_20260501`，避免和老 results 冲突）。

| Job | 任务 | 结果 |
|---|---|---|
| 64703 | `stack_the_cups` restart | **正常**：23:27 起跑，至 2026-05-02 00:00 已到 `step≈3K / 80K`，loss≈0.029 |
| 64704 | `sort_the_table` restart | **23:32 死掉**，没进入训练 |

64704 的失败 traceback：

```
make_dataset → LeRobotDatasetMetadata._load_metadata
 → load_episodes → load_nested_dataset
 → datasets.Dataset.from_parquet → builder.download_and_prepare
 → FileLock(lock_path)._acquire
 → os.open(O_RDWR|O_CREAT|O_TRUNC)
 PermissionError: [Errno 1] Operation not permitted:
   /home/n84416302/.cache/huggingface/datasets/parquet/
   default-1283ec8211bd548d/0.0.0/
   ec3958b52035e87929642b609b5a781870321434b151a14fbe8fd5a26ff494bb_builder.lock
```

诊断：

- `~/.cache/huggingface/...` 在 JuiceFS 上，目录权限正常（`n84416302:n84416302 drwxrwxr-x`）。
- 出错位置在 `os.open` 而非 `flock`，是 JuiceFS 在 4 个 rank 同时 `O_CREAT|O_RDWR|O_TRUNC` 同一个 lock 文件时的偶发 EPERM。
- HF 的 parquet builder cache（`dataset_info.json` + `parquet-train.arrow`）当时已经 build 完整，所以这次启动只是要拿 lock 而已。
- 同一波启动里 stack（不同 cache 哈希）没踩到，说明不是稳定可复现的权限问题。

### 7.3 第三次提交（sort，2026-05-02 00:14，job 64715）

不改训练脚本/代码，只在投递前预先 `touch` 那个 lock 文件，让 `os.open(O_CREAT)` 走"打开已有文件"分支，绕开 JuiceFS 的并发创建竞态：

```bash
touch /home/n84416302/.cache/huggingface/datasets/parquet/default-1283ec8211bd548d/0.0.0/ec3958b52035e87929642b609b5a781870321434b151a14fbe8fd5a26ff494bb_builder.lock
sbatch submit_train_pi05_prgvla_sort_the_table_leftcam_v1.sh   # → job 64715
```

- 64715 实际于 `2026-05-02 00:26:43` 启动在 `lrc-alpha-sg-gpu06`。
- 00:32 进入 training step，至 00:36 已到 `step:600 / 80K`，`loss:0.124`，无任何 Error/Traceback。lock 绕过成功。

### 7.4 双任务同时运行确认（2026-05-02 00:46 快照）

| Job | 任务 | 节点 | 状态 | 进度 |
|---|---|---|---|---|
| 64703 | `stack_the_cups` restart | gpu02 | RUNNING ~1h20m | `step≈7K / 80K`, loss 0.016 |
| 64715 | `sort_the_table` restart | gpu06 | RUNNING ~20m | `step:600 / 80K`, loss 0.124 |

后续要做的事：

- 等到第一次 `save_freq=20000` checkpoint 落盘后，确认磁盘配额和 checkpoint 完整性。
- 80K step 训完后下载/上传 checkpoint。

复现这次 lock 绕过的方法（如果以后再踩到同类 EPERM）：

```bash
# 先按报错的实际路径 touch
touch <path_in_PermissionError_message>
# 再 sbatch 重投
```

根因层面更稳的做法（暂未实施）：把 `HF_DATASETS_CACHE` 重定向到节点本地 SSD（如 `/tmp/$SLURM_JOB_ID/hf_cache`），完全避开 JuiceFS 上的并发文件创建竞态。

## 8. 离线 fit-check（看每个 dim 拟合曲线）

### 8.1 是什么

`train4arx/fit_check_pi05.py` —— lerobot-native 的离线 fit-check 脚本：

- 载训完的 pi05 checkpoint
- 在原始训练数据集上跑 deployment-style replan inference（每 `execute_horizon=10` 步重 plan 一次）
- 对每个 episode 收集 GT vs predicted action，画 8 行 subplot 的拟合曲线（`raw` 关节单位 + `norm` 训练空间各一张）
- 落 `action_fit_summary.json`（schema 与同目录外发的 `external_predictor_eval.py` 兼容）

设计上有意做的事：

- 完全 lerobot-native：用 `PI05Policy.from_pretrained` + `make_pi05_pre_post_processors(ds.meta.stats)`，state / image keys / chunk_size 全部从 ckpt config 自动读，**不**手 hardcode action_dim。
- norm 空间走 **q01/q99**（和训练 `normalization_mapping={"ACTION":"QUANTILES",...}` 完全一致），所以 norm 曲线就是模型实际看到的 [-1, 1] 空间——**训推一致**。
- 默认对 dataset 里**所有 episode** 跑（sort 36 个 / stack 30 个）。要快速 smoke 加 `--num-episodes 2`。
- `MAX_EPISODE_STEPS=240` 截断长 episode，需要看完整长度可加 `--max-episode-steps 9999`。

### 8.2 跑

第一个 ckpt 在 `step=20000` 落盘后（路径 `results/pi05_prgvla_<task>_leftcam_v1_4gpu_b16_c60_s80k_restart_20260501/checkpoints/020000/pretrained_model/`）：

```bash
conda activate lerobot-new
cd /home/n84416302/lerobot

# stack
python train4arx/fit_check_pi05.py \
  --checkpoint  results/pi05_prgvla_stack_the_cups_leftcam_v1_4gpu_b16_c60_s80k_restart_20260501/checkpoints/020000/pretrained_model \
  --dataset-root /home/n84416302/dataset/prgvla_stack_the_cups_leftcam_v1 \
  --task-prompt  "stack the cups" \
  --task-slug    stack_cups \
  --output-dir   results/fit_check/stack_cups_step20k

# sort
python train4arx/fit_check_pi05.py \
  --checkpoint  results/pi05_prgvla_sort_the_table_leftcam_v1_4gpu_b16_c60_s80k_restart_20260501/checkpoints/020000/pretrained_model \
  --dataset-root /home/n84416302/dataset/prgvla_sort_the_table_leftcam_v1 \
  --task-prompt  "sort the table" \
  --task-slug    sort_table \
  --output-dir   results/fit_check/sort_table_step20k
```

预计 stack 全 30 ep ~25-30 min，sort 全 36 ep ~30-40 min（单 H200，bf16）。

后续 step 40000 / 60000 / 80000 ckpt 落盘后再跑一次同样的命令、改 `--checkpoint` 路径、改 `--output-dir` 后缀（如 `_step40k`）就能拿到不同步数下的拟合演化。

### 8.3 输出

每个 task 目录下：

```
results/fit_check/<task>_step<N>k/
  action_fit_summary.json                                   # 顶层指标（aggregate.raw + aggregate.normalized）
  episode_<eid>_<task>_action_dims_raw.png                  # 8-dim 拟合曲线（关节单位）
  episode_<eid>_<task>_action_dims_normalized.png           # 8-dim 拟合曲线（[-1, 1] 训练空间）
  ...
```

### 8.4 关键 CLI 参数

- `--episode-indices 0 5 10 ...`：只跑指定 episode
- `--num-episodes N`：取前 N 个
- `--execute-horizon`：默认 10，对应 deployment 重 plan 频率
- `--max-episode-steps`：默认 240，单 episode 评估上限
- `--device`：默认 cuda
- `--dtype`：默认 bfloat16

### 8.5 和外发 harness 的关系

- 内部 fit-check（`fit_check_pi05.py`）：直接走 lerobot policy 接口，state/image 自动正确传，跑得最简单——你自己看自己模型用这个。
- 外发 harness（`/home/n84416302/lerobot/external_predictor_eval.py`，starVLA 那边发来的 `EXTERNAL_EVAL_GUIDE.md`）：model-agnostic 的 my_predict 插件式，给 Pi0 / 其他 codebase 跨工程对比用，有几个已知问题（缺 state 入参、video key 写死 `primary_image` 而非 `camera_left` 等），暂时不直接用，留作"如果以后要跨 codebase 比较"的接入点。

两份脚本的 `action_fit_summary.json` schema 兼容，意味着将来如果需要把内部 fit-check 数字和 starVLA / Pi0 的数字横向对比，可以直接读两个 JSON 比 `aggregate.raw.mae_mean`。

