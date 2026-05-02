# prgvla pi05 部署指南（给真机端的同学）

这份文档说明如何在 Franka 真机上部署 prgvla（`sort_the_table` / `stack_the_cups`，60 步 chunk，7 关节 + 1 gripper）训练好的 **pi05** 权重。

## 0. 两条路径，按需选

**A. 最简（推荐先试）：离线 Python 直接拿 action**
权重 + config 在手，调一次 `predict_action_chunk` → 拿到 `(60, 8)` 的 action 数组（已经在训练机的关节单位里）。看第 5 节，<40 行 Python 就够。

**B. 生产部署：lerobot async policy server + 真机 client**
GPU 主机起 `lerobot.async_inference.policy_server`（gRPC 协议），真机端连接发送 (image + state + task) → 收 action chunk → 执行前 N 步 → 再查询。看第 6-8 节。

> 大部分情况你们只需要路径 A —— 拿到 action 之后怎么映射到 Franka 自己的控制器，是你们那边的事。路径 B 解决的是 GPU 和真机不在同一台机器、需要跨机通信的场景。

## 1. ⚠ 关键前提：目标机 = Franka，**不是**训练数据采集那台 ARX

prgvla 训练数据是用 **ARX** 机械臂采集的（7 joints + gripper position 的 ARX-style layout），而你们的目标部署机是 **Franka 七轴机械臂**。两台机器都是 7 关节 + 1 gripper，**维度对得上**，但单位 / 零位 / 关节限位 / gripper 接口几乎肯定**不一样**。

实际意义：

- **服务器端代码可以原样用**：`lerobot.async_inference.policy_server` 是 robot-agnostic 的——它只负责把 image + state + 文本 prompt → 8 维 action chunk。
- **客户端代码要自己写或改造**：lerobot 自带的 `lerobot.async_inference.robot_client` 通过 `--robot.type=<...>` 加载 lerobot 内置的机器人 driver（目前内置有 `koch_follower / so_follower / openarm_follower / lekiwi / unitree_g1 / reachy2 / hope_jr / omx_follower / earthrover_mini_plus / bi_*_follower`，**没有 Franka**）。所以你们要么：
  1. 在 lerobot 里加一个 `franka_follower/` 适配器（看 [src/lerobot/robots/openarm_follower/openarm_follower.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/openarm_follower/openarm_follower.py) 当模板，~150-300 行）
  2. 或者跳过 `robot_client`，自己写一个走 lerobot gRPC 协议的薄客户端（参考 `examples/tutorial/async-inf/robot_client.py`）
- **关节单位映射要自己处理**：server 返回的 action 是训练时那台 ARX 的关节范围。如果 Franka 的关节零位 / 限位 / gripper 单位不同，**要在 client 端做一层 remap**。**强烈建议先在悬空（无力矩 / gravity-comp）模式下打印 action chunk 看一眼实际范围、确认安全后再上电**。

## 2. 模型契约（两个 prgvla checkpoint 都一样）

```
policy_type:           pi05
action_dim:            8                              # joint_0..joint_6 + gripper
chunk_size:            60                             # 模型一次返回 60 步
n_action_steps:        60
include_state:         true                           # ⚠ pi05 必须送 state（与某些 VLA 不同）
camera 数量:           1
camera key:            observation.images.camera_left # 训练用的是静态左视图（310222076420）
state key:             observation.state
image size 输入:       480×640 RGB 即可（pi05 内部会 resize-pad 到 224×224 + normalize 到 [-1, 1]）
state shape:           (8,) float32
state 顺序:            [joint_0, joint_1, ..., joint_6, gripper]   # 严格按这个顺序
normalization:         q01/q99 (QUANTILES) —— 由 ckpt 内置 processor 自动处理，client 不要自己 normalize
num_inference_timesteps: 默认（pi05 flow matching，4 步）
```

任务 prompt（必须照训练原文，不要改写）：

| checkpoint | task_prompt |
|---|---|
| `sort_table` | `"sort the table"` |
| `stack_cups` | `"stack the cups"` |

**控制频率**：训练数据是 **10 FPS**，真机控制 loop 不要直接拉到 30/60 Hz——动力学不匹配会让模型推断的 action 偏离。**10 FPS 起跑**，或在客户端做插值再下发。

## 3. 代码 + 权重在哪（已上传）

### GitHub

这套部署用的是 **HuggingFace 官方 lerobot**（pi05 已在 main 分支），不需要 fork：

```bash
# 部署机上（GPU host）：
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[pi,async]"
```

> 训练侧的脚本（`train4arx/`）部署不需要。`pi` extra 装 paligemma 依赖，`async` extra 装 gRPC 通信依赖。

### HuggingFace（pi05 ckpt）

| Checkpoint | HF repo URL | 训练步 | raw MAE（训练集自评，仅供 sanity）|
|---|---|---:|---:|
| `pi05_prgvla_stack_the_cups` | `https://huggingface.co/spikefly/pi05-prgvla-stack-cups-step60k` | 60000 | 0.00437 |
| `pi05_prgvla_sort_the_table` | `https://huggingface.co/spikefly/pi05-prgvla-sort-table-step40k` | 40000 | 0.00493 |

> 上面 raw MAE 是在**训练数据上**拿训完的 ckpt 重新推理 + GT 对照得到的，本质是看模型有没有过拟合训练集（数字小 = 拟合到了）。**真机泛化效果还得在 Franka 上跑才能知道**。

每个 HF repo 根目录直接放 lerobot 的 "pretrained_model" 内容：

```
<repo>/
  config.json                                                  # PI05Config
  model.safetensors                                            # 9.35 GB 权重
  policy_preprocessor.json
  policy_preprocessor_step_2_normalizer_processor.safetensors  # 输入归一化（state/image）
  policy_postprocessor.json
  policy_postprocessor_step_0_unnormalizer_processor.safetensors  # 输出反归一化（action）
  train_config.json                                            # 训练配置快照（可读，但部署不会读）
```

下载到本地（任选一份）：

```bash
hf download spikefly/pi05-prgvla-stack-cups-step60k \
  --local-dir <CKPT_DIR>/pi05_prgvla_stack_the_cups
```

或者直接让 lerobot 在线加载，传 `pretrained_name_or_path=spikefly/pi05-prgvla-stack-cups-step60k` 即可。

## 4. GPU 主机环境

### 4.1 Conda

```bash
conda create -n lerobot-pi python=3.12 -y
conda activate lerobot-pi
cd <REPO_ROOT>/lerobot
pip install -e ".[pi,async]"
```

### 4.2 外部依赖

pi05 模型继承自 `google/paligemma-3b-pt-224`。HF release 上传时已经把 paligemma 权重 baked 进 `model.safetensors` 了，**部署阶段不需要再下载 paligemma 或单独 HF token**。

> 训练阶段才需要 `HF_TOKEN` 拉 `paligemma-3b-pt-224`。部署只 load 你们这份 ckpt，跳过 paligemma 下载。

### 4.3 GPU 显存

- 4B 参数模型，bf16 加载约 **8 GB** GPU 显存
- 推理（bs=1，chunk=60）约 12 GB peak
- 单张 H100 / H200 / A100 都跑得动；**3090/4090（24GB）也够**

## 5. 离线推理最小代码（路径 A）

### 5.1 命令行 smoke

仓库里已经有：[train4arx/fit_check_pi05.py](https://github.com/.../train4arx/fit_check_pi05.py)（训练机生成的 fit-check 脚本，部署侧也可用）。它做的事情就是路径 A：load ckpt → 喂 dataset 一段 episode → 跑 deployment-style replan → 画曲线 + JSON。

可以直接拿来对你们的 prgvla 数据集（如果你们也有同份转过的 LeRobot 格式）做 sanity check，确认 Franka 端整条链路通不通：

```bash
python train4arx/fit_check_pi05.py \
  --checkpoint  <CKPT_DIR>/pi05_prgvla_stack_the_cups \
  --dataset-root <YOUR>/prgvla_stack_the_cups_leftcam_v1 \
  --task-prompt  "stack the cups" \
  --task-slug    stack_cups \
  --output-dir   ./fit_check_smoke \
  --num-episodes 2
```

### 5.2 集成进自己代码核心代码

```python
import torch, numpy as np
from PIL import Image
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05 import PI05Policy, make_pi05_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset

CKPT = "spikefly/pi05-prgvla-stack-cups-step60k"   # HF repo id 或本地路径都行
DEVICE = "cuda"

# --- load 一次 ---
config = PreTrainedConfig.from_pretrained(CKPT)
config.device = DEVICE
config.dtype = "bfloat16"
policy = PI05Policy.from_pretrained(CKPT, config=config).eval()

# 正常情况下做 stats 用一份你们这边 LeRobot 转换的 prgvla 数据集；
# 如果只在线推理不读 dataset，可以从 ckpt 自己的 processor 文件直接做处理（lerobot 内部已支持）。
ds = LeRobotDataset(repo_id="...", root="<YOUR>/prgvla_stack_the_cups_leftcam_v1")
preprocessor, postprocessor = make_pi05_pre_post_processors(config=config, dataset_stats=ds.meta.stats)

# --- 每帧 / 每个 plan step 调一次 ---
@torch.no_grad()
def get_action_chunk(image_pil, state_8d, task_text):
    """
    image_pil  PIL.Image RGB（任意分辨率，pi05 内部会 resize-pad 到 224×224）
    state_8d   shape (8,) np.ndarray float32 —— [joint_0..joint_6, gripper] 原始关节单位
    task_text  e.g. "stack the cups"
    返回:
       np.ndarray (60, 8) float32, raw 关节单位（与训练数据同 ARX 单位）
    """
    img = torch.from_numpy(np.asarray(image_pil.convert("RGB"))).permute(2, 0, 1).float() / 255.0
    batch = {
        "task": [task_text],
        "observation.state": torch.from_numpy(state_8d).float().unsqueeze(0).to(DEVICE),
        "observation.images.camera_left": img.unsqueeze(0).to(DEVICE),
    }
    batch = preprocessor(batch)
    actions_norm = policy.predict_action_chunk(batch)         # [1, 60, 8] in norm space
    actions_raw = postprocessor(actions_norm)                 # raw 关节单位
    return actions_raw.detach().cpu().numpy()[0]              # (60, 8)
```

每帧调一次 `get_action_chunk(img, state, task)` 拿 60 步 → 执行前 N 步（建议 10 步）后再调一次，循环到任务结束。

### 5.3 想看完整 reference 推理 loop？

[train4arx/fit_check_pi05.py](/home/n84416302/lerobot/train4arx/fit_check_pi05.py) 就是 deployment-style autoregressive replan 推理 loop（用训练数据当输入、与 GT 对比、画曲线）。底层 `predict_action_chunk` 调用方式跟上面一样，多了 episode 循环和画图——看那份代码就能完整理解推理时的所有细节。

---

## 6. 启动 lerobot async policy server（路径 B）

```bash
cd <REPO_ROOT>/lerobot
conda activate lerobot-pi
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}

CUDA_VISIBLE_DEVICES=0 python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8080 \
  --fps=10 \
  --inference_latency=0.1 \
  --obs_queue_timeout=1
```

启动后 server 自己**没有**模型——它会等第一次 client handshake，client 发 `pretrained_name_or_path=spikefly/pi05-prgvla-stack-cups-step60k` 过来后再加载。

> ⚠ pi05 4B 模型加载大约 **30-90 秒**（第一次会从 HF 拉 9.35 GB 权重；二次走本地缓存就快）。client 端 connection timeout 要给够。

## 7. 真机端 client（你们要写的部分）

### 7.1 现成的 `robot_client` 路径（如果你们能给 lerobot 加 Franka driver）

```bash
python -m lerobot.async_inference.robot_client \
    --server_address=<GPU_HOST>:8080 \
    --robot.type=franka_follower \                         # ← 你们要新加的 driver
    --robot.id=franka_lab1 \                               # 用来 load 校准文件
    --robot.cameras="{ camera_left: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 10} }" \
    --task="stack the cups" \
    --policy_type=pi05 \
    --pretrained_name_or_path=spikefly/pi05-prgvla-stack-cups-step60k \
    --actions_per_chunk=60 \
    --chunk_size_threshold=0.5 \
    --policy_device=cuda \
    --client_device=cpu
```

需要先在 lerobot 里加 `src/lerobot/robots/franka_follower/`：

- `franka_follower.py` 实现 `Robot` 接口（`get_observation()` / `send_action()` / `connect()` / `disconnect()`）
- `config_franka_follower.py` dataclass，含 `port`、`cameras` 等字段
- `__init__.py` 注册到 lerobot 的 robot registry

`get_observation()` 返回的 dict 必须含：

```python
{
  "observation.state": np.ndarray (8,) float32,           # [joint_0..joint_6, gripper]
  "observation.images.camera_left": np.ndarray (H, W, 3) uint8 RGB,
}
```

`send_action(action: np.ndarray (8,))`：把 8 维 action **remap 到 Franka 单位**后下发到控制器。这一步是 ARX → Franka 关节适配的关键。

### 7.2 自定义客户端（如果不想动 lerobot 仓库）

直接写一个 gRPC 客户端，按 lerobot 的 [transport/services.proto](https://github.com/huggingface/lerobot/blob/main/src/lerobot/transport/services.proto) 协议通信。参考 `examples/tutorial/async-inf/robot_client.py` 看流程：

1. 连接 gRPC server
2. 第一次发 handshake（含 `pretrained_name_or_path`、`policy_type=pi05`、camera spec）
3. 控制 loop：每步 capture image + state → 序列化（pickle / msgpack）→ 发到 server → 收 `action_chunk` (60, 8)
4. 执行前 N 步 → 执行完前 K 步时（chunk_size_threshold * actions_per_chunk）再发下一帧观测异步触发下一次推理

### 7.3 控制循环（伪代码）

```python
ws = connect_to_lerobot_server(host, port)
ws.handshake(policy_type="pi05",
             pretrained_name_or_path="spikefly/pi05-prgvla-stack-cups-step60k",
             ...)

EXECUTE_HORIZON = 10        # 执行前 10 步再 replan
CONTROL_DT = 0.1            # 训练 fps=10

action_chunk = None
chunk_offset = 0

while not done:
    # 当 chunk 用完一半就异步触发下一次推理
    if action_chunk is None or chunk_offset >= EXECUTE_HORIZON // 2:
        img = capture_camera_left_rgb()             # ← 你们的相机捕获 (H, W, 3) uint8
        state = read_franka_joints_8d()             # ← [joint_0..joint_6, gripper] in ARX-训练单位
                                                     #   ⚠ 如果 Franka 实际单位不同，要 inverse_remap 到训练单位
        ws.send_observation(image=img, state=state, task=TASK_PROMPT)
        # 异步收，如果还没回来就用现有 chunk
        new_chunk = ws.try_recv_action_chunk()
        if new_chunk is not None:
            action_chunk = new_chunk
            chunk_offset = 0

    raw_action = action_chunk[chunk_offset]         # (8,) raw ARX 关节单位
    franka_cmd = remap_arx_to_franka(raw_action)    # ← 关节角 / gripper 单位 remap
    send_to_franka_controller(franka_cmd)
    chunk_offset += 1
    time.sleep(CONTROL_DT)
```

## 8. 上电前的 smoke test（强烈建议）

### 8.1 离线 fit-check（什么真机硬件都不用）

直接拿训练数据集（一段 prgvla episode）当输入，跑路径 A 那个最小代码，看 action chunk 数值范围和形态：

```bash
python train4arx/fit_check_pi05.py \
  --checkpoint <CKPT_DIR>/pi05_prgvla_stack_the_cups \
  --dataset-root <YOUR>/prgvla_stack_the_cups_leftcam_v1 \
  --task-prompt "stack the cups" --task-slug stack_cups \
  --output-dir ./smoke --num-episodes 2
```

### 8.2 Franka 悬空（gravity-comp）模式 dryrun

先**关掉电机闭环**，让 Franka 处于 gravity comp / passive 模式，相机正常，跑你的 client → server 链路：

1. 看 server 返回的 `raw_action[k]` 数值范围是不是合理（关节弧度应在 ±π 内、gripper position 应在合理 mm/m 范围）
2. 看你的 `remap_arx_to_franka(raw_action)` 输出是不是落在 Franka 关节限位内
3. 这一步**不要给电机发命令**，只 print 调试

### 8.3 低速 + scaled 联动测试

通过前两步后，**先把每步 action 缩 50%**，再上电跑几个 episode，确认动作方向正确再恢复全速。

## 9. 关于"和训练时归一化口径一致"

pi05 训练用 `normalization_mapping={"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}`，即 state / action 都用 q01/q99 quantile 归一化到 [-1, 1]。

**这部分对客户端是透明的**：

- ckpt 里的 `policy_preprocessor_step_2_normalizer_processor.safetensors` 已经包含训练时的 q01/q99 stats
- `policy_postprocessor_step_0_unnormalizer_processor.safetensors` 包含反归一化 stats
- 客户端**送 raw 关节单位的 state 进去、收 raw 关节单位的 action 出来**就行，不用自己 normalize

唯一要保证的：`observation.state` 的物理单位必须和训练时**完全一致**（即 ARX 训练数据 HDF5 里的 `joint_positions` / `gripper_position` 的单位）。如果 Franka 默认报的是不同单位（比如度 vs 弧度），要在客户端**反过来 remap 到训练单位**再送给 server。

## 10. 常见故障

| 现象 | 大概率原因 |
|---|---|
| Server 加载 ckpt 时 GatedRepoError | 没装好 `[pi]` extra 或在线 paligemma 拉取卡住——pi05 的 `model.safetensors` 已经包含 paligemma 权重，请确认环境装的是新版本 lerobot |
| Server 启动 OK 但 client connect 卡住 | `host=0.0.0.0` 绑全网卡了吗？防火墙开了 8080 吗？ |
| Client 第一次 query 30s 不响应 | 正常——pi05 4B 第一次加载要 30-90s，给 timeout 加大 |
| 真机执行时 action 数值看着像 [-1, 1] | Postprocessor 没生效，检查 ckpt 目录是不是完整含 `policy_postprocessor*` 文件 |
| 真机执行时 action 全是 NaN / 不动 | 检查图像格式（必须 RGB uint8 或 [0,1] float，不要 BGR）、state shape 必须是 (8,) 不是 (7,) |
| 关节驱到危险位置 | ARX 单位没 remap 到 Franka（参考第 1 节），或者 gripper 单位不一致 |
| Pi05 报 "All image features are missing" | camera key 名字不对，必须用 `observation.images.camera_left` |
| Pi05 报 "State is required for PI05" | `observation.state` 没送到 batch 里 |

## 11. 文件索引

部署侧需要看的代码：

```
src/lerobot/async_inference/policy_server.py             # gRPC server，按原样用
src/lerobot/async_inference/robot_client.py              # 通用 client（要扩 Franka driver）
src/lerobot/async_inference/helpers.py                   # 协议辅助
src/lerobot/policies/pi05/modeling_pi05.py               # PI05Policy 实现
src/lerobot/policies/pi05/processor_pi05.py              # 输入预处理流水
src/lerobot/robots/openarm_follower/openarm_follower.py  # Franka driver 模板
examples/tutorial/async-inf/robot_client.py              # 自定义 client demo
train4arx/fit_check_pi05.py                              # 离线 fit-check 推理脚本（路径 A 参考）
```

每个 HF repo 拉下来直接是 lerobot pretrained_model layout，看：

```
config.json                                                   # PI05Config 完整训练配置
train_config.json                                             # 训练时的 LeRobotTrainConfig 快照
policy_preprocessor.json + ...normalizer_processor.safetensors  # 输入归一化
policy_postprocessor.json + ...unnormalizer_processor.safetensors # 输出反归一化
model.safetensors                                             # 9.35 GB 权重
```

---

## 12. 写给训练侧自己（push 之前的 checklist）

- [x] 确认两个 ckpt 的 `pretrained_model/` 目录完整（含 `model.safetensors` ~9.35 GB + 4 个 processor 文件）
- [x] 上传到 HF（spikefly/pi05-prgvla-stack-cups-step60k 和 spikefly/pi05-prgvla-sort-table-step40k）
- [ ] 推送 GitHub 时记下 commit / tag，写进本文档第 3 节（如果 fork 了 lerobot 而不是用 upstream）
- [x] 在本文档第 3 节填上实际的 HF repo URL
- [ ] 提醒部署侧读第 1 节（Franka ≠ ARX 训练单位）和第 2 节（契约）这两段
- [ ] 提醒部署侧训练 fps 是 10，真机 control loop 也用 10 Hz
