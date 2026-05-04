# prgvla pi05 部署指南（给真机端的同学）

这份文档说明如何在 Franka 真机上部署 prgvla（`sort_the_table` / `stack_the_cups`，60 步 chunk，7 关节 + 1 gripper）训练好的 **pi05** 权重。

## 0. 两条路径，按需选

**A. 最简（推荐先试）：离线 Python 直接拿 action**
权重 + config 在手，调一次 `predict_action_chunk` → 拿到 `(60, 8)` 的 action 数组（已经在训练机的关节单位里）。看第 5 节，<40 行 Python 就够。

**B. 生产部署：lerobot async policy server + 真机 client**
GPU 主机起 `lerobot.async_inference.policy_server`（gRPC 协议），真机端连接发送 (image + state + task) → 收 action chunk → 执行前 N 步 → 再查询。看第 6-8 节。

> 大部分情况你们只需要路径 A —— 拿到 action 之后怎么映射到 Franka 自己的控制器，是你们那边的事。路径 B 解决的是 GPU 和真机不在同一台机器、需要跨机通信的场景。

## 1. ⚠ 关键前提：训练数据本来就是 Franka（fr3）采的

prgvla 训练数据**本身就是 Franka Research 3 机械臂采集的**（看 [train4arx/convert_prgvla_to_lerobot.py:26](https://github.com/Fei-Ni/lerobot/blob/main/train4arx/convert_prgvla_to_lerobot.py#L26) `ROBOT_TYPE = "fr3"`）。所以训练机和你们目标部署机**本体、自由度、关节顺序、单位**全部一致 —— **没有跨机器人 remap 的问题**。

但是，**不同的 Franka 个体**之间还是会有：

- 出厂校准（zero offset）的微小差异
- gripper 硬件版本不同（Franka Hand vs 第三方夹爪 vs 自定义末端），单位接口（position / width / 比例）可能不一样
- 控制器（FCI / libfranka 版本）报告 `q` 关节角的精度可能略有差异

实际意义：

- **服务器端代码原样用**：`lerobot.async_inference.policy_server` 是 robot-agnostic 的，只把 image + state + 文本 prompt → 8 维 action chunk。
- **客户端可选两条路**：
  1. 在 lerobot 里加一个 `franka_follower/` 适配器（看 [src/lerobot/robots/openarm_follower/openarm_follower.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/openarm_follower/openarm_follower.py) 当模板，~150-300 行），然后用 `lerobot.async_inference.robot_client` 一条命令拉起来
  2. 跳过 `robot_client`，自己写一个走 lerobot gRPC 协议的薄客户端（参考 `examples/tutorial/async-inf/robot_client.py`）
- **同型号 Franka 之间**：默认假设关节角和 gripper 单位 = 训练数据原文，**不需要 remap 也能跑**。如果发现行为完全错乱，先按第 9 节"故障排查清单"逐条 check，不要先去做关节单位映射；如果一切都对了但仍然偏差大，再考虑校准 / gripper 硬件差异。**第一次上电仍然建议先在悬空（gravity-comp）模式下打印 action chunk 看一眼实际范围、确认安全后再上电**。

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

**推荐用 80K 全步**（训练完整跑完）。中间 ckpt 也都已上传，对比/排错时可以用。

| 任务 | ckpt | HF repo URL | aggregate raw_MAE | aggregate norm_MAE |
|---|---:|---|---:|---:|
| stack_cups | **80K（推荐）** | https://huggingface.co/spikefly/pi05-prgvla-stack-cups-step80k | **0.00413** | 0.01103 |
| stack_cups | 60K | https://huggingface.co/spikefly/pi05-prgvla-stack-cups-step60k | 0.00437 | 0.01163 |
| sort_table | **80K（推荐）** | https://huggingface.co/spikefly/pi05-prgvla-sort-table-step80k | **0.00429** | 0.00994 |
| sort_table | 40K | https://huggingface.co/spikefly/pi05-prgvla-sort-table-step40k | 0.00493 | 0.01140 |

> 上面 raw MAE 是在**训练数据上**拿训完的 ckpt 重新推理 + GT 对照得到的，本质是看模型有没有过拟合训练集（数字小 = 拟合到了）。**真机泛化效果还得在 Franka 上跑才能知道**。
>
> 这些是 deploy-style autoregressive replan、`execute_horizon=10`、`do_sample` 默认，全 episode 跑出来的 aggregate（具体数字也写在每个 ckpt 旁的 `action_fit_summary.json` 里）。**部署侧自己跑 fit-check 应该重现这些数字**，差太多 → 环境 / 输入 wiring 出问题，不是 ckpt 的事。

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
hf download spikefly/pi05-prgvla-stack-cups-step80k \
  --local-dir <CKPT_DIR>/pi05_prgvla_stack_the_cups
```

或者直接让 lerobot 在线加载，传 `pretrained_name_or_path=spikefly/pi05-prgvla-stack-cups-step80k` 即可。

### HuggingFace（LeRobot v3 数据集——和训练时完全一致的转换版）

如果你们想本地跑 fit-check 重现我们这边的数字、或者排查"raw 数据 vs lerobot 转换数据"差异：

| 任务 | HF dataset URL | size |
|---|---|---:|
| sort_table | https://huggingface.co/datasets/spikefly/prgvla-sort-the-table-leftcam-v1 | 56 MB |
| stack_cups | https://huggingface.co/datasets/spikefly/prgvla-stack-the-cups-leftcam-v1 | 60 MB |

每个 dataset repo 根直接是 LeRobot v3 layout（`data/ images/ meta/ videos/`），下载后直接 `LeRobotDataset(repo_id="spikefly/prgvla-...", root="<DOWNLOAD_DIR>")` 能读。每个 dataset repo 还附带了一份 `convert_prgvla_to_lerobot.py`，是这份数据集是怎么从 `zhuoKCL/prgvla` 原始 HDF5 转过来的脚本。

**关键事实：转换脚本是纯格式重排，state / action 数值就是 `zhuoKCL/prgvla` 原始 HDF5 里的 `observation/robot_state/joint_positions` + `gripper_position` / `action/joint_position` + `gripper_position`，没有任何单位变换、scale、归一化**。视频是 `cv2.cvtColor(BGR2RGB)`，FPS 和分辨率全部保持原样。所以"raw HDF5 vs 我们 LeRobot 数据集"在数值层面是 identity，不应该有任何差异。

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

### 8.1 离线 fit-check（什么真机硬件都不用）—— **最重要的一步，先做这个**

直接拿训练数据集当输入，跑训完的 ckpt，复现我们这边的 raw MAE（应该 ≈ 0.004 量级，全 episode aggregate 见第 3 节那张表）：

```bash
# 从 HF 拉转换好的 LeRobot v3 数据集（56-60 MB）
hf download spikefly/prgvla-stack-the-cups-leftcam-v1 --repo-type dataset --local-dir /tmp/prgvla_stack
hf download spikefly/prgvla-sort-the-table-leftcam-v1 --repo-type dataset --local-dir /tmp/prgvla_sort

# 跑 fit-check（脚本在 https://github.com/Fei-Ni/lerobot/blob/main/train4arx/fit_check_pi05.py ，~280 行）
python train4arx/fit_check_pi05.py \
  --checkpoint  spikefly/pi05-prgvla-stack-cups-step80k \
  --dataset-root /tmp/prgvla_stack \
  --task-prompt "stack the cups" --task-slug stack_cups \
  --output-dir ./smoke_stack --num-episodes 2

python train4arx/fit_check_pi05.py \
  --checkpoint  spikefly/pi05-prgvla-sort-table-step80k \
  --dataset-root /tmp/prgvla_sort \
  --task-prompt "sort the table" --task-slug sort_table \
  --output-dir ./smoke_sort --num-episodes 2
```

预期：

- 跑出 raw_MAE ≈ **0.004** 左右（具体见第 3 节表）→ 你们环境完全 OK，剩下纯属真机端 wiring 的问题
- 跑出 raw_MAE **>> 0.01** → 你们这边 lerobot 版本 / processor 加载 / dataset stats 出问题，**先解决这个再上真机**
- 完全跑不通 → 看报错；常见见第 10 节

### 8.2 Franka 悬空（gravity-comp）模式 dryrun

先**关掉电机闭环**，让 Franka 处于 gravity comp / passive 模式，相机正常，跑你的 client → server 链路：

1. 看 server 返回的 `raw_action[k]` 数值范围是不是合理（关节弧度应在 ±π 内、gripper 数值应在你 Franka 那台 gripper 的合理范围）
2. 把当前 Franka 真机读出来的 `q` 关节角和 server 返回的 `raw_action[0]` 做 diff，应该是同量级、同符号、连续可微的曲线
3. 这一步**不要给电机发命令**，只 print + 画图调试

### 8.3 低速 + scaled 联动测试

通过前两步后，**先把每步 action 缩 50%**，再上电跑几个 episode，确认动作方向正确再恢复全速。

## 9. 关于"和训练时归一化口径一致"

pi05 训练用 `normalization_mapping={"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}`，即 state / action 都用 q01/q99 quantile 归一化到 [-1, 1]。

**这部分对客户端是透明的**：

- ckpt 里的 `policy_preprocessor_step_2_normalizer_processor.safetensors` 已经包含训练时的 q01/q99 stats
- `policy_postprocessor_step_0_unnormalizer_processor.safetensors` 包含反归一化 stats
- 客户端**送 raw 关节单位的 state 进去、收 raw 关节单位的 action 出来**就行，不用自己 normalize

唯一要保证的：`observation.state` 的物理单位必须和训练时**完全一致**（即 `zhuoKCL/prgvla` HDF5 里的 `joint_positions` / `gripper_position` 的单位）。训练数据是 Franka fr3 采的，所以同型号 Franka 默认应该单位一致；如果你们的 Franka 经过自定义校准、或者用的是非原装 gripper，可能要做一层 remap。

## 10. 故障排查清单（建议从上往下逐条 check）

下面每一条都是一个独立的可验证假设。**强烈建议先按这个清单查一遍**，再讨论"是不是 ckpt 有问题"。我们这边 fit-check 已经在训练数据上得到 raw_MAE ≈ 0.004，证明 ckpt 本身完全 OK。

### 10.1 ckpt 和环境（最先 check）

- [ ] **复现 fit-check 数字**（最关键）：按 §8.1 跑一遍，目标是复现第 3 节表里那个 raw_MAE ≈ 0.004。如果跑出来差不多 → 你们环境完全 OK，**所有问题都在真机端 wiring**，下面的 §10.2 / §10.3 / §10.4 才是排查方向。如果差很大 → 是你们 lerobot 版本 / 安装问题，先解决。
- [ ] lerobot 版本：`pip show lerobot` 至少要支持 pi05（main 分支即可），太老的 release 没有 pi05。
- [ ] 全部依赖：`pip install -e ".[pi,async]"` 都装了吗。
- [ ] ckpt 完整性：`ls <ckpt_dir>` 必须有 7 个文件：`config.json`、`model.safetensors`、`policy_preprocessor.json`、`policy_preprocessor_step_2_normalizer_processor.safetensors`、`policy_postprocessor.json`、`policy_postprocessor_step_0_unnormalizer_processor.safetensors`、`train_config.json`。任一缺失都会让 normalize / unnormalize 静默错位。

### 10.2 输入 wiring（真机推理出错最常见的原因）

按"出错可能性从高到低"排：

- [ ] **图像通道**：cv2 读出来默认是 BGR，pi05 训练用的是 RGB。client 端必须 `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`，或者用 PIL（PIL 直接是 RGB）。颜色通道反了，pi05 视觉表征整段错位 —— 在训练集复现可能仍然部分对得上，但泛化到真实环境会乱。
- [ ] **camera key 名字**：必须是 `observation.images.camera_left`（不能是 `image` / `rgb` / `primary_image` / `top` 等）。pi05 内部 `_preprocess_images` 严格按 `config.input_features` 的 key 取图，缺了会报 "All image features are missing"。
- [ ] **task 文本**：必须精确匹配训练原文：
  - `sort_table` ckpt → `"sort the table"`
  - `stack_cups` ckpt → `"stack the cups"`
  - 大小写、空格、单复数、有无标点全部敏感。
- [ ] **state 顺序和单位**：必须 `[joint_0, joint_1, ..., joint_6, gripper]`，全部 raw 关节角弧度 + gripper position 原值。**不要自己 normalize**（pi05 内部 preprocessor 用 q01/q99 自动归一化）。
- [ ] **state shape**：必须 `(8,)` float32，不是 `(7,)` 或 `(8, 1)` 或 `(1, 8)`，client 端送 batch 时统一加 batch 维度成 `(1, 8)`。
- [ ] **不要 pre-normalize**：千万别在 client 端把 state 缩到 [-1, 1] 再送进去，pi05 内部已经做了。
- [ ] **图像分辨率**：原始 480×640（HxW）即可，pi05 内部会 resize-pad 到 224×224。如果 client 强制塞别的尺寸，pi05 也能 resize，但**色彩范围**必须保持 uint8 [0, 255] 或 float32 [0, 1]，不要送 [-1, 1]。
- [ ] **图像分辨率**也不要预先缩小成 224×224 —— pi05 的 resize-pad 算法和直接 cv2.resize 不同，会有 letterbox padding；让 pi05 自己 resize 能保证 train/inference 一致。

### 10.3 控制循环

- [ ] **频率**：训练数据 10 FPS，control loop 也用 **10 Hz**。如果跑到 30 / 60 Hz 会让模型推断的 action 在时间上"超前"实际机器人状态，行为会震荡或漂走。
- [ ] **execute_horizon**：每次拿 60 步 chunk 后，**执行前 10 步**就重新查询（不要一口气执行 60 步）。详见 §7.3 伪代码。
- [ ] **不要跨 episode 复用 chunk**：episode 切换时 client 端要清空 chunk buffer，不然会从旧 chunk 残留动作开始执行新 episode。

### 10.4 真机硬件（如果以上都对、还是行为奇怪才考虑这条）

- [ ] **gripper 硬件**：训练数据里的 gripper 是哪种夹爪？position 单位是 m 还是 normalized？你们 Franka 上是不是同一型号？如果不同，需要做一层 gripper remap。
- [ ] **Franka 校准**：同型号 Franka 之间存在出厂 zero-offset 差异，通常很小（< 1°），不影响整体行为；如果你们这台 Franka 经过自定义重校准，先确认 `q` 输出是不是 vanilla 弧度。
- [ ] **末端执行器/工具偏置**：如果你们装了非原装末端，重心和动力学会有差异，但短期 action 预测影响不大。

### 10.5 常见报错对照

| 报错 | 大概率原因 |
|---|---|
| `RuntimeError: All image features are missing from the batch` | camera key 不对，必须 `observation.images.camera_left` |
| `ValueError: State is required for PI05` | `observation.state` 没在 batch 里 |
| Server 加载 ckpt 时 `GatedRepoError` | lerobot 版本太老，pi05 的 `model.safetensors` 已经 baked paligemma，新版本不需要 paligemma gated 拉取 |
| Server 启动 OK 但 client connect 卡住 | `host=0.0.0.0` 没绑全网卡？防火墙开了 8080 吗？ |
| Client 第一次 query 30s 不响应 | 正常——pi05 4B 第一次加载要 30-90s，给 timeout 加大 |
| 真机执行时 action 数值看着像 [-1, 1] | Postprocessor 没生效，ckpt 目录是不是缺 `policy_postprocessor*` 文件 |
| 真机执行时 action 全是 NaN | 图像 dtype 错（送了 [-1, 1] float），或 state shape 错 |
| 关节驱到危险位置 | 多半是 state 顺序错位 / gripper 单位不一致；先按 §8.2 dryrun 确认范围 |
| Pi05 在仿真上能跑、真机抖 | 控制频率不对（不是 10 Hz）或 BGR/RGB 错 |

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

- [x] 4 个 ckpt 全部上传 HF（80K + 中间 ckpt：60K stack、40K sort）
- [x] 两个转换好的 LeRobot v3 数据集上传 HF（sort-the-table、stack-the-cups）
- [x] 数据集 repo 里附带了 `convert_prgvla_to_lerobot.py`
- [x] 仓库镜像到 https://github.com/Fei-Ni/lerobot
- [x] 每个 ckpt 的 fit-check aggregate 数字写进本文档 §3 表格 + §10.1 排查清单
- [ ] 提醒部署侧：第一步是 §8.1 的离线 fit-check 复现 raw_MAE ≈ 0.004；如果跑不出来，问题在他们环境而不是我们 ckpt
- [ ] 提醒部署侧：训练数据本身就是 Franka，**不需要做 ARX → Franka remap**
- [ ] 提醒部署侧：训练 fps = 10，control loop 也用 10 Hz
