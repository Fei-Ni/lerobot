# PI05 Finetune Notes

这份说明是给当前 `lerobot` 仓库里的 `pi05` 微调用的，重点写清：

- 用哪个 conda 环境
- `HF_TOKEN` 和 `WANDB_API_KEY` 分别是干什么的
- 如果换自己的账号，哪些地方必须改
- 新数据里的 `STATE` 怎么看、怎么改成自己要训练的内容
- 训练脚本里最常改的参数有哪些

当前这套脚本主要是这 4 个：

- `train4arx/train_pi05_dual_fold_blanket_v4.sh`
- `train4arx/train_pi05_take_part_bricks_v3.sh`
- `submit_train_pi05_dual_fold_blanket_v4.sh`
- `submit_train_pi05_take_part_bricks_v3.sh`

## 1. Conda 环境

当前建议环境：

```bash
conda activate lerobot-new
cd /home/n84416302/lerobot
```

先确认 Python 版本：

```bash
python --version
```

当前仓库的 `pyproject.toml` 要求的是 `Python >= 3.12`，所以这里用 `lerobot-new`。

安装建议：

```bash
pip install -e ".[training,pi]"
```

说明：

- `.[pi]` 只装 `pi0/pi05` 相关依赖，不包含 `accelerate`
- `.[training]` 里才有 `accelerate` 和 `wandb`
- 所以只装 `.[pi]` 不够，跑训练会报 `accelerate: command not found`

装完最好检查一下：

```bash
which accelerate
which lerobot-train
python -c "import lerobot.policies.pi05; print('pi05 ok')"
```

## 2. HF_TOKEN 是干什么的

`HF_TOKEN` 是 Hugging Face 的访问令牌，主要用来做两件事：

1. 下载 gated model / gated tokenizer
2. 访问你自己有权限的私有或受限资源

在当前这套 `pi05` 训练里，最容易撞到的是：

```text
google/paligemma-3b-pt-224
```

`pi05` 的 tokenizer 会去拉这个资源。如果当前 token 没权限，就会报：

```text
403 Forbidden
GatedRepoError
```

推荐在 shell 里显式设置自己的 token，不要依赖机器上默认登录态：

```bash
export HF_TOKEN=你的_hf_token
hf auth login --token "$HF_TOKEN"
```

也可以先做一个最小检查：

```bash
python - <<'PY'
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
print("HF token ok")
PY
```

如果这里就 403，说明不是训练脚本问题，而是当前 `HF_TOKEN` 本身没拿到这个 gated repo 的访问权。

## 3. WANDB_API_KEY 是干什么的

`WANDB_API_KEY` 是 Weights & Biases 的登录凭据，只影响日志上传，不影响模型本身的 forward/backward。

它主要负责：

1. 创建 run
2. 上传 loss / lr / grad norm 等训练指标
3. 把 run 记到指定的 `entity/project`

如果 `WANDB_API_KEY` 错了，或者 key 不属于脚本里写的 `entity`，就会报：

```text
wandb.errors.errors.CommError
403 permission denied
```

推荐做法：

```bash
export WANDB_API_KEY=你的_wandb_key
wandb login --relogin "$WANDB_API_KEY"
```

如果你只是想先把训练跑起来，不想上传 wandb，可以直接关掉：

```bash
--wandb.enable=false
```

或者：

```bash
WANDB_MODE=offline bash train4arx/train_pi05_dual_fold_blanket_v4.sh
```

## 4. 如果换自己的账号，需要改什么

最少要确认这几项：

```bash
export HF_TOKEN=你的_hf_token
export WANDB_API_KEY=你的_wandb_key
```

然后检查训练脚本里这几项：

- `--wandb.entity=...`
- `--wandb.project=...`
- `--policy.repo_id=...`

如果 `entity` 不是你自己的 wandb 组织或用户，就会 403。

如果你不想把自己的 token 写进脚本，推荐只在 shell 里 export，不要把密钥硬编码进 repo。

## 5. 新数据里的 STATE 怎么看

### 5.1 先看 `meta/info.json`

最重要的是这一项：

```text
features.observation.state
```

直接看命令：

```bash
python - <<'PY'
import json
ds = "/home/n84416302/dataset/你的数据集名/meta/info.json"
info = json.load(open(ds))
state = info["features"]["observation.state"]
print("state shape:", state["shape"])
print("state names:", state["names"])
PY
```

如果你想顺带看图像键名：

```bash
python - <<'PY'
import json
ds = "/home/n84416302/dataset/你的数据集名/meta/info.json"
info = json.load(open(ds))
for k, v in info["features"].items():
    print(k, v["shape"] if isinstance(v, dict) and "shape" in v else None)
PY
```

### 5.2 如果有 `meta/modality.json`，可以看分组

有些数据集还会带：

```text
meta/modality.json
```

这个文件适合看 state/action 的分段定义，比如：

- 左臂 joint 在哪几维
- 右臂 joint 在哪几维
- gripper 在哪一维

命令：

```bash
python - <<'PY'
import json
ds = "/home/n84416302/dataset/你的数据集名/meta/modality.json"
print(json.dumps(json.load(open(ds)), indent=2, ensure_ascii=False))
PY
```

注意：不是每个数据集都有 `modality.json`。  
比如当前 `dual_fold_blanket_v4` 有，`take_part_bricks_v3` 不一定有；没有的话就以 `info.json` 为准。

### 5.3 当前这两个数据集的 STATE

当前两套数据：

- `dual_fold_blanket_v4`
- `take_part_bricks_v3`

它们现在的 `observation.state` 都是：

```text
shape = [14]
names = [
  left_joint_0, left_joint_1, left_joint_2, left_joint_3, left_joint_4, left_joint_5,
  left_gripper,
  right_joint_0, right_joint_1, right_joint_2, right_joint_3, right_joint_4, right_joint_5,
  right_gripper
]
```

图像键当前也是：

- `observation.images.camera_h`
- `observation.images.camera_l`
- `observation.images.camera_r`

## 6. STATE 怎么设置成自己要训练的内容

这里有一个很重要的点：

**当前 `pi05` 这条训练链默认吃的状态键是 `observation.state`。**

但这里还有一个实际训练里很容易踩的坑：

**如果你不显式设置 `--policy.input_features`，`lerobot` 会从数据集自动推断输入特征。**

自动推断时，它会把数据集里所有符合规则的输入都带进来：

- 所有 `observation.images.*` 会被当成 `VISUAL`
- 所有 `observation.*` 这类状态键都会被当成 `STATE`

这意味着如果你的数据集里除了 `observation.state` 以外，还有：

- `observation.qvel`
- `observation.effort`
- `observation.eef`
- 其他你不想训练进去的 `observation.*`

那么**不手动写 `--policy.input_features` 的话，这些键可能会一起被加载进模型输入里**。

所以这里推荐的做法不是“省略不写”，而是：

- **始终显式设置 `--policy.input_features`**
- 只把你真的要训练的状态键和图像键写进去
- 用这个方式防止额外的 observation/state 类特征被自动带进来

也就是说：

- 如果你的目标状态已经存在于 `observation.state` 里，只需要把 `shape` 配对好
- 如果你想训练的是别的状态键，比如 `observation.eef`
- 或者你只想训练 `observation.state` 里的某几个维度

那么**不能只改训练脚本里的 `shape`**，因为训练脚本不会自动帮你切片。

### 正确做法

你应该让数据集里的 `observation.state` 本身就是你要训练的状态向量，然后再训练。

比如你只想训练：

- 左右手末端位姿
- 或者只想训练某几个关节

那推荐流程是：

1. 在数据预处理阶段生成新的 `observation.state`
2. 让它只包含你要的维度
3. 同步更新 `meta/info.json` 里的：
   - `features["observation.state"]["shape"]`
   - `features["observation.state"]["names"]`
4. 重新生成对应的统计量（stats）
5. 再把训练脚本里的 `shape` 改成新的维度数

### 训练脚本里要对齐的字段

当前脚本里这一段要和数据集一致：

```bash
--policy.input_features='{
  "observation.state": {"type": "STATE", "shape": [14]},
  "observation.images.camera_h": {"type": "VISUAL", "shape": [3, 480, 640]},
  "observation.images.camera_l": {"type": "VISUAL", "shape": [3, 480, 640]},
  "observation.images.camera_r": {"type": "VISUAL", "shape": [3, 480, 640]}
}'
```

这里的作用不只是“告诉模型 shape 是多少”，更重要的是：

- 它在**白名单式**地声明：模型到底只吃哪些输入
- 只要你把 `observation.state` 和 3 个相机键显式写死，`observation.qvel`、`observation.effort`、`observation.eef` 之类的键就不会被自动带进来
- 所以如果你的目的是“只训练 joint state，不要把其他 observation state 类键也喂进去”，这一段必须保留，而且要明确写

如果你把 `observation.state` 改成 10 维，那这里就也要改成：

```bash
"observation.state": {"type": "STATE", "shape": [10]}
```

## 7. 训练脚本里最常改的参数

下面这些是最常改的。

### 7.1 数据集

```bash
--dataset.repo_id=...
--dataset.root=...
```

说明：

- `repo_id` 主要用于标识数据集
- `root` 是本地数据目录

### 7.2 预训练模型

```bash
--policy.pretrained_path=/home/n84416302/lerobot/pretrain_models/Lerobot_Pi05
```

这应该指向本地 `pi05` 预训练模型目录。

### 7.3 GPU 数和进程数

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
--num_processes=4
```

这两个必须一致。

如果你改成 2 卡：

```bash
export CUDA_VISIBLE_DEVICES=0,1
--num_processes=2
```

注意：

- `submit` 里申请几张卡，训练脚本就应该配同样的 `CUDA_VISIBLE_DEVICES` 和 `--num_processes`
- 改了卡数以后，最好也把 `output_dir`、`job_name`、`policy.repo_id`、`wandb.notes` 里的 `2gpu/4gpu` 描述一起改掉，避免名字和实际配置不一致

### 7.4 batch size

```bash
--batch_size=16
```

这是 **每卡 batch size**。  
实际 effective batch size 是：

```text
batch_size * num_processes
```

比如现在 4 卡时：

```text
16 * 4 = 64
```

### 7.5 chunk 和 action steps

```bash
--policy.chunk_size=70
--policy.n_action_steps=70
```

一般要求：

```text
n_action_steps <= chunk_size
```

如果你改了 `n_action_steps` 或 `chunk_size`，最好把下面这些名字里的 `cXXnYY` 也一起改掉：

- `--output_dir`
- `--job_name`
- `--policy.repo_id`
- `--wandb.notes`

### 7.6 训练步数和存 checkpoint

```bash
--steps=30000
--save_freq=15000
```

### 7.7 输出目录

```bash
--output_dir=...
--job_name=...
```

这两个建议每次新实验都改，不然很容易报：

```text
FileExistsError: Output directory ... already exists and resume is False
```

如果你是新实验：

- 换一个新的 `output_dir`
- 或者改新的 `job_name`

如果你是断点续训，再考虑 `resume=true`

### 7.8 wandb

```bash
--wandb.enable=true
--wandb.project=...
--wandb.entity=...
```

如果只是本地先试通：

```bash
--wandb.enable=false
```

## 8. 当前两套脚本分别对应什么

### dual_fold_blanket_v4

- train: `train4arx/train_pi05_dual_fold_blanket_v4.sh`
- submit: `submit_train_pi05_dual_fold_blanket_v4.sh`
- dataset root: `/home/n84416302/dataset/dual_fold_blanket_v4`

运行：

```bash
bash /home/n84416302/lerobot/train4arx/train_pi05_dual_fold_blanket_v4.sh
```

或：

```bash
sbatch /home/n84416302/lerobot/submit_train_pi05_dual_fold_blanket_v4.sh
```

### take_part_bricks_v3

- train: `train4arx/train_pi05_take_part_bricks_v3.sh`
- submit: `submit_train_pi05_take_part_bricks_v3.sh`
- dataset root: `/home/n84416302/dataset/take_part_bricks_v3`

运行：

```bash
bash /home/n84416302/lerobot/train4arx/train_pi05_take_part_bricks_v3.sh
```

或：

```bash
sbatch /home/n84416302/lerobot/submit_train_pi05_take_part_bricks_v3.sh
```

## 9. 常见报错怎么判断

### 9.1 `accelerate: command not found`

原因：

- 环境里没装 `training` extra

处理：

```bash
pip install -e ".[training,pi]"
```

### 9.2 `invalid choice: 'pi05'`

原因通常是：

- 进错环境
- 跑到旧版 `lerobot`
- 当前环境没有正确装到支持 `pi05` 的源码

优先检查：

```bash
python -c "import lerobot, lerobot.policies.pi05; print(lerobot.__file__)"
```

### 9.3 `wandb 403 permission denied`

原因：

- `WANDB_API_KEY` 不对
- key 对应账号没有当前 `entity/project` 的写权限

### 9.4 `huggingface 403 / GatedRepoError`

原因：

- `HF_TOKEN` 没权限访问 `google/paligemma-3b-pt-224`

### 9.5 `Output directory already exists`

原因：

- `--output_dir` 已经存在
- 但 `resume=false`

处理：

- 改一个新的 `output_dir`
- 或者手动删旧目录
- 或者明确走 resume 流程

## 10. 推荐习惯

推荐把 token 放在 shell 或 slurm 环境里，不要长期硬编码在脚本里：

```bash
export HF_TOKEN=...
export WANDB_API_KEY=...
```

每次开新实验至少改这几项：

- `--dataset.root`
- `--dataset.repo_id`
- `--output_dir`
- `--job_name`
- `--policy.repo_id`
- `--wandb.project`

如果换了 state 维度或相机键名，再改：

- `--policy.input_features`
