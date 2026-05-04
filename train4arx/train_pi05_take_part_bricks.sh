#!/usr/bin/env bash
set -euo pipefail

export PATH=/home/n84416302/miniconda3/envs/lerobot-new/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT=https://huggingface.co
export WANDB_API_KEY=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
: "${HF_TOKEN:?Set HF_TOKEN env var (e.g. export HF_TOKEN=hf_xxx) before running this script}"
unset HUGGINGFACE_HUB_BASE_URL


accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  "$(which lerobot-train)" \
  --dataset.repo_id=yaoxianze/take_part_bricks \
  --dataset.root=/home/n84416302/dataset/take_part_bricks \
  --policy.type=pi05 \
  --policy.pretrained_path=/home/n84416302/lerobot/pretrain_models/Lerobot_Pi05 \
  --policy.train_expert_only=false \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --policy.freeze_vision_encoder=false \
  --policy.input_features='{"observation.state": {"type": "STATE", "shape": [14]}, "observation.images.camera_h": {"type": "VISUAL", "shape": [3, 480, 640]}, "observation.images.camera_l": {"type": "VISUAL", "shape": [3, 480, 640]}, "observation.images.camera_r": {"type": "VISUAL", "shape": [3, 480, 640]}}' \
  --policy.chunk_size=80 \
  --policy.n_action_steps=80 \
  --policy.normalization_mapping='{"ACTION": "QUANTILES", "STATE": "QUANTILES", "VISUAL": "IDENTITY"}' \
  --output_dir=/home/n84416302/lerobot/results/arx_pi05_take_part_bricks_4gpu_b16_c80n80_stateonly_s30k \
  --job_name=pi05_take_part_bricks_4gpu_b16_c80n80_stateonly_s30k \
  --policy.repo_id=yaoxianze/arx_pi05_take_part_bricks_4gpu_b16_c80n80_stateonly_s30k \
  --policy.push_to_hub=false \
  --batch_size=16 \
  --steps=40000 \
  --save_checkpoint=true \
  --save_freq=15000 \
  --wandb.enable=true \
  --wandb.project=arx_pi05_take_part_bricks \
  --wandb.entity=yao-xian-ze \
  --wandb.notes="dataset=yaoxianze/take_part_bricks; root=/home/n84416302/dataset/take_part_bricks; machine=local_4gpu; policy=pi05; inputs=state+camera_h/l/r; chunk=80; n_action=80; expert_only=0; batch_per_gpu=16; steps=30000; save_freq=15000" \
  --wandb.disable_artifact=true
