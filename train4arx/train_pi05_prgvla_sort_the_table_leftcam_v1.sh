#!/usr/bin/env bash
set -euo pipefail

export PATH=/home/n84416302/miniconda3/envs/lerobot-new/bin:$PATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_ENDPOINT=https://huggingface.co
unset HUGGINGFACE_HUB_BASE_URL

if [ -z "${HF_TOKEN:-}" ]; then
  export HF_TOKEN="$(grep '^export HF_TOKEN=' /home/n84416302/lerobot/train4arx/train_pi05_dual_fold_blanket_v4.sh | head -n1 | cut -d= -f2-)"
fi

: "${HF_TOKEN:?Please export HF_TOKEN with access to google/paligemma-3b-pt-224 before training pi05.}"

accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  "$(which lerobot-train)" \
  --dataset.repo_id=prgvla_sort_the_table_leftcam_v1 \
  --dataset.root=/home/n84416302/dataset/prgvla_sort_the_table_leftcam_v1 \
  --policy.type=pi05 \
  --policy.pretrained_path=/home/n84416302/lerobot/pretrain_models/Lerobot_Pi05 \
  --policy.train_expert_only=false \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --policy.freeze_vision_encoder=false \
  --policy.input_features='{"observation.state": {"type": "STATE", "shape": [8]}, "observation.images.camera_left": {"type": "VISUAL", "shape": [3, 480, 640]}}' \
  --policy.chunk_size=60 \
  --policy.n_action_steps=60 \
  --policy.normalization_mapping='{"ACTION": "QUANTILES", "STATE": "QUANTILES", "VISUAL": "IDENTITY"}' \
  --output_dir=/home/n84416302/lerobot/results/pi05_prgvla_sort_the_table_leftcam_v1_4gpu_b16_c60_s80k_restart_20260501 \
  --job_name=pi05_prgvla_sort_the_table_leftcam_v1_4gpu_b16_c60_s80k_restart_20260501 \
  --policy.repo_id=pi05_prgvla_sort_the_table_leftcam_v1_4gpu_b16_c60_s80k_restart_20260501 \
  --policy.push_to_hub=false \
  --batch_size=16 \
  --steps=80000 \
  --save_checkpoint=true \
  --save_freq=20000 \
  --eval_freq=20000 \
  --wandb.enable=false \
  --wandb.disable_artifact=true
