#!/bin/bash
#SBATCH --job-name=fit_check_pi05
#SBATCH --partition=lrc-dev
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/home/n84416302/lerobot/slurm/fit_check_pi05_%x_%j.out
#SBATCH --error=/home/n84416302/lerobot/slurm/fit_check_pi05_%x_%j.err

# 通用 sbatch wrapper for train4arx/fit_check_pi05.py
# Usage:
#   sbatch -J fit_check_stack submit_fit_check_pi05.sh stack 20000        # new (restart) 训练的 ckpt
#   sbatch -J fit_check_stack submit_fit_check_pi05.sh stack 60000 old    # 老训练 64548/64549 的 ckpt
#
# Args:
#   $1 = task slug: "stack" or "sort"
#   $2 = ckpt step (e.g. 20000 / 40000 / 60000)
#   $3 = 训练版本: "new" (默认，restart_20260501) 或 "old" (老 4gpu_b16_c60_s80k 不带后缀)

set -euo pipefail

TASK="${1:?usage: $0 <stack|sort> <step> [new|old]}"
STEP="${2:?usage: $0 <stack|sort> <step> [new|old]}"
RUN="${3:-new}"
STEP_PADDED=$(printf "%06d" "$STEP")
STEP_TAG="step$(awk "BEGIN{print $STEP/1000}")k"

case "$RUN" in
  new) RUN_SUFFIX="_restart_20260501" ;;
  old) RUN_SUFFIX="" ;;
  *)   echo "unknown run: $RUN (expected new|old)" >&2; exit 2 ;;
esac

case "$TASK" in
  stack)
    DATASET=/home/n84416302/dataset/prgvla_stack_the_cups_leftcam_v1
    PROMPT="stack the cups"
    SLUG=stack_cups
    CKPT_BASE="/home/n84416302/lerobot/results/pi05_prgvla_stack_the_cups_leftcam_v1_4gpu_b16_c60_s80k${RUN_SUFFIX}"
    ;;
  sort)
    DATASET=/home/n84416302/dataset/prgvla_sort_the_table_leftcam_v1
    PROMPT="sort the table"
    SLUG=sort_table
    CKPT_BASE="/home/n84416302/lerobot/results/pi05_prgvla_sort_the_table_leftcam_v1_4gpu_b16_c60_s80k${RUN_SUFFIX}"
    ;;
  *)
    echo "unknown task: $TASK (expected stack|sort)" >&2; exit 2 ;;
esac

CKPT="$CKPT_BASE/checkpoints/$STEP_PADDED/pretrained_model"
OUT="/home/n84416302/lerobot/results/fit_check/${SLUG}_${RUN}_${STEP_TAG}"

if [ ! -d "$CKPT" ]; then
  echo "checkpoint not found: $CKPT" >&2; exit 3
fi

mkdir -p "$OUT"
cd /home/n84416302/lerobot

export PATH=/home/n84416302/miniconda3/envs/lerobot-new/bin:$PATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[wrapper] task=$TASK step=$STEP ckpt=$CKPT"
echo "[wrapper] out=$OUT"

python train4arx/fit_check_pi05.py \
  --checkpoint   "$CKPT" \
  --dataset-root "$DATASET" \
  --task-prompt  "$PROMPT" \
  --task-slug    "$SLUG" \
  --output-dir   "$OUT"
