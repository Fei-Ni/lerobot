#!/bin/bash
#SBATCH --job-name=train_pi05_take_part_bricks_v3
#SBATCH --partition=lrc-xlong
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:4
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --output=/home/n84416302/lerobot/slurm/train_pi05_take_part_bricks_v3_%j.out
#SBATCH --error=/home/n84416302/lerobot/slurm/train_pi05_take_part_bricks_v3_%j.err

set -euo pipefail

cd /home/n84416302/lerobot
export WANDB_API_KEY="${WANDB_API_KEY:-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}"

bash /home/n84416302/lerobot/train4arx/train_pi05_take_part_bricks_v3.sh
