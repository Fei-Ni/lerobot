#!/bin/bash
#SBATCH --job-name=train_pi05_prgvla_sort_table_r2
#SBATCH --partition=lrc-xlong
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:4
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --output=/home/n84416302/lerobot/slurm/train_pi05_prgvla_sort_the_table_leftcam_v1_restart_20260501_%j.out
#SBATCH --error=/home/n84416302/lerobot/slurm/train_pi05_prgvla_sort_the_table_leftcam_v1_restart_20260501_%j.err

set -euo pipefail

cd /home/n84416302/lerobot
bash /home/n84416302/lerobot/train4arx/train_pi05_prgvla_sort_the_table_leftcam_v1.sh
