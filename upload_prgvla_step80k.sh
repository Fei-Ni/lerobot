#!/bin/bash
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN env var (e.g. export HF_TOKEN=hf_xxx) before running this script}"
export HF_HUB_ENABLE_HF_TRANSFER=1

STACK_DIR=/home/n84416302/lerobot/results/pi05_prgvla_stack_the_cups_leftcam_v1_4gpu_b16_c60_s80k_restart_20260501/checkpoints/080000/pretrained_model
SORT_DIR=/home/n84416302/lerobot/results/pi05_prgvla_sort_the_table_leftcam_v1_4gpu_b16_c60_s80k_restart_20260501/checkpoints/080000/pretrained_model

STACK_REPO=spikefly/pi05-prgvla-stack-cups-step80k
SORT_REPO=spikefly/pi05-prgvla-sort-table-step80k

echo "=== [1/2] uploading stack_the_cups step80k -> ${STACK_REPO} ==="
hf repo create "${STACK_REPO}" --repo-type model --exist-ok
hf upload "${STACK_REPO}" "${STACK_DIR}" "." --repo-type model

echo "=== [2/2] uploading sort_the_table step80k -> ${SORT_REPO} ==="
hf repo create "${SORT_REPO}" --repo-type model --exist-ok
hf upload "${SORT_REPO}" "${SORT_DIR}" "." --repo-type model

echo "=== done ==="
