EXP_DIR=/home/n84416302/lerobot/results/arx_pi05_take_part_bricks_4gpu_b16_c80n80_stateonly_s30k
REPO_ID=yaoxianze/pi05_take_part_bricks_80chunk
: "${HF_TOKEN:?Set HF_TOKEN env var (e.g. export HF_TOKEN=hf_xxx) before running this script}"

cd "$EXP_DIR"

echo "开始上传30k大重量权重..."
HF_HUB_ENABLE_HF_TRANSFER=1 HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var (e.g. export HF_TOKEN=hf_xxx) before running this script}" \
hf upload "$REPO_ID" \
  "$EXP_DIR/checkpoints/030000" \
  "checkpoints/030000" \
  --repo-type model

echo "开始上传40k大重量权重..."
HF_HUB_ENABLE_HF_TRANSFER=1 HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var (e.g. export HF_TOKEN=hf_xxx) before running this script}" \
hf upload "$REPO_ID" \
  "$EXP_DIR/checkpoints/040000" \
  "checkpoints/040000" \
  --repo-type model

echo "开始上传轻量级文件..."
HF_HUB_ENABLE_HF_TRANSFER=1 HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var (e.g. export HF_TOKEN=hf_xxx) before running this script}" \
hf upload "$REPO_ID" \
  "$EXP_DIR" \
  "." \
  --repo-type model \
  --exclude "checkpoints/*" 
