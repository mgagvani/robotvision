#!/bin/bash
# Counterfactual visual edits, stage 1: YOLO detections -> VLM edit proposal (served locally by vLLM)
# -> diffusion image edit. Writes $VISUAL_GEN_ROOT/manifest.jsonl plus the edited images.
# Needs the optional deps:  pip install -e '.[visual-edits]'   and the YOLO weights ($YOLO_MODEL).
#SBATCH --job-name=visual-gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256g
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
set -euo pipefail
source "${REPO_ROOT:-$SLURM_SUBMIT_DIR}/paths.env"
cd "$REPO_ROOT/src"
NUM_ITEMS=${NUM_ITEMS:-1000}
OUTPUT_DIR=${OUTPUT_DIR:-$VISUAL_GEN_ROOT}
VLLM_BIN=${VLLM_BIN:-$(dirname "$PYTHON_BIN")/vllm}
export BASE_URL=http://127.0.0.1:8000/v1 API_KEY=EMPTY MODEL=qwen3.5 VLLM_DEEP_GEMM_WARMUP=skip SGLANG_MAMBA_CONV_DTYPE=float16
mkdir -p "$OUTPUT_DIR"

cleanup() { [[ -n "${VLLM_PID:-}" ]] && kill "$VLLM_PID" 2>/dev/null && wait "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT
"$VLLM_BIN" serve "$QWEN_CHECKPOINT" --served-model-name "$MODEL" --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.35 --max-num-seqs 64 --max-model-len 8192 \
  --dtype float16 --limit-mm-per-prompt '{"image": 1}' --reasoning-parser qwen3 \
  > "$OUTPUT_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
for i in $(seq 1 180); do
  curl -fsS "$BASE_URL/models" >/dev/null 2>&1 && break
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM exited early"; tail -n 100 "$OUTPUT_DIR/vllm.log"; exit 1; }
  sleep 10
done
curl -fsS "$BASE_URL/models" >/dev/null || { echo "vLLM not ready"; exit 1; }

"$PYTHON_BIN" analyze_sae_visual_gen.py generate-edits \
  --data_dir "$DATA_DIR" \
  --index_file index_val.pkl \
  --output_dir "$OUTPUT_DIR" \
  --max_items "$NUM_ITEMS" \
  --yolo_model "${YOLO_MODEL:-yolo26x.pt}" \
  --generator local \
  --editor firered
