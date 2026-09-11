#!/bin/bash
# Scorer-adapter capacity ladder (Appendix "Scorer Adapter Details and Ablations").
#   MODE=cache    sbatch scripts/run_adapter_ladder.slurm        # slim train/val caches from the token files
#   MODE=manifest sbatch --array=0-36 scripts/run_adapter_ladder.slurm   # one run per manifest row
#   then:  python -m adapter_ladder.stats --output_dir $LADDER_OUT ; python -m adapter_ladder.plot --input_dir $LADDER_OUT
#SBATCH --job-name=adapter-ladder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
set -euo pipefail
source "${REPO_ROOT:-$SLURM_SUBMIT_DIR}/paths.env"
cd "$REPO_ROOT/src"
LADDER_ROOT=${LADDER_ROOT:-$OUTPUT_ROOT/adapter_ladder}
LADDER_OUT=${LADDER_OUT:-$LADDER_ROOT/out}
TRAIN_CACHE=$LADDER_ROOT/slim_train.pt
VAL_CACHE=$LADDER_ROOT/slim_val.pt
SAE_CKPT=${SAE_CKPT:-$RUN_ROOT/model/block_3/sae_checkpoint.pt}
MODE=${MODE:-run}
CONDITION=${CONDITION:-C4}; ENCODER_DRAW=${ENCODER_DRAW:-0}; ADAPTER_SEED=${ADAPTER_SEED:-0}; HIDDEN_DIM=${HIDDEN_DIM:-256}
mkdir -p "$LADDER_ROOT" "$LADDER_OUT"
if [[ "$MODE" == "cache" ]]; then
  "$PYTHON_BIN" -c "
from pathlib import Path
from adapter_ladder.data import build_slim_cache
print('val  ', build_slim_cache(Path('$RUN_ROOT/tokens/planner_tokens_val.pt'), Path('$VAL_CACHE')))
print('train', build_slim_cache(Path('$RUN_ROOT/tokens/planner_tokens_train.pt'), Path('$TRAIN_CACHE')))
"
  exit 0
fi
if [[ "$MODE" == "manifest" ]]; then
  MANIFEST=$LADDER_OUT/manifest.json
  [[ -f "$MANIFEST" ]] || "$PYTHON_BIN" -m adapter_ladder.stats --output_dir "$LADDER_OUT" --make_manifest
  read -r CONDITION ENCODER_DRAW ADAPTER_SEED HIDDEN_DIM < <("$PYTHON_BIN" -c "
import json; m=json.load(open('$MANIFEST'))[$SLURM_ARRAY_TASK_ID]
print(m['condition'], m['encoder_draw'], m['adapter_seed'], m['hidden_dim'])")
fi
"$PYTHON_BIN" -m adapter_ladder.train \
  --condition "$CONDITION" --encoder_draw "$ENCODER_DRAW" --adapter_seed "$ADAPTER_SEED" --hidden_dim "$HIDDEN_DIM" \
  --batch_size 16 --recipe paper \
  --train_cache "$TRAIN_CACHE" --val_cache "$VAL_CACHE" --sae_checkpoint "$SAE_CKPT" --output_dir "$LADDER_OUT"
