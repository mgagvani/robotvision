#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_navsim_stages.sh /path/to/deep_monocular.ckpt

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <deep_monocular_checkpoint>"
  exit 1
fi

CKPT_PATH="$1"
: "${NAVSIM_DEVKIT_ROOT:?NAVSIM_DEVKIT_ROOT is required}"
: "${NAVSIM_EXP_ROOT:?NAVSIM_EXP_ROOT is required}"
: "${OPENSCENE_DATA_ROOT:?OPENSCENE_DATA_ROOT is required}"

echo "[Stage 1] Metric cache on warmup split"
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
  train_test_split=warmup_two_stage \
  metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache"

echo "[Stage 1] ConstantVelocity baseline eval"
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
  train_test_split=warmup_two_stage \
  experiment_name=cv_agent \
  metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache" \
  synthetic_sensor_path="$OPENSCENE_DATA_ROOT/warmup_two_stage/sensor_blobs" \
  synthetic_scenes_path="$OPENSCENE_DATA_ROOT/warmup_two_stage/synthetic_scene_pickles"

echo "[Stage 2] Optional: Transfuser baseline (requires checkpoint override)"
echo "  python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \\"
echo "    train_test_split=warmup_two_stage experiment_name=transfuser_eval \\"
echo "    agent=transfuser_agent agent.checkpoint_path=/path/to/transfuser.ckpt ..."

echo "[Stage 4] DeepMonocular on warmup split"
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
  train_test_split=warmup_two_stage \
  experiment_name=deep_monocular_eval \
  agent=deep_monocular_agent \
  agent.checkpoint_path="$CKPT_PATH" \
  metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache" \
  synthetic_sensor_path="$OPENSCENE_DATA_ROOT/warmup_two_stage/sensor_blobs" \
  synthetic_scenes_path="$OPENSCENE_DATA_ROOT/warmup_two_stage/synthetic_scene_pickles"

echo "Done. Results are under $NAVSIM_EXP_ROOT/pdm_score/"

