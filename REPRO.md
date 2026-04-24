# DeepMonocular Multi-Dataset Reproducibility

This document provides a single source of truth for reproducing DeepMonocularModel
results across Waymo, nuScenes, and NAVSIM.

## 1) Environment

### Robotvision environment

```bash
cd /anvil/scratch/x-pmathur1/robotvision
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### NAVSIM environment

```bash
cd /anvil/scratch/x-pmathur1
git clone https://github.com/autonomousvision/navsim.git
cd navsim
conda env create --name navsim -f environment.yml
conda activate navsim
pip install -e .
```

The following environment variables are set in `~/.bashrc`:

```bash
export NAVSIM_DEVKIT_ROOT="/anvil/scratch/x-pmathur1/navsim"
export NAVSIM_EXP_ROOT="/anvil/scratch/x-pmathur1/navsim_exp"
export OPENSCENE_DATA_ROOT="/anvil/scratch/x-pmathur1/navsim_data"
export NUPLAN_MAPS_ROOT="/anvil/scratch/x-pmathur1/navsim_data/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export ROBOTVISION_E2E_ROOT="/anvil/scratch/x-pmathur1/robotvision/src/camera-based-e2e"
```

## 2) Dataset setup

### Waymo E2E (robotvision open-loop)

Expected index files and shards:

- `index_train.pkl`
- `index_val.pkl`
- dataset protobuf shards in `--data-dir`

### nuScenes (robotvision open-loop)

Expected nuScenes directory with metadata + images:

- `v1.0-trainval` metadata
- synchronized camera frames and can bus data

### NAVSIM (PDM scoring)

```bash
cd $NAVSIM_DEVKIT_ROOT/download
./download_maps
./download_warmup_two_stage
# optional larger eval split
./download_navhard_two_stage
```

## 3) Baseline validation stages

### Stage A: ConstantVelocity sanity check

```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
  train_test_split=warmup_two_stage \
  metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
  train_test_split=warmup_two_stage \
  experiment_name=cv_agent \
  metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache \
  synthetic_sensor_path=$OPENSCENE_DATA_ROOT/warmup_two_stage/sensor_blobs \
  synthetic_scenes_path=$OPENSCENE_DATA_ROOT/warmup_two_stage/synthetic_scene_pickles
```

### Stage B: Transfuser baseline (optional, recommended)

```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
  train_test_split=warmup_two_stage \
  experiment_name=transfuser_eval \
  agent=transfuser_agent \
  agent.checkpoint_path=/path/to/transfuser.ckpt \
  metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache \
  synthetic_sensor_path=$OPENSCENE_DATA_ROOT/warmup_two_stage/sensor_blobs \
  synthetic_scenes_path=$OPENSCENE_DATA_ROOT/warmup_two_stage/synthetic_scene_pickles
```

## 4) DeepMonocular NAVSIM evaluation

The NAVSIM wrapper agent is provided at:

- `navsim/agents/deep_monocular/deep_monocular_agent.py`
- `navsim/agents/deep_monocular/deep_monocular_features.py`
- `navsim/planning/script/config/common/agent/deep_monocular_agent.yaml`

Run:

```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
  train_test_split=warmup_two_stage \
  experiment_name=deep_monocular_eval \
  agent=deep_monocular_agent \
  agent.checkpoint_path=/path/to/deep_monocular.ckpt \
  metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache \
  synthetic_sensor_path=$OPENSCENE_DATA_ROOT/warmup_two_stage/sensor_blobs \
  synthetic_scenes_path=$OPENSCENE_DATA_ROOT/warmup_two_stage/synthetic_scene_pickles
```

For larger local validation:

```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
  train_test_split=navhard_two_stage \
  experiment_name=deep_monocular_navhard \
  agent=deep_monocular_agent \
  agent.checkpoint_path=/path/to/deep_monocular.ckpt \
  metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache \
  synthetic_sensor_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs \
  synthetic_scenes_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles
```

## 5) Unified evaluation script

Use `scripts/evaluate.py` for single-command evaluation outputs in JSON.

### Waymo

```bash
python scripts/evaluate.py \
  --dataset waymo \
  --checkpoint /path/to/deep_monocular.ckpt \
  --split val \
  --data-dir /path/to/waymo_dataset \
  --index-file index_val.pkl \
  --output-json results/waymo_val.json
```

### nuScenes

```bash
python scripts/evaluate.py \
  --dataset nuscenes \
  --checkpoint /path/to/deep_monocular.ckpt \
  --split val \
  --data-dir /path/to/nuscenes \
  --output-json results/nuscenes_val.json
```

### NAVSIM

```bash
python scripts/evaluate.py \
  --dataset navsim \
  --checkpoint /path/to/deep_monocular.ckpt \
  --split warmup_two_stage \
  --synthetic-sensor-path $OPENSCENE_DATA_ROOT/warmup_two_stage/sensor_blobs \
  --synthetic-scenes-path $OPENSCENE_DATA_ROOT/warmup_two_stage/synthetic_scene_pickles \
  --output-json results/navsim_warmup.json
```

## 6) Metric reporting conventions

- Waymo / nuScenes:
  - `ade_mean`, `fde_mean`
  - `ade_p90`, `fde_p90`
- NAVSIM:
  - `extended_pdm_score_combined`

## 7) Artifact pinning checklist

- Record checkpoint SHA256:

  ```bash
  sha256sum /path/to/deep_monocular.ckpt
  ```

- Export environments:

  ```bash
  # robotvision env
  pip freeze > requirements.lock.txt

  # navsim env
  conda env export --name navsim > navsim_environment.lock.yml
  ```

- Tag code for paper submission:

  ```bash
  git tag -a paper-repro-v1 -m "Paper reproducibility tag"
  ```

