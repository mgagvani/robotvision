# Mechanistic Interpretability for End-to-End Self-Driving

## Layout

```
src/                    
  loader.py             WOD-E2E reader (seeks into the TFRecords through index_{split}.pkl)
  models/               SMTS (monocular.py), LitModel trainer, SAE
  train.py              planner training            train_sae.py            SAE training
  extract_planner_tok.py  planner-query tokens      analyze_sae_*.py        SAE analyses
  index_{train,val}.pkl   TFRecord index (rebuild with build_index.py for a new dataset version)
sae_checkpoints/        the paper's SAEs (sae_block_{n}.pt)
scripts/                one Slurm job per pipeline stage; all read paths.env
paths.env               every machine-specific path; the only file to edit
```

## Setup

```bash
uv venv && uv pip install -e . 
uv pip install -e '.[visual-edits]' 
cp paths.env paths.env.local 2>/dev/null; $EDITOR paths.env  
source paths.env 
```

Data: you will need to download WOD-E2E v1.0.0 (`waymo_open_dataset_end_to_end_camera_v_1_0_0`) and point
`DATA_DIR` at the TFRecord directory. The `src/index_*.pkl` files in the repo are indexes for this data to improve access speed. Running `python build_index.py $DATA_DIR` will regenerate them.

Checkpoints: the paper's SMTS checkpoint is `$RUN_ROOT/camera-e2e-epoch=04-val_loss=2.90.ckpt`
(placed there by `python artifacts.py download`); the SAEs are in `sae_checkpoints/`. `RUN_ROOT` is the
SAE run root: `tokens/planner_tokens_{train,val}.pt` from step 2 and `model/block_{n}/sae_checkpoint.pt`
from step 3.

Every SAE analysis reads `$RUN_ROOT/tokens/planner_tokens_val.pt`, so that file must have been
extracted from the same planner checkpoint the SAEs were trained on (step 2 with the paper
checkpoint). A token cache from any other checkpoint silently changes every
downstream number: with the paper checkpoint the block-3 activation filter keeps 257 of 384
features, matching the paper's 258; with a stale cache it keeps 192.

## Usage

Note that every script accepts overrides through environment variables (see the top of each file).

| # | Paper result | Command | Output |
|---|---|---|---|
| 1 | SMTS training (Sec. 4.1, App. A) | `sbatch scripts/run_train_waymo.slurm` | `$CHECKPOINT_DIR/camera-e2e-deepmonocular-waymo-n250000-*.ckpt` |
| 1b | Data-efficiency sweep (App. "Open-Loop Planning") | `sbatch scripts/run_train_waymo_sweep.slurm` | 50k..250k checkpoints |
| 2 | Planner-query tokens | `sbatch scripts/run_extract_tokens.slurm` | `$RUN_ROOT/tokens/` |
| 3 | SAE training (Sec. 3.2, App. A) | `sbatch scripts/run_train_sae.slurm` | `$RUN_ROOT/model/block_{0..3}/` |
| 4 | Feature clustering and intent correlation (Sec. 4.2, App. B) | `sbatch scripts/run_sae_characterization.slurm` | `$RUN_ROOT/analysis/block_3/` |
| 5 | Latent steering and controllability (Sec. 4.5, App. D) | `sbatch scripts/run_sae_control.slurm` | `$RUN_ROOT/analysis/block_3/sae_control_*.csv` |
| 6 | Counterfactual edits, stage 1 (Sec. 4.4, App. C) | `sbatch scripts/run_visual_gen.slurm` | `$VISUAL_GEN_ROOT/manifest.jsonl` + images |
| 7 | Counterfactual edits, stage 2 | `sbatch scripts/run_visual_gen_pt2.slurm` | `$VISUAL_GEN_ROOT/sae_analysis/` |
