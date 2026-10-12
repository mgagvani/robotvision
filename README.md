# Mechanistic Interpretability for End-to-End Self-Driving

Code for the SMTS planner (Simple Monocular Trajectory Scorer), the per-block TopK sparse
autoencoders trained on its planner-query tokens, and every analysis in the paper:
feature clustering, counterfactual visual edits, and latent steering.
Everything runs on the Waymo Open Dataset End-to-End Driving (WOD-E2E) v1.0.0 camera release.

## Layout

```
src/                    all Python; run every command from this directory (or via scripts/)
  loader.py             WOD-E2E reader (seeks into the TFRecords through index_{split}.pkl)
  models/               SMTS (monocular.py), LitModel trainer, SAE
  train.py              planner training            train_sae.py            SAE training
  extract_planner_tok.py  planner-query tokens      analyze_sae_*.py        SAE analyses
  index_{train,val}.pkl   TFRecord index (rebuild with build_index.py for a new dataset version)
sae_checkpoints/        the paper's block-0..3 SAEs (sae_block_{n}.pt)
scripts/                one Slurm job per pipeline stage; all read paths.env
paths.env               every machine-specific path; the only file to edit
```

## Setup

```bash
uv venv && uv pip install -e .                 # python >= 3.12, CUDA torch
uv pip install -e '.[visual-edits]'            # only for counterfactual-edit generation (stage 1)
cp paths.env paths.env.local 2>/dev/null; $EDITOR paths.env   # data, checkpoint, output, HF-cache locations
source paths.env                               # exports SBATCH_ACCOUNT / SBATCH_PARTITION too
```

Data: download WOD-E2E v1.0.0 (`waymo_open_dataset_end_to_end_camera_v_1_0_0`) and point
`DATA_DIR` at the TFRecord directory. The shipped `src/index_*.pkl` files index that exact
release; `python build_index.py $DATA_DIR` regenerates them.

Checkpoints: the paper's SMTS checkpoint is `$RUN_ROOT/camera-e2e-epoch=04-val_loss=2.90.ckpt`
(placed there by `python artifacts.py download`); the SAEs are in `sae_checkpoints/`. `RUN_ROOT` is the
SAE run root: `tokens/planner_tokens_{train,val}.pt` from step 2 and `model/block_{n}/sae_checkpoint.pt`
from step 3 (copy the shipped SAEs there to skip step 3).

Every SAE analysis reads `$RUN_ROOT/tokens/planner_tokens_val.pt`, so that file must have been
extracted from the same planner checkpoint the SAEs were trained on (step 2 with the paper
checkpoint). A token cache from any other checkpoint silently changes every
downstream number: with the paper checkpoint the block-3 activation filter keeps 257 of 384
features, matching the paper's 258; with a stale cache it keeps 192.

## Reproducing the paper

Submit from the repository root after `source paths.env`. Every script accepts overrides through
environment variables (see the header of each file).

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
