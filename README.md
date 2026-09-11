# Mechanistic Interpretability for End-to-End Self-Driving

Code for the SMTS planner (Simple Monocular Trajectory Scorer), the per-block TopK sparse
autoencoders trained on its planner-query tokens, and every analysis in the paper:
feature clustering, counterfactual visual edits, latent steering, and the SAE scorer adapter.
Everything runs on the Waymo Open Dataset End-to-End Driving (WOD-E2E) v1.0.0 camera release.

## Layout

```
src/                    all Python; run every command from this directory (or via scripts/)
  loader.py             WOD-E2E reader (seeks into the TFRecords through index_{split}.pkl)
  models/               SMTS (monocular.py), GTRS-Dense and DrivoR baselines, LitModel trainer, SAE
  train.py              planner training            train_sae.py            SAE training
  extract_planner_tok.py  planner-query tokens      submission.py           WOD-E2E test submission
  evaluate_waymo_ade_*.py open-loop ADE tables      analyze_sae_*.py        SAE analyses
  adapter_ladder/       scorer-adapter ablations    visualizations/paper/   figure scripts
  index_{train,val,test}.pkl   TFRecord index (rebuild with build_index.py for a new dataset version)
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

Checkpoints: `PLANNER_CHECKPOINT` is the paper's SMTS checkpoint
(`camera-e2e-epoch=04-val_loss=2.90.ckpt`); the SAEs are in `sae_checkpoints/`. `RUN_ROOT` is the
SAE run root: `tokens/planner_tokens_{train,val}.pt` from step 4 and `model/block_{n}/sae_checkpoint.pt`
from step 5 (copy the shipped SAEs there to skip step 5).

Every SAE analysis reads `$RUN_ROOT/tokens/planner_tokens_val.pt`, so that file must have been
extracted from the same planner checkpoint the SAEs were trained on (step 4 with
`PLANNER_CHECKPOINT`). A token cache from any other checkpoint silently changes every
downstream number: with the paper checkpoint the block-3 activation filter keeps 257 of 384
features, matching the paper's 258; with a stale cache it keeps 192.

## Reproducing the paper

Submit from the repository root after `source paths.env`. Every script accepts overrides through
environment variables (see the header of each file).

| # | Paper result | Command | Output |
|---|---|---|---|
| 1 | SMTS training (Sec. 4.1, App. A) | `sbatch scripts/run_train_waymo.slurm` | `$CHECKPOINT_DIR/camera-e2e-deepmonocular-waymo-n250000-*.ckpt` |
| 1b | GTRS-Dense / DrivoR baselines | `MODEL=gtrs sbatch scripts/run_train_waymo.slurm` (and `drivor`) | same |
| 1c | Data-efficiency sweep (App. "Open-Loop Planning") | `MODEL=<m> sbatch scripts/run_train_waymo_sweep.slurm` per model | 50k..250k checkpoints |
| 2 | Validation ADE@3s / ADE@5s | `sbatch scripts/run_waymo_ade_table.slurm`, then `sbatch --array=0-N scripts/run_waymo_full_val_ade.slurm` and `MODE=aggregate sbatch scripts/run_waymo_full_val_ade.slurm` | `$OUTPUT_ROOT/ade_eval/` |
| 3 | Test-set Rater Feedback Score | `MODEL=deepmonocular CHECKPOINT=... sbatch scripts/run_waymo_submission.slurm`, upload the tarball to the WOD-E2E server | `$OUTPUT_ROOT/submissions/` |
| 4 | Planner-query tokens | `sbatch scripts/run_extract_tokens.slurm` | `$RUN_ROOT/tokens/` |
| 5 | SAE training (Sec. 3.2, App. A) | `sbatch scripts/run_train_sae.slurm` | `$RUN_ROOT/model/block_{0..3}/` |
| 6 | Feature clustering and intent correlation (Sec. 4.2, App. B) | `sbatch scripts/run_sae_characterization.slurm` | `$RUN_ROOT/analysis/block_3/` |
| 7 | Latent steering and controllability (Sec. 4.5, App. D) | `sbatch scripts/run_sae_control.slurm` | `$RUN_ROOT/analysis/block_3/sae_control_*.csv` |
| 8 | Counterfactual edits, stage 1 (Sec. 4.4, App. C) | `sbatch scripts/run_visual_gen.slurm` | `$VISUAL_GEN_ROOT/manifest.jsonl` + images |
| 9 | Counterfactual edits, stage 2 | `sbatch scripts/run_visual_gen_pt2.slurm` | `$VISUAL_GEN_ROOT/sae_analysis/` |
| 10 | Fig. 3, gallery, supplement edit panels | `sbatch scripts/run_figures_counterfactual.slurm` | `$OUTPUT_ROOT/figures/` |
| 11 | Fig. 5b edit-to-steering alignment | `sbatch scripts/run_edit_to_steering_alignment.slurm` (needs 7 and 9) | `$RUN_ROOT/analysis/block_3/edit_sensitivity/` |
| 12 | Scorer adapter and ablations (Sec. 4.6, App. E) | `MODE=cache sbatch scripts/run_adapter_ladder.slurm`, then `MODE=manifest sbatch --array=0-36 scripts/run_adapter_ladder.slurm`, then `python -m adapter_ladder.stats --output_dir $OUTPUT_ROOT/adapter_ladder/out` | `$OUTPUT_ROOT/adapter_ladder/out/` |

Paper conditions for step 12: C1 = dense query adapter, C6 = random-init trained encoder,
C4 = frozen SAE latent adapter, C0 = base scorer.

Running any Python entry point directly works the same way, for example
`cd src && python train.py --data_dir $DATA_DIR --model deepmonocular`; every script prints its
flags with `--help`.

## What a reproducer needs besides this repository

| Artifact | Why | How to get it |
|---|---|---|
| WOD-E2E v1.0.0 TFRecords (1.5 TB) | everything | Waymo download under its license |
| `camera-e2e-epoch=04-val_loss=2.90.ckpt` (260 MB) | every SAE experiment reads this planner | [mgagvani/mech-interp-for-e2e-driving](https://huggingface.co/mgagvani/mech-interp-for-e2e-driving); retraining gives a similar but not identical model |
| `sae_checkpoints/*.pt` | the paper's SAEs | in this repository |
| `visual_gen_1000/` (73 MB: manifest, edited images, `sae_analysis/`) | steps 9 to 11 exactly as in the paper | same Hub repo, `counterfactual_edits/`; stage 1 is a sampled VLM plus diffusion pipeline and will not regenerate the same edits |
| GTRS / DrivoR sweep checkpoints (2.6 GB for the ten table rows) | exact ADE table and RFS rows | not published; retrain with step 1c (numbers move at the second decimal) |
| Perception Encoder and Depth-Anything-V2-Small weights | training and token extraction | auto-downloaded from Hugging Face; pre-populate the cache on clusters whose compute nodes have no internet (see `paths.env`) |
| Waymo challenge account | Rater Feedback Score | scoring happens on Waymo's server |
| Qwen3.5 checkpoint, FireRed-Image-Edit, YOLO26x weights | stage 1 of the edits only | Hugging Face / Ultralytics |

`python artifacts.py download` fetches the Hub repo and links the planner and SAEs into `RUN_ROOT` and the
edits into `VISUAL_GEN_ROOT`; then run step 4 to build the token caches. `python artifacts.py upload` is how
the repo was published.

Exactness: token extraction, SAE analyses, ADE evaluation, and the figures are deterministic
given the artifacts above (validated bit-for-bit against the pre-release code). Planner and SAE
training use fixed seeds but multi-GPU bf16 training is not bit-reproducible. The
`requirements-tested.txt` file records the exact package versions the paper artifacts were made with.

## Running on another cluster

Edit `paths.env` only. Account and partition come from `SBATCH_ACCOUNT` / `SBATCH_PARTITION`
in that file (Slurm reads them from the environment of `sbatch`), so the job scripts carry only
resource requests. Training uses 4 GPUs on one node with a global batch of 128; everything else
is a single GPU.
