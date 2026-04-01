# Proposal Planner — Debug & Fix Changelog

## Phase 1: Debug Instrumentation

### Problem
The proposal planner had terrible performance after 10 epochs of training:
- Train ADE ~16.8m, Val ADE ~14.3m (extremely high for 5-second horizon)
- All 16 proposals collapsed to the same trajectory (zero regret, zero diversity)
- Score loss dominated total loss by ~25x

### What was added

**`models/debug_callbacks.py`** — `GradientDebugCallback` (Lightning Callback)
- Per-module gradient L2 norms: `scene_encoder`, `proposal_init`, `refinement` (per block), `scorer`
- Per-sublayer gradient norms within refinement (`cross_attn`, `mlp`, `traj_residual`, `traj_enc`)
- Per-sublayer gradient norms within scorer (`geom_proj`, `score_mlp`) and proposal_init (`ego_enc`, `traj_decoder`, `proposal_embed`)
- Parameter norms per module (detect weight explosion/vanishing)
- Gradient dominance ratio (max_module / total — detect when one module dominates)
- Activation statistics via forward hooks: mean, std, abs_max, dead fraction, saturated fraction
- Proposal diagnostics: endpoint spread, pairwise diversity, trajectory lengths, score distribution, refinement delta

**`debug_viz.py`** — Post-training diagnostic plot generator
- 10 individual plots + 1 summary dashboard
- Reads `metrics.csv` from Lightning CSVLogger
- Can be run standalone: `python debug_viz.py --log_dir /path/to/logs`

**`train.py`** changes:
- `--debug` flag enables the callback and auto-generates plots after training
- `--debug_log_every N` controls logging frequency

### Diagnosis from debug plots

| Signal | Observation | Implication |
|--------|-------------|-------------|
| `loss_score=427` vs `loss_ade=16.8` | Score loss 25x larger | Score loss dominates optimization |
| `ade_pred ≈ ade_oracle`, regret ≈ 0 | All proposals identical | Complete mode collapse |
| `grad_norm/proposal_init` ~1000 | 10x higher than everything else | 90% of gradient signal goes through proposal_init |
| `grad_norm/scene_encoder` ~100 | Order of magnitude lower | Scene encoder barely learns |
| `proposals/pairwise_dist` decreasing | Proposals converging | Diversity loss too weak |
| `proposals/score_range` near 0 | Scorer can't discriminate | All proposals look the same to scorer |
| `act/scene_enc_out_mean` ≈ 0 | Scene features near zero | Visual info not reaching proposals |
| `grad_norm/propinit_embed` very low | Proposal seeds barely updated | Can't maintain diversity |

### Root cause chain
1. Score loss (MSE to ADE, weight=1.0) produces loss ~400, dwarfing ADE loss ~15
2. Score loss gradient flows through scorer + proposal_init, bypassing refinement/scene_encoder
3. Without gradient pressure to diversify, all proposals collapse to the mean trajectory
4. Once collapsed, diversity loss = 0, refinement has nothing to differentiate, cross-attention degenerates
5. Scene encoder never gets gradient signal because its only path (refinement cross-attn) is starved

---

## Phase 2: Architecture & Loss Fixes

### Fix 1: Scene-conditioned ProposalInit (`models/proposal_init.py`)

**Problem:** ProposalInit only saw ego trajectory + intent. Camera images had no direct path
to the ADE loss — their only route was through the refinement cross-attention bottleneck,
which received 10x less gradient signal than proposal_init.

**Fix:** Added cross-attention from proposal embeddings to scene feature tokens inside
ProposalInit. This gives the scene encoder a direct gradient path:
`ADE loss → proposals → traj_decoder → scene_cross_attn → scene_encoder.proj`

Also increased `proposal_embed` init std from 0.02 to 0.1 to provide more initial diversity
among proposal seeds (wider starting spread resists early collapse).

### Fix 2: Score loss downweight + warmup (`models/base_model.py`)

**Problem:** Score loss (MSE between predicted scores and actual ADEs) was weighted 1.0x,
producing loss ~400 vs ADE loss ~15. This overwhelmed the trajectory quality signal and
drove all proposals toward identical outputs.

**Fix:**
- New `score_weight` parameter (default 0.1 instead of implicit 1.0)
- New `score_warmup_epochs` parameter (default 2): score loss is 0 for the first N epochs,
  then ramps linearly to `score_weight` over 1 epoch. This lets proposals diversify before
  the scorer starts pulling them toward similar predictions.

### Fix 3: Detach proposals from score loss gradient (`models/base_model.py`)

**Problem:** The scorer's MSE loss backpropagated through both the scorer weights AND the
proposal generation pipeline. Since all proposals had similar ADE, the scorer gradient
effectively pushed proposals to produce outputs that are easy to score (i.e., identical).

**Fix:** `pred_scores.detach().argmin(dim=1)` when selecting the best proposal for ADE
computation. The scorer still learns to rank, but its gradients only update scorer weights,
not the proposal generator. The proposal generator is driven only by ADE + diversity losses.

### Fix 4: Diversity weight increase (`train.py`, `sbatch`)

**Problem:** Diversity weight was 0.1, but the diversity loss magnitude (~0.3) was negligible
compared to score loss (~400). Even after downweighting score loss, diversity needs to be
strong enough to prevent collapse.

**Fix:** Default `diversity_weight` changed from 0.1 to 1.0.

### Fix 5: Gradient clipping (`train.py`)

**Problem:** Gradient norms reached ~1000 with high variance (spikes visible in plots),
causing unstable updates especially for the scorer.

**Fix:** Added `gradient_clip_val=1.0` with norm clipping to the Lightning Trainer.
New `--grad_clip` CLI arg (0 to disable).

### Summary of parameter changes

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `score_weight` | 1.0 (implicit) | 0.1 | Was 25x larger than ADE, dominated optimization |
| `score_warmup_epochs` | 0 (N/A) | 2 | Let proposals diversify before scorer activates |
| `diversity_weight` | 0.1 | 1.0 | Too weak to prevent collapse against score loss |
| `proposal_embed` std | 0.02 | 0.1 | Wider initial spread resists early collapse |
| `gradient_clip_val` | None | 1.0 | Stabilize training, prevent gradient spikes |
| ProposalInit inputs | past + intent | past + intent + scene_feat | Give scene encoder a direct gradient path |

### Files changed

| File | Change |
|------|--------|
| `models/proposal_init.py` | Added scene cross-attention, increased embed init std |
| `models/proposal_planner.py` | Pass `scene_feat` to `proposal_init()` |
| `models/base_model.py` | Added `score_weight`, `score_warmup_epochs`, detached scorer gradients |
| `train.py` | New CLI args: `--score_weight`, `--score_warmup_epochs`, `--grad_clip`; updated defaults |
| `scripts/run_train_proposal.sbatch` | Updated args for new defaults |
| `models/debug_callbacks.py` | Created (Phase 1) |
| `debug_viz.py` | Created (Phase 1) |
| `CHANGES.md` | This file |

---

## Phase 3: iPad-inspired Architectural Changes

Reference: [iPad — Iterative Proposal-centric End-to-End Autonomous Driving](https://arxiv.org/abs/2505.15111)

Phase 2 fixed the optimization dynamics (score loss no longer dominates, proposals no
longer collapse), but raw ADE stayed at ~14.3m throughout training. The model
architecture itself had structural limitations that prevented learning to generate
accurate trajectories. Phase 3 addresses three key gaps relative to the iPad paper.

### Fix 6: BCE scorer loss with quality target (`models/base_model.py`, `models/scorer.py`)

**Problem:** The old MSE scorer trained on raw ADE values (range 0–50+). This
produced unbounded loss magnitudes, required aggressive downweighting, and gave the
scorer no natural notion of "good" vs "bad" — an ADE of 12 vs 13 had the same loss
gradient magnitude as 1 vs 2, even though the latter distinction is far more important.

**Fix:** Replaced `MSE(logit, ADE)` with `BCE(sigmoid(logit), exp(-ADE/τ))`:
- Quality target `exp(-ADE/τ)` maps ADE into [0, 1] — lower ADE → higher quality
- BCE loss is naturally bounded (~0.0 to ~0.7), no need for aggressive downweighting
- The exponential mapping concentrates learning on distinguishing good proposals
  (low ADE) rather than wasting capacity on large-ADE differences
- Temperature τ (default 5.0) controls sensitivity: lower τ → sharper discrimination
- Scorer output semantics flipped: **higher score = better** (was: lower = better)
- All `argmin` for score-based selection → `argmax` throughout the codebase
- Default `score_weight` raised from 0.1 to 1.0 (BCE is well-scaled, safe at 1.0)

New CLI arg: `--score_temperature` (default 5.0)

### Fix 7: Weight-shared iterative refinement (`models/refinement.py`)

**Problem:** The old refinement used `num_steps` separate `RefinementBlock` instances,
each with independent weights. This meant:
- Each block only saw gradients from a single position in the chain
- Parameter count scaled linearly with `num_steps` (more steps = more parameters)
- No iterative self-correction: each block learned a fixed transformation, not a
  general "look at the scene and improve proposals" operation

**Fix:** iPad-style weight sharing — a **single** `RefinementBlock` applied
`num_steps` times in a loop:
- Same weights at every iteration → the block learns a general refinement operation
- Increasing `num_steps` (e.g., 2 → 4) adds compute but zero extra parameters
- Gradients flow through all iterations (like an RNN), giving the shared weights
  stronger learning signal from multiple applications

**Practical benefit:** `--num_refinement_steps 4` now uses the same parameter count
as the old 2-step model, but applies 4 iterations of refinement.

### Fix 8: Full trajectory re-prediction (`models/refinement.py`)

**Problem:** The old `RefinementBlock` predicted **residual** waypoint deltas:
`proposals_new = proposals + traj_residual(feat)`. This constrained each iteration
to small perturbations around the previous trajectory, preventing the model from
fundamentally correcting bad initial proposals.

**Fix:** Replaced residual prediction with **full re-prediction**:
`proposals_new = traj_decoder(feat)`. At each iteration the model predicts complete
new trajectories from scratch based on the current proposal features. The features
still accumulate information across iterations (via residual connections in the
feature update path), but the trajectories are free to change substantially.

This matches iPad's approach where proposals are predicted fresh at each iteration
of the ProFormer loop.

### Summary of Phase 3 parameter changes

| Parameter | Phase 2 | Phase 3 | Reason |
|-----------|---------|---------|--------|
| Score loss | MSE | BCE with `exp(-ADE/τ)` target | Bounded loss, [0,1] quality scale |
| `score_weight` | 0.1 | 1.0 | BCE is naturally well-scaled |
| `score_temperature` | N/A | 5.0 (new) | Controls quality target sensitivity |
| Score semantics | lower = better | higher = better | Matches BCE probability interpretation |
| Refinement weights | Independent per step | Shared single block | Stronger gradient signal, free to add iterations |
| `num_refinement_steps` | 2 | 4 (sbatch) | More iterations at zero parameter cost |
| Trajectory prediction | Residual (`proposals + delta`) | Full re-prediction | Can correct bad proposals, not just perturb |

### Phase 3 files changed

| File | Change |
|------|--------|
| `models/refinement.py` | Single shared `RefinementBlock`, `traj_residual` → `traj_decoder`, full re-prediction |
| `models/scorer.py` | Updated docstring: higher = better, BCE-trained |
| `models/base_model.py` | BCE loss with quality target, `argmin` → `argmax`, `score_temperature` param |
| `models/debug_callbacks.py` | Updated for shared block (`.block` not `.blocks`), flipped score argmin/argmax |
| `train.py` | New `--score_temperature` arg, `score_weight` default 0.1 → 1.0 |
| `scripts/run_train_proposal.sbatch` | `num_refinement_steps` 2 → 4, `score_weight` 0.1 → 1.0, added `score_temperature` |
| `CHANGES.md` | This section |

---

## Phase 4: Full iPad-faithful Rewrite

Reference: [iPad source code](https://github.com/Kguo-cs/iPad) and [paper appendix](https://arxiv.org/html/2505.15111v1)

Phase 3 changes did not move the needle: ADE stayed at 14.3m with complete mode
collapse (all proposals identical, zero regret, zero diversity, score loss stuck at
ln(2)=0.693). After studying the iPad source code in detail, several fundamental
architectural and loss differences were identified. Phase 4 is a faithful rewrite
copying the core iPad design patterns.

### Root cause: Why Phases 2-3 failed

The model was still stuck because:
1. **Top-5 WTA L2 loss** averaged the best 5 proposals' L2 errors — this pulled ALL
   proposals toward the mean trajectory, actively fighting diversity.
2. **One feature per proposal** (B, K, C) was too limited to represent a 20-timestep
   trajectory. Each proposal had a single 256-dim vector to encode 40 coordinates.
3. **No intermediate supervision** — only the final proposals were supervised. The
   shared refinement block had no gradient signal for early iterations.
4. **Diversity loss as a band-aid** — negative mean pairwise distance is a weak signal.
   iPad doesn't use diversity loss at all; MoN naturally handles it.

### Fix 9: Per-timestep proposal features (`models/proposal_init.py`)

**Problem:** Each proposal had one feature vector (B, K, C). A single 256-dim vector
must encode all 20 timesteps' (x,y) positions, severely limiting expressiveness.

**Fix:** Learnable embeddings of shape (N×T, C) = (16×20, C) = (320, C). Each
proposal gets T=20 separate feature vectors, one per timestep. This matches iPad's
`init_feature = nn.Embedding(poses_num * proposal_num, tf_d_model)`.

The ego status (past trajectory + intent) is encoded to a single vector and broadcast-
added to all N×T tokens, exactly matching iPad's pattern:
`bev_feature = ego_feature + init_feature.weight[None]`

Removed: scene cross-attention from ProposalInit (iPad doesn't have it — scene
features are incorporated only through refinement cross-attention).

### Fix 10: Per-timestep trajectory decoding (`models/refinement.py`)

**Problem:** The old decoder mapped a single (B, K, C) feature to a full (B, K, T×2)
trajectory via one MLP. This forced a single vector to predict 40 output dimensions.

**Fix:** Each timestep's feature independently predicts that timestep's (x,y):
`proposals = traj_decoder(bev_feature)  # (B, N*T, C) → (B, N*T, 2) → (B,N,T,2)`

The decoder is a 2-layer MLP with only 2 output dimensions per timestep (matching
iPad's `MLP(d_model, d_ffn, state_size)` where state_size=2 for us).

### Fix 11: Predict-then-refine loop with intermediate proposals (`models/refinement.py`)

**Problem:** Old refinement encoded proposals back and then predicted residuals.
No intermediate proposals were returned for supervision.

**Fix:** iPad-faithful predict→encode→attend→refine loop:
1. `proposals = traj_decoder(bev_feature)` — predict from current features
2. `bev_feature += traj_enc(proposals)` — encode predictions back (proposal-anchoring)
3. `bev_feature += cross_attn(bev_feature, scene_feat)` — attend to scene
4. `bev_feature += mlp(bev_feature)` — FFN update
5. Return both proposals and updated features
6. Collect `proposal_list` across all K iterations for intermediate supervision

### Fix 12: MoN L1 loss with discounted intermediate supervision (`models/base_model.py`)

**Problem:** Top-5 WTA L2 ADE loss averaged the best 5 modes, pulling all proposals
toward the mean and causing mode collapse. Only final proposals were supervised.

**Fix:** Faithful copy of iPad's loss:
- **MoN L1**: `min_n mean_t (|Δx| + |Δy|)` — Minimum over N proposals of mean L1
  displacement. Only the SINGLE closest proposal to GT is optimized, leaving others
  free to explore different modes. L1 is more robust to outliers than L2.
- **Discounted intermediate supervision**: `L = Σ_k λ^(K-1-k) × MoN_L1(P_k)`
  with λ=0.1 (iPad's `prev_weight`). Later iterations get stronger supervision,
  earlier ones are relaxed. This gives gradient signal at every refinement step.
- **Removed diversity loss** (set to weight 0). MoN naturally encourages diversity
  because only one proposal is pulled toward GT per sample — the others receive no
  gradient and can remain spread out. iPad uses `inter_weight=0`.

### Fix 13: Simplified scorer (`models/scorer.py`)

**Problem:** Old scorer fused learned proposal features with handcrafted geometric
features (velocity norms, acceleration norms, flat waypoints). This was complex and
the geometric features were redundant given per-timestep features.

**Fix:** Match iPad's scorer exactly:
1. Reshape per-timestep features: (B, N×T, C) → (B, N, T, C)
2. Max-pool over temporal dimension: (B, N, T, C) → (B, N, C)
3. MLP → scalar logit: (B, N, C) → (B, N)

This is simpler and matches iPad's `proposal_feature = bev_feature.amax(-2)` followed
by `pred_score(proposal_feature)`.

### Fix 14: Score quality target uses L1 (`models/base_model.py`)

The scorer's BCE quality target now uses L1 displacement (consistent with the
trajectory loss) instead of L2 ADE: `quality = exp(-L1_per_mode / τ)`.

### Summary of Phase 4 parameter changes

| Parameter | Phase 3 | Phase 4 | Reason |
|-----------|---------|---------|--------|
| Trajectory loss | Top-5 WTA L2 ADE | MoN L1 | Only best proposal supervised; L1 more robust |
| Intermediate supervision | None | Discounted λ=0.1 | Gradient signal at every refinement step |
| `prev_weight` | N/A | 0.1 (new) | iPad discount factor for earlier iterations |
| Proposal features | (B, K, C) | (B, K×T, C) | Per-timestep features, 20× more expressive |
| Trajectory decoder | MLP(C → T×2) per proposal | MLP(C → 2) per timestep | Each timestep independently decoded |
| Scorer | Geometric + learned features | Max-pool temporal + MLP | Simpler, matches iPad |
| `diversity_weight` | 1.0 | 0.0 | MoN handles diversity; explicit loss fights it |
| `smoothness_weight` | 0.01 | 0.0 | Simplify loss to match iPad |
| `comfort_weight` | 0.01 | 0.0 | Simplify loss to match iPad |
| `max_epochs` | 10 | 20 | iPad trains for 20 epochs |

### Phase 4 files changed

| File | Change |
|------|--------|
| `models/proposal_init.py` | Rewritten: per-timestep embeddings (N×T, C), removed scene cross-attn |
| `models/refinement.py` | Rewritten: predict-then-refine loop, returns proposal_list |
| `models/scorer.py` | Rewritten: max-pool temporal + MLP, removed geometric features |
| `models/proposal_planner.py` | Updated: new calling pattern, passes proposal_list through |
| `models/base_model.py` | MoN L1 loss, discounted intermediate supervision, removed diversity loss |
| `models/debug_callbacks.py` | Updated for new architecture (scorer, propinit, proposal_list) |
| `train.py` | New `--prev_weight` arg, `diversity_weight` default 0, `num_refinement_steps` default 4 |
| `scripts/run_train_proposal.sbatch` | Updated all params, 20 epochs |
| `CHANGES.md` | This section |

---

## Phase 5: Scorer Improvements (Ranking-focused)

Phase 4 achieved strong multimodal proposals (val `ade_oracle` ~0.57 m) but deployed
trajectory quality was bottlenecked by scorer ranking: val `ade_pred` ~1.66 m with
`ade_regret` ~1.09 m (66% of total error). The scorer couldn't reliably pick the
best proposal from the diverse set MoN produced.

### Fix 15: Cross-entropy ranking loss (`models/base_model.py`)

**Problem:** Per-proposal BCE with soft quality targets `exp(-L1/τ)` does not directly
optimize for correct ranking. The targets for good vs mediocre proposals are close
together (e.g., 0.90 vs 0.82 with τ=5), giving weak ranking signal. A scorer can
achieve low BCE loss while still ranking proposals incorrectly.

**Fix:** Replaced BCE with cross-entropy classification:
`loss_score = CrossEntropy(logits, argmin(L1_per_mode))`. This directly optimizes
for `argmax(scores) == argmin(L1)`, exactly matching how proposals are selected at
inference. The `score_temperature` parameter is no longer used by the loss.

### Fix 16: Detached scorer inputs (`models/scorer.py`)

**Problem:** Score loss gradients flowed back through the proposal generation pipeline
via `bev_feature`, conflicting with MoN trajectory loss. MoN wants only one proposal
to move toward GT; score BCE pushed all proposal features toward their individual
quality targets through shared representations.

**Fix:** `bev_feature.detach()` before pooling in the scorer forward pass. Score loss
now only updates scorer weights, making it a pure discriminator. MoN has exclusive
control over proposal generation — no gradient conflict.

### Fix 17: Trajectory-aware scorer (`models/scorer.py`)

**Problem:** The scorer only saw abstract learned features (`bev_feature`). After
cross-attention and MLP updates, features for different proposals can be very similar
even when decoded trajectories differ substantially. The scorer was ranking based on
features that didn't fully capture trajectory-level differences.

**Fix:** Concatenate projected trajectory coordinates with pooled BEV features:
- `traj_proj`: MLP mapping (T*2) to 64-dim trajectory embedding per proposal
- `score_mlp` input: `[max_pooled_feat | traj_feat]` with LayerNorm before the MLP
- Both `bev_feature` and `proposals` are detached so score loss is fully isolated

This gives the scorer explicit spatial information (velocity profiles, endpoint
locations, curvature) for more discriminative ranking.

### Summary of Phase 5 changes

| Parameter | Phase 4 | Phase 5 | Reason |
|-----------|---------|---------|--------|
| Score loss | BCE with `exp(-L1/tau)` target | Cross-entropy on `argmin(L1)` | Directly optimizes ranking for `argmax` selection |
| Scorer input gradient | Flows back to proposals | Detached | Eliminates gradient conflict with MoN |
| Scorer input features | Max-pooled BEV only | BEV + projected trajectory coords | Explicit spatial discrimination |
| Scorer MLP input | `d_model` | `d_model + traj_dim` with LayerNorm | Richer, normalized input |
| `score_temperature` | Used (tau=5.0) | Unused (kept as CLI arg) | CE loss doesn't need soft targets |

### Phase 5 files changed

| File | Change |
|------|--------|
| `models/scorer.py` | Rewritten: detached inputs, traj_proj, LayerNorm, updated forward signature |
| `models/proposal_planner.py` | Pass `proposals` to `scorer(bev_feature, proposals)` |
| `models/base_model.py` | BCE to cross-entropy with `argmin(L1)` label |
| `models/debug_callbacks.py` | Added `scorer_traj_proj` gradient norm logging |
| `CHANGES.md` | This section |

---

## Phase 6: Multi-loss scorer experiments

To enable parallel scorer-loss ablations from one codebase, scorer loss is now
configurable via CLI:

- `bce`: iPad-faithful BCE with soft quality target `exp(-L1/tau)`
- `ce`: hard top-1 cross-entropy on `argmin(L1)`
- `bce_pairwise`: BCE + pairwise margin-ranking auxiliary
- `listnet`: listwise KL objective with `softmax(-L1/tau)` target distribution

New args in `train.py`:
- `--score_loss_type`
- `--score_rank_weight`
- `--score_margin`
- `--score_topk`

Additional scorer diagnostics are logged to metrics:
- `*_score_top1_acc`: `argmax(score) == argmin(L1)` accuracy
- `*_score_gap_best_second`: average logit gap between oracle-best and second-best mode

---

## Phase 7: NAVSIM-style quality target

### Motivation (iPad paper Section 3.3)

The iPad scorer uses a multi-factor ground-truth score from log-replay simulation
(NAVSIM Eq. 5): `S = NC * DAC * (5*EP + 5*TTC + 2*Comf) / 12`, covering safety,
efficiency, and comfort — not just trajectory closeness. Our `exp(-L1/tau)` target
only captures geometric accuracy, which may explain why the scorer struggles to
discriminate between proposals that are close in L1 but differ in driving quality.

### Fix 18: Geometric NAVSIM approximation (`models/base_model.py`)

Without agent trajectories or road boundaries in the current data pipeline, we
approximate the NAVSIM sub-metrics from trajectory geometry alone:

| Sub-metric | iPad weight | Our approximation |
|---|---|---|
| EP (Ego Progress) | 5/12 | `clamp(dot(prop_disp, gt_dir) / gt_dist, 0, 1)` |
| TTC (Time-to-Collision) | 5/12 | 1.0 (no agent data) |
| Comf (Comfort) | 2/12 | fraction of timesteps with jerk < threshold |
| NC (No at-fault Collision) | gate | 1.0 (no agent data) |
| DAC (Drivable Area Compliance) | gate | 1.0 (no map data loaded) |

Combined: `quality = (5*EP + 5*1.0 + 2*Comf) / 12`, clamped to [0, 1].

New `_compute_quality_target()` method dispatches between `l1` (existing) and
`navsim` (new) based on `--score_target_type` CLI arg.

### New CLI args

- `--score_target_type`: `l1` (default, backward-compatible) or `navsim`
- `--comfort_jerk_threshold`: jerk threshold in m/s^3 for comfort metric (default 5.0)

### Files changed

| File | Change |
|------|--------|
| `models/base_model.py` | Added `_compute_navsim_score()`, `_compute_quality_target()`, new hparams |
| `train.py` | New `--score_target_type`, `--comfort_jerk_threshold` args |
| `scripts/run_train_proposal.sbatch` | Added new default args |
| `score_experiment_viz.py` | Tracks `score_target_type` in approach labels and summary |
| `CHANGES.md` | This section |

---

## Phase 8: RFS-based scorer quality target

### Motivation

For datasets where full NAVSIM-style supervision is unavailable, we want the scorer
to learn proposal ranking from an RFS-style quality signal (longitudinal/lateral
deviation with speed-aware thresholds) instead of only `exp(-L1/tau)`.

### Fix 19: Add per-proposal RFS quality target (`models/base_model.py`)

Added a new scorer target path that computes `(B, K)` quality directly from RFS
logic for **all proposals** (not only top-1):

- New method: `_compute_rfs_quality(proposals, reference, past)`
  - Uses the same directional decomposition as existing `rfs_loss`:
    - longitudinal and lateral error components
    - time thresholds at 3s / 5s (indices 11, 19)
    - speed scaling from the existing RFS formula
  - Converts deviation to score with the same piecewise definition:
    - `1.0` when within threshold
    - `0.1 ** (deviation - 1)` beyond threshold
  - Averages over selected timesteps to get per-proposal quality in `[0,1]`
  - Optional comfort multiplier from jerk-based comfort term
- Extended `_compute_quality_target(...)`:
  - Supports `score_target_type == "rfs"`
  - Requires `past` for speed scaling
- Updated scorer call site in `_shared_step(...)` to pass `past` into
  `_compute_quality_target(...)`.

### New/updated hyperparameters and CLI

- `train.py`
  - `--score_target_type`: now supports `l1 | navsim | rfs`
  - `--no_rfs_target_comfort`: disables jerk comfort multiplier for RFS targets
- `LitModel` hparams
  - `rfs_target_use_comfort: bool = True`

### Training script default

- `scripts/run_train_proposal.sbatch`
  - Default scorer target changed from `l1` to `rfs`
  - Added inline note that this can be overridden via `EXTRA_ARGS`

### Notes

- Existing standalone trajectory-side `loss_rfs` path remains intact.
- If using `--score_target_type rfs`, keep `rfs_weight=0` unless you explicitly
  want both scorer-target RFS and trajectory regularization RFS active together.
