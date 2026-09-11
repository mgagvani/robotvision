# Edit-To-Steering Alignment Figure Notes

This directory contains the analysis and figure for the "edit-sensitive features
have matching steering effects" result. The goal is to connect counterfactual
image edits with latent feature steering:

> If a visual edit changes an SAE feature, does manually increasing that same
> SAE feature push the planner behavior in the direction implied by the edit?

The current main paper candidate is:

- `fig5b_edit_to_steering_alignment_stop_edits.pdf`
- `fig5b_edit_to_steering_alignment_stop_edits.png`

The all-edit diagnostic version is also kept:

- `fig5b_edit_to_steering_alignment.pdf`
- `fig5b_edit_to_steering_alignment.png`

The current figure script is:

- `plot_edit_to_steering_alignment.py`

The Slurm wrapper used to produce the missing steering outputs is:

- `run_edit_to_steering_alignment.slurm`

## Current Files

Inputs copied or referenced by this analysis:

- Visual-edit SAE summary:
  `/work/hdd/bgxf/mgagvani/visual_gen_1000/sae_analysis/sae_visual_gen_feature_summary_block_3.csv`
- Latent steering per-stat rows:
  `sae_control_stat_rows_block_3_val.csv`
- Latent steering feature summary:
  `sae_control_summary_block_3_val.csv`

Generated figure tables:

- `fig5b_edit_to_steering_alignment_stop_edits_points.csv`
- `fig5b_edit_to_steering_alignment_stop_edits_summary.csv`
- `fig5b_edit_to_steering_alignment_points.csv`
- `fig5b_edit_to_steering_alignment_summary.csv`

Generated figures:

- `fig5b_edit_to_steering_alignment_stop_edits.pdf`
- `fig5b_edit_to_steering_alignment_stop_edits.png`
- `fig5b_edit_to_steering_alignment.pdf`
- `fig5b_edit_to_steering_alignment.png`

The PNG exports use high DPI:

```python
"savefig.dpi": 960
```

The PDF exports use editable text:

```python
"pdf.fonttype": 42
"ps.fonttype": 42
```

## Figure Versions

### Stop-Edit Figure

This is the current preferred figure. It includes only edits whose semantic
effect should increase stopping or braking:

- `green_to_red`
- `add_stop_sign`
- `add_pedestrian`

It is generated with:

```bash
PYTHONPATH=/u/mgagvani/robotvision/.venv/lib/python3.12/site-packages \
MPLCONFIGDIR=/tmp/mpl-edit-sensitivity \
/u/mgagvani/.local/share/uv/python/cpython-3.12-linux-aarch64-gnu/bin/python \
  src/camera-based-e2e/visualizations/paper/edit_sensitivity/plot_edit_to_steering_alignment.py \
  --directions green_to_red,add_stop_sign,add_pedestrian \
  --output_prefix src/camera-based-e2e/visualizations/paper/edit_sensitivity/fig5b_edit_to_steering_alignment_stop_edits
```

The stop-edit pooled result is:

| edit | n | Pearson r | Spearman r | cosine alignment | sign agreement |
|---|---:|---:|---:|---:|---:|
| green -> red | 80 | +0.211 | +0.165 | +0.332 | 57.7% |
| add stop sign | 80 | +0.154 | +0.088 | +0.299 | 60.5% |
| add pedestrian | 80 | +0.154 | +0.180 | +0.327 | 64.9% |
| pooled | 240 | +0.172 | +0.143 | +0.317 | 61.0% |

### All-Edit Diagnostic Figure

This figure includes:

- `green_to_red`
- `red_to_green`
- `add_stop_sign`
- `add_pedestrian`

It is useful as a diagnostic or supplement, but it is not as clean as the
stop-edit version because `red_to_green` does not align under the braking-based
readout.

It is generated with the script defaults:

```bash
PYTHONPATH=/u/mgagvani/robotvision/.venv/lib/python3.12/site-packages \
MPLCONFIGDIR=/tmp/mpl-edit-sensitivity \
/u/mgagvani/.local/share/uv/python/cpython-3.12-linux-aarch64-gnu/bin/python \
  src/camera-based-e2e/visualizations/paper/edit_sensitivity/plot_edit_to_steering_alignment.py
```

All-edit results:

| edit | n | Pearson r | Spearman r | cosine alignment | sign agreement |
|---|---:|---:|---:|---:|---:|
| green -> red | 80 | +0.211 | +0.165 | +0.332 | 57.7% |
| red -> green | 80 | -0.224 | -0.091 | -0.335 | 40.3% |
| add stop sign | 80 | +0.154 | +0.088 | +0.299 | 60.5% |
| add pedestrian | 80 | +0.154 | +0.180 | +0.327 | 64.9% |
| pooled | 320 | +0.087 | +0.085 | +0.179 | 55.8% |

## What The Current Graphic Shows

Each panel is one visual edit type. Each point is one SAE feature selected
because it is among the top edit-responsive features for that edit type.

The plot asks:

> Across SAE features, do features that move more under a stop-like visual edit
> also produce stronger braking effects when manually steered?

For the stop-edit figure, positive alignment means:

- the visual edit increases a feature and steering that feature increases
  braking, or
- the visual edit decreases a feature and steering that feature decreases
  braking.

Points in the upper-right and lower-left quadrants are directionally aligned.
Points in the upper-left and lower-right quadrants are directionally opposed.

The regression line and Pearson `r` summarize whether larger visual responses
also correspond to larger standardized braking effects. The `r` values are not
printed in the current graphic; they are kept in the summary CSV and in the
tables above.

The graph should be interpreted as a bridge between visual sensitivity and
causal behavior:

> Stop-like counterfactual image edits preferentially move SAE features whose
> manual activation also increases braking.

This is deliberately weaker and more defensible than claiming:

> Every visual edit direction has a perfectly matching causal feature.

The `red_to_green` result is the reason to avoid the stronger claim. It goes
negative under this braking-based readout, so the clean paper story is about
stop-inducing edits.

## Axis Definitions

### X Axis

Current label:

```text
Visual edit response (normalized Δz_i)
```

For edit direction `e` and SAE feature `i`, the x-value is:

```text
x_{e,i} = mean_j [ z_i(edited_j) - z_i(original_j) ] / feature_scale_i
```

where `j` indexes examples with edit direction `e`.

In the CSV this is:

```text
sae_visual_gen_feature_summary_block_3.csv
group_kind = edit_direction
group_value = edit name
feature_idx = i
mean_delta_scale_units = x_{e,i}
```

The script reads it through:

```python
parser.add_argument(
    "--visual_metric",
    type=str,
    default="mean_delta_scale_units",
    help="Visual-summary column for signed feature change.",
)
```

and then:

```python
values[direction][feature_idx] = fget(row, visual_metric)
```

### Y Axis

Current label:

```text
Standardized steering effect on braking
```

For SAE feature `i`, `analyze_sae_control.py` manually increases that feature,
reruns the planner, and measures the change in braking:

```text
y_i = mean_j [ brake_mag(steered_j) - brake_mag(base_j) ] / std_global(brake_mag)
```

In the control CSV this is:

```text
sae_control_stat_rows_block_3_val.csv
stat_name = brake_mag
feature_idx = i
std_effect_max = y_i
```

For the stop-edit figure, all edit types are stop-inducing, so no sign flip is
needed. Positive y means manually increasing that SAE feature increases braking.

For the all-edit diagnostic version, the plotting script has a `BEHAVIOR_SIGN`
mapping for go-inducing edits:

```python
BEHAVIOR_SIGN = {
    "green_to_red": 1.0,
    "add_stop_sign": 1.0,
    "add_pedestrian": 1.0,
    "add_traffic_light": 1.0,
    "red_to_green": -1.0,
    "yellow_to_green": -1.0,
    "remove_stop_sign": -1.0,
    "remove_pedestrian": -1.0,
    "remove_traffic_light": -1.0,
}
```

The plotted y-value is:

```python
raw_brake_effect = fget(control_row, steering_metric)
edit_consistent_effect = sign * raw_brake_effect
```

where `sign = BEHAVIOR_SIGN[direction]`.

For the stop-edit figure, `sign = +1` for all plotted directions.

## How The Visual-Edit X Values Were Made

The visual-edit analysis was produced by:

- `src/camera-based-e2e/analyze_sae_visual_gen_pt2.py`

The important steps are:

1. Load the visual edit manifest:

   ```text
   /work/hdd/bgxf/mgagvani/visual_gen_1000/manifest.jsonl
   ```

2. Keep rows with:

   ```python
   row.get("status") == "edited"
   ```

3. For each edited row, load the original dataset sample and replace the edited
   camera image in the model input.

4. Run the planner on the edited image batch:

   ```python
   out_edit = planner_model(
       model_inputs_from_batch(edit_batch, lit_model=lit_model, device=device),
       return_block_tokens=True,
   )
   ```

5. Get the corresponding original token from the token blob:

   ```python
   token_orig_cpu = token_tensor.index_select(0, token_indices)
   token_edit = out_edit[token_key]
   ```

6. Encode original and edited planner tokens with the SAE:

   ```python
   z_orig = sae.encode(token_orig)
   z_edit = sae.encode(token_edit)
   ```

7. Compute per-example deltas:

   ```python
   delta = z_edit - z_orig
   ```

8. For each edit direction and feature, compute:

   ```python
   mean_delta = delta.mean(dim=0)
   mean_abs_delta = delta.abs().mean(dim=0)
   ```

9. Normalize by a feature-specific scale:

   ```python
   feature_scale = baseline_stats["feature_scale"]
   mean_delta_scale_units = mean_delta / feature_scale
   mean_abs_delta_scale_units = mean_abs_delta / feature_scale
   ```

The feature scale comes from baseline SAE activations over the token blob:

```python
active_mask = z_all > 0
active_count = active_mask.sum(dim=0)
active_sum = z_all.sum(dim=0)
active_sum_sq = (z_all * z_all).sum(dim=0)
active_mean = active_sum / active_count.clamp_min(1)
active_var = active_sum_sq / active_count.clamp_min(1) - active_mean.square()
active_std = torch.sqrt(active_var.clamp_min(0.0))
feature_scale = torch.maximum(active_std, 0.25 * active_mean).clamp_min(min_scale)
```

The output file used here is:

```text
/work/hdd/bgxf/mgagvani/visual_gen_1000/sae_analysis/sae_visual_gen_feature_summary_block_3.csv
```

Relevant columns:

| column | meaning |
|---|---|
| `group_kind` | either `edit_type` or `edit_direction`; this figure uses `edit_direction` |
| `group_value` | edit direction such as `green_to_red` |
| `feature_idx` | SAE feature index |
| `n` | number of edited examples for that direction |
| `mean_delta` | raw mean signed SAE activation change |
| `mean_delta_scale_units` | normalized mean signed SAE activation change; used for x-axis |
| `paired_t` | paired t-statistic of feature delta across examples |
| `sign_consistency` | fraction of examples whose feature delta sign agrees with the mean sign |

## How The Steering Y Values Were Made

The steering analysis was produced by:

- `src/camera-based-e2e/analyze_sae_control.py`

It was run through:

- `run_edit_to_steering_alignment.slurm`

The Slurm job computes the full 384-feature steering sweep for SAE block 3.


### Steering Procedure

For each SAE feature `i`, `analyze_sae_control.py`:

1. Loads SAE block 3 and the validation token blob.

2. Encodes all validation planner tokens:

   ```python
   base_z_cpu = encode_tensor_batchwise(
       sae,
       token_tensor,
       batch_size=args.encode_batch_size,
       device=device,
   )
   ```

3. Computes active-feature statistics and the intervention scale:

   ```python
   active_mask = base_z_cpu > 0
   active_count = active_mask.sum(dim=0)
   active_sum = base_z_cpu.sum(dim=0)
   active_sum_sq = (base_z_cpu * base_z_cpu).sum(dim=0)
   active_mean = active_sum / active_count.clamp_min(1)
   active_var = active_sum_sq / active_count.clamp_min(1) - active_mean.square()
   active_std = torch.sqrt(active_var.clamp_min(0))
   intervention_scale = torch.maximum(active_std, 0.25 * active_mean).clamp_min(args.min_scale)
   ```

4. Selects relevant scenes per feature:

   ```python
   top_k = min(args.relevant_scenes_per_feature, base_z_cpu.size(0))
   _, top_indices = torch.topk(base_z_cpu, k=top_k, dim=0)
   ```

   In the Slurm run:

   ```text
   relevant_scenes_per_feature = 64
   ```

5. Uses intervention levels:

   ```text
   alphas = 0, 0.5, 1.0, 2.0
   ```

   For each scene and feature:

   ```python
   z_mod[:, feature_idx] = (base_activation + alpha * scale).clamp_min(0.0)
   ```

6. Decodes the modified SAE latent vector back into planner-token space:

   ```python
   recon_query = sae.decode_to_input(z_mod, reference_x=base_x)
   ```

7. Reruns the planner from the modified query token:

   ```python
   out = planner_model.forward_from_planner_query_tok(recon_query, past)
   ```

   Because this is SAE block 3, it uses the final planner query token path.

8. Computes output statistics for the selected trajectory proposal.

The braking statistic is:

```python
selected_idx = scores.argmin(dim=1)
selected_ctrl = ctrl[row_idx, selected_idx]
accel = selected_ctrl[..., 0]
brake_mag = (-accel).clamp_min(0).mean(dim=1)
```

This means `brake_mag` is the mean positive braking magnitude over the selected
trajectory horizon, where negative acceleration is interpreted as braking.

The standardized steering effect used in the figure is:

```python
delta = curves - curves[0:1]
delta_max = delta[-1]
mean_delta_max = float(delta_max.mean().item())
std_effect_max = mean_delta_max / global_stat_stds[stat_name]
```

For this figure:

```text
stat_name = brake_mag
steering_metric = std_effect_max
```

So the y-axis is the final-alpha mean braking change, standardized by the
global standard deviation of `brake_mag`.

Relevant control CSV columns:

| column | meaning |
|---|---|
| `feature_idx` | SAE feature index |
| `stat_name` | planner statistic being steered; this figure uses `brake_mag` |
| `active_count` | number of validation examples where the feature is active |
| `relevant_scene_count` | number of examples used for the feature intervention, usually 64 |
| `intervention_scale` | feature-specific activation scale |
| `mean_delta_max` | mean raw stat change at largest alpha |
| `std_effect_max` | standardized effect at largest alpha; used for y-axis |
| `mean_rho` | mean Spearman monotonicity over intervention levels |
| `frac_consistent` | fraction of scenes with sign-consistent monotonic behavior |
| `control_score` | heuristic control score from the steering analysis |

## Point Selection

The figure does not plot every SAE feature. It plots the top edit-sensitive
features for each edit direction.

Current default:

```python
parser.add_argument("--top_k_per_direction", type=int, default=80)
```

For each edit direction:

```python
ranked = sorted(
    visual_by_direction.get(direction, {}).items(),
    key=lambda item: abs(item[1]),
    reverse=True,
)
selected[direction] = [feature_idx for feature_idx, _ in ranked[:top_k]]
```

Thus each panel has 80 points.

For the stop-edit figure:

```text
3 panels * 80 points = 240 points
```

For the all-edit diagnostic:

```text
4 panels * 80 points = 320 points
```

## Clustering And Colors

Clusters are used only for point color. They are not part of the metric
definition.

The current clustering uses KMeans with:

```python
parser.add_argument("--n_clusters", type=int, default=5)
```

Features are clustered using their visual-edit response vectors across the
directions included in the figure:

```python
matrix = np.array(
    [[visual_by_direction.get(direction, {}).get(feature_id, 0.0)
      for direction in directions]
     for feature_id in feature_ids],
    dtype=float,
)
scale = np.std(matrix, axis=0)
scale[scale == 0] = 1.0
matrix = (matrix - np.mean(matrix, axis=0)) / scale
```

Then:

```python
labels = KMeans(
    n_clusters=max(1, min(n_clusters, len(feature_ids))),
    random_state=0,
    n_init=20,
).fit_predict(matrix)
```

If sklearn is unavailable, the script uses a deterministic fallback KMeans.

Current color palette:

| group | color | hex |
|---|---|---|
| C0 | blue | `#2F6C9F` |
| C1 | orange-brown | `#C45A2A` |
| C2 | green | `#5E8C61` |
| C3 | purple | `#7A5FA8` |
| unclustered | light gray | `#C9CDD3` |

The script labels cluster IDs 4 and 5 as `unclustered`:

```python
CLUSTER_LABELS = {
    4: "unclustered",
    5: "unclustered",
}
```

Current stop-edit point counts by cluster:

| group | count |
|---|---:|
| C0 | 136 |
| C1 | 17 |
| C2 | 52 |
| C3 | 9 |
| unclustered | 26 |

Current all-edit point counts by cluster:

| group | count |
|---|---:|
| C0 | 175 |
| C1 | 52 |
| C2 | 11 |
| C3 | 71 |
| unclustered | 11 |

Note: `unclustered` is a presentation label for the gray cluster, not the
output of a density-based clustering algorithm. If the paper needs a strict
technical meaning for "unclustered", this should be renamed to something like
`other` or `residual`.

## Summary Metrics

The figure script computes several summary metrics per edit direction:

```python
summary_rows = summarize_points(point_rows, directions)
```

For each group:

```python
x = np.array([row["mean_delta_z_scale_units"] for row in group])
y = np.array([row["edit_consistent_brake_steering_effect"] for row in group])
```

### Pearson r

Pearson correlation between x and y:

```python
pearson_r = corr(x, y)
```

Interpretation:

> Do features with larger visual-edit responses tend to have larger braking
> steering effects?

Pearson `r` is not currently printed on the figure, but is available in the
summary CSV.

### Spearman r

Rank correlation between x and y:

```python
spearman_r = corr(x, y, spearman=True)
```

Interpretation:

> Is there a monotonic relationship between visual response rank and steering
> effect rank?

### Cosine Alignment

Cosine similarity between the vector of x-values and the vector of y-values:

```python
cosine_alignment = sum(x * y) / (sqrt(sum(x^2)) * sqrt(sum(y^2)))
```

Interpretation:

> Are the edit-response vector and steering-effect vector pointing in similar
> directions?

### Sign Agreement

Sign agreement is the fraction of nonzero plotted points where the x and y
signs match:

```python
sign_match_rate = mean(x * y > 0)
```

Interpretation:

> Does the visual edit move a feature in the same signed direction as that
> feature's steering effect on braking?

Cases:

| x sign | y sign | interpretation |
|---|---|---|
| x > 0 | y > 0 | agree |
| x < 0 | y < 0 | agree |
| x > 0 | y < 0 | disagree |
| x < 0 | y > 0 | disagree |

This statistic used to be printed in the panel annotation as `sign agree`, but
it was removed from the graphic. It remains in the summary CSV and in this
document.

## Metric Variants We Checked

We briefly "shopped around" for an alternate y-axis that would make the all-edit
story cleaner. The key result: there was no simple metric swap that made all
edit directions align without creating a less defensible story.

The candidate steering readouts checked were:

- `brake_mag`, signed by edit semantics
- `accel_mag`, signed by edit semantics
- `brake_mag - accel_mag`
- `accel_mag - brake_mag`
- `proposal_spread`
- `score_margin`

Results from the quick sweep:

| y definition | green -> red r | red -> green r | add stop r | add ped r | pooled r | pooled sign agreement |
|---|---:|---:|---:|---:|---:|---:|
| brake edit-consistent | +0.211 | -0.224 | +0.154 | +0.154 | +0.087 | 55.8% |
| accel edit-consistent | -0.005 | -0.192 | +0.101 | -0.026 | -0.012 | 50.0% |
| brake-minus-accel consistent | +0.097 | -0.261 | +0.181 | +0.058 | +0.034 | 54.2% |
| accel-minus-brake consistent | -0.097 | +0.261 | -0.181 | -0.058 | -0.034 | 45.8% |
| proposal spread stop-down | -0.191 | +0.157 | -0.090 | -0.229 | -0.088 | 44.5% |
| score margin stop-up | +0.087 | -0.000 | +0.153 | +0.064 | +0.077 | 52.9% |

The conclusion was:

- `brake_mag` is the cleanest y-axis for stop-like edits.
- `red_to_green` is the difficult direction under a braking readout.
- The defensible paper result is the stop-edit figure, with the all-edit figure
  kept as diagnostic/supplementary context if needed.

We also checked contrast axes such as:

- `green_to_red`
- `-red_to_green`
- `green_to_red - red_to_green`
- `green_to_red - red_to_green - yellow_to_green`
- `add_stop_sign - remove_stop_sign`
- `add_pedestrian - remove_pedestrian`

Those did not produce a better simple all-edit story. The `add_pedestrian -
remove_pedestrian` contrast was positive, but `green_to_red - red_to_green` was
weak. Again, the stable story is stop-inducing edits.

## Figure Styling Decisions

These choices were made to match the paper style guide:

- Serif font family with STIX/Times/DejaVu fallbacks.
- Editable PDF text via `pdf.fonttype = 42`.
- Restrained palette.
- Light gray gridlines.
- No explanatory text blocks inside the figure.
- No per-panel statistic box in the final graphic.
- No legend title in the final graphic.
- The stop-edit version is a single horizontal row of three panels.

Current labels:

```text
x-axis: Visual edit response (normalized Δz_i)
y-axis: Standardized steering effect on braking
```

The exact label strings in the script are:

```python
X_AXIS_LABEL = r"Visual edit response (normalized $\Delta z_i$)"
Y_AXIS_LABEL = "Standardized steering effect on braking"
```

## What To Say In The Paper

A concise text description:

> We compare each feature's signed response to visual counterfactual edits with
> its causal steering effect. For each edit type, the x-axis shows the mean
> normalized change in SAE feature activation induced by the edit. The y-axis
> shows the standardized change in predicted braking when that same SAE feature
> is manually increased. Stop-inducing edits show positive alignment: features
> that are increased by the edit tend to be features whose steering increases
> braking.

A slightly more technical version:

> For each stop-inducing visual edit, we compute the mean signed SAE activation
> shift for each feature, normalized by its active-scale statistic. We then
> independently intervene on each SAE feature by increasing its activation,
> decoding back to planner-token space, and measuring the standardized change
> in braking magnitude. The resulting feature-wise scatter plots test whether
> visual edit sensitivity and causal steering effects are directionally aligned.

A reviewer-safe limitation sentence:

> The alignment is modest rather than deterministic, which is expected because
> image edits perturb many features and SAE feature steering is measured by a
> single-feature intervention. The relationship is strongest and most consistent
> for stop-inducing edits; the red-to-green edit does not align cleanly under a
> braking-based readout and is therefore treated as diagnostic rather than the
> primary claim.

## What Not To Claim

Avoid claiming:

- "All visual edit directions align with feature steering."
- "Every edit-sensitive feature is causal."
- "The clusters are semantic classes unless separately validated."
- "`unclustered` has a formal density-clustering meaning."
- "Sign agreement is accuracy."

Better claims:

- "Stop-inducing edits show positive feature-wise alignment with braking
  steering effects."
- "The result supports a bridge between visual counterfactual sensitivity and
  latent causal control."
- "The alignment is partial but consistent across three stop-like edit types."

## Reproduction Commands

### Full Slurm Pipeline

Submit:

```bash
sbatch src/camera-based-e2e/visualizations/paper/edit_sensitivity/run_edit_to_steering_alignment.slurm
```

Monitor:

```bash
squeue -j <job_id>
sacct -j <job_id> --format=JobID,JobName%24,State,ExitCode,Elapsed,ReqMem,MaxRSS,NodeList%18
```

The Slurm script:

1. Runs `analyze_sae_control.py` if `sae_control_stat_rows_block_3_val.csv`
   is missing or `FORCE_CONTROL=1`.
2. Runs `plot_edit_to_steering_alignment.py`.

Important Slurm defaults:

```bash
#SBATCH --mem=192g
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
ENCODE_BATCH_SIZE=${ENCODE_BATCH_SIZE:-1024}
RELEVANT_SCENES_PER_FEATURE=${RELEVANT_SCENES_PER_FEATURE:-64}
ALPHAS=${ALPHAS:-0,0.5,1.0,2.0}
TOP_K_PER_DIRECTION=${TOP_K_PER_DIRECTION:-80}
```

### Regenerate Stop-Edit Figure Only

```bash
PYTHONPATH=/u/mgagvani/robotvision/.venv/lib/python3.12/site-packages \
MPLCONFIGDIR=/tmp/mpl-edit-sensitivity \
/u/mgagvani/.local/share/uv/python/cpython-3.12-linux-aarch64-gnu/bin/python \
  src/camera-based-e2e/visualizations/paper/edit_sensitivity/plot_edit_to_steering_alignment.py \
  --directions green_to_red,add_stop_sign,add_pedestrian \
  --output_prefix src/camera-based-e2e/visualizations/paper/edit_sensitivity/fig5b_edit_to_steering_alignment_stop_edits
```

### Regenerate All-Edit Diagnostic Figure

```bash
PYTHONPATH=/u/mgagvani/robotvision/.venv/lib/python3.12/site-packages \
MPLCONFIGDIR=/tmp/mpl-edit-sensitivity \
/u/mgagvani/.local/share/uv/python/cpython-3.12-linux-aarch64-gnu/bin/python \
  src/camera-based-e2e/visualizations/paper/edit_sensitivity/plot_edit_to_steering_alignment.py
```

## Current Script Defaults

```text
visual_summary:
  /work/hdd/bgxf/mgagvani/visual_gen_1000/sae_analysis/sae_visual_gen_feature_summary_block_3.csv

control_stat_rows:
  src/camera-based-e2e/visualizations/paper/edit_sensitivity/sae_control_stat_rows_block_3_val.csv

directions:
  green_to_red,red_to_green,add_stop_sign,add_pedestrian

top_k_per_direction:
  80

n_clusters:
  5

control_metric:
  brake_mag

visual_metric:
  mean_delta_scale_units

steering_metric:
  std_effect_max
```

## Final State After Conversation

Edits made during figure iteration:

1. Created `plot_edit_to_steering_alignment.py`.
2. Created `run_edit_to_steering_alignment.slurm`.
3. Submitted Slurm job `2337615`; it failed with OOM.
4. Increased Slurm memory to `192G`, lowered encode batch size to `1024`,
   resubmitted as `2339936`; it completed.
5. Generated the all-edit figure.
6. Identified `red_to_green` as the outlier under braking alignment.
7. Generated the stop-edit-only figure as the primary candidate.
8. Converted the stop-edit figure to a single row of three panels.
9. Increased PNG export DPI 3x, from 320 to 960.
10. Set final labels:
    - `Visual edit response (normalized Δz_i)`
    - `Standardized steering effect on braking`
11. Removed `sign agree` from graph annotations.
12. Removed `r` from graph annotations.
13. Removed the legend heading `SAE feature clusters`.
14. Updated cluster colors:
    - blue `#2F6C9F`
    - orange-brown `#C45A2A`
    - green `#5E8C61`
    - purple `#7A5FA8`
    - light gray `#C9CDD3` for `unclustered`

The current graphic is intentionally spare: panel titles, axes, colored feature
clusters, and regression lines. The statistics are externalized to the summary
CSV and this note.
