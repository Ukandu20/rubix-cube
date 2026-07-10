# PPO Curriculum Experiment Tracker

Created: 2026-07-09

Purpose: track controlled PPO curriculum-learning experiments for the depth-1–10 Rubik's cube policy. This document records the intended config changes, the reason for each change, and the metrics that should be compared after each run.

## Current bottleneck summary

The recent depth-1–10 PPO runs reached the harder curriculum stages but stalled around depth 6.

Observed issue:

- The model can solve some depth-6 states.
- Average solution length is acceptable when it does solve.
- The main failure mode is reliability: too many episodes time out before solving.

For depth 6, the relevant metrics to watch are:

- `solve_rate`
- `timeout_rate`
- `average_solution_length`
- `median_solution_length`
- `inverse_move_rate`
- `train_solve_rate`
- `mean_episode_reward`
- `entropy`
- `approx_kl`
- `clip_fraction`

The key interpretation:

- If `average_solution_length` is below the threshold but `timeout_rate` is high, the model is not primarily inefficient. It is inconsistent.
- If `solve_rate` improves with more timesteps and more depth-6 exposure, the bottleneck is likely training exposure.
- If solve rate plateaus, changes to exploration, model capacity, or reward shaping may be needed.

## Baseline references

Use these as comparison artifacts when reviewing future runs:

- `models/artifacts/ppo/depth_1_10_onehot/v003/metrics.json`
- `models/artifacts/ppo/depth_1_10_onehot/v004/metrics.json`
- `models/artifacts/ppo/depth_1_10_onehot/v004/history.csv`

Baseline notes from `v004`:

- Reached curriculum depth 6.
- Used `2,000,000` configured training timesteps.
- Depth-6 evaluation still failed the advancement gate because solve rate was too low and timeout rate was too high.

## Experiment 001: More depth-6+ exposure with longer training

Status: planned

### Hypothesis

The current model is under-trained at depth 6 rather than fundamentally unable to learn the task. Increasing training time and increasing the sampling weight on the active hard depth should reduce timeout rate and improve solve rate.

### Planned changes

#### 1. Increase total timesteps

Change training budget from the current lower budget to:

```text
total_timesteps: 5_000_000
```

Expected effect:

- More PPO updates at the current difficult curriculum stage.
- More opportunity for depth-6 reliability to improve.

Risk:

- If the policy has already plateaued, this will increase runtime without materially improving depth-6 solve rate.

#### 2. Adjust advancement thresholds

Use a consistent success/timeout interpretation for the harder stage gate:

```yaml
success_rate: 0.70
max_timeout_rate: 0.30
```

Recommended application for this experiment:

- Apply to the depth-6 bottleneck gate.
- If applying to depths 7–10 as well, record that explicitly in the run config and compare cautiously, because this changes the meaning of validation at later stages.

Reasoning:

- `success_rate >= 0.70` allows up to roughly `0.30` failure rate.
- If `max_timeout_rate` remains stricter than that, the timeout gate can become the real advancement gate.
- Aligning `0.70` success with `0.30` timeout makes the threshold easier to interpret.

Risk:

- Threshold changes do not improve learning directly. They only change when the curriculum advances.
- If thresholds are too loose, the model can advance before it is stable.

#### 3. Adjust mixed sampling weights

Increase exposure to the current hard depth for stages 6–10 while retaining some easier-depth rehearsal.

```yaml
mixed_sampling_weights:
  6:
    1: 0.02
    2: 0.03
    3: 0.05
    4: 0.10
    5: 0.20
    6: 0.60
  7:
    1: 0.025
    2: 0.025
    3: 0.05
    4: 0.05
    5: 0.10
    6: 0.15
    7: 0.60
  8:
    1: 0.025
    2: 0.025
    3: 0.025
    4: 0.025
    5: 0.05
    6: 0.10
    7: 0.15
    8: 0.60
  9:
    1: 0.025
    2: 0.025
    3: 0.025
    4: 0.025
    5: 0.05
    6: 0.05
    7: 0.10
    8: 0.10
    9: 0.60
  10:
    1: 0.025
    2: 0.025
    3: 0.025
    4: 0.025
    5: 0.05
    6: 0.05
    7: 0.05
    8: 0.05
    9: 0.10
    10: 0.60
```

Expected effect:

- At stage 6, depth-6 samples increase from the previous `0.40` to `0.60`.
- The model receives more direct training signal on the current bottleneck.
- Easier depths remain in the mix to reduce catastrophic forgetting.

Risk:

- Harder-depth exposure may increase failed rollouts early in the stage.
- If failures are too sparse or repetitive, extra exposure may not be enough without reward-shaping or exploration changes.

### Run checklist

Before training:

- Confirm `total_timesteps` is `5_000_000`.
- Confirm depth-6 sampling weight sums to `1.0`.
- Confirm depth-7 sampling weight sums to `1.0`.
- Confirm depth-8 sampling weight sums to `1.0`.
- Confirm depth-9 sampling weight sums to `1.0`.
- Confirm depth-10 sampling weight sums to `1.0`.
- Confirm which depths use `success_rate: 0.70` and `max_timeout_rate: 0.30`.
- Save the exact config used for the run.

After training:

- Record artifact path.
- Record final curriculum depth.
- Record whether the model advanced beyond depth 6.
- Record final and best depth-6 `solve_rate`.
- Record final and best depth-6 `timeout_rate`.
- Record final and best depth-6 `average_solution_length`.
- Compare greedy Streamlit benchmark results by length 1–10.
- Compare retry-assisted Streamlit benchmark results by length 1–10.

### Result log

Fill this in after the experiment completes.

```text
Run artifact:
Started:
Completed:
Configured timesteps:
Actual timesteps:
Final curriculum depth:
Best checkpoint:
Final checkpoint:

Depth-6 best solve_rate:
Depth-6 best timeout_rate:
Depth-6 best average_solution_length:

Depth-6 final solve_rate:
Depth-6 final timeout_rate:
Depth-6 final average_solution_length:

Advanced past depth 6:
If yes, timestep of advancement:

Streamlit benchmark summary:
Notes:
```

## Decision criteria

Treat Experiment 001 as successful if:

- Depth-6 solve rate improves materially over `v004`.
- Depth-6 timeout rate drops materially.
- The model advances to depth 7 without a collapse in lower-depth performance.

Treat it as inconclusive if:

- Depth-6 improves slightly but remains below threshold.
- Metrics are noisy without a clear upward trend.

Treat it as unsuccessful if:

- Depth-6 solve rate remains near the previous plateau.
- Timeout rate remains high.
- Lower-depth performance regresses sharply.

## Next possible experiments

Only change one major lever at a time after Experiment 001.

Potential follow-up experiments:

1. Increase entropy/exploration:
   - Try `ent_coef: 0.015`.
   - Compare depth-6 solve rate, timeout rate, entropy, and greedy benchmark results.

2. Increase network capacity:
   - Try hidden layers `[512, 512]`.
   - Keep the curriculum changes from Experiment 001.
   - Use a larger timestep budget because the larger model may need more training.

3. Add anti-cycle reward shaping:
   - Penalize repeated states within an episode.
   - Penalize immediate inverse moves lightly.
   - Keep shaping small enough that solve reward remains dominant.

4. Separate reporting for greedy vs retry-assisted solving:
   - Greedy solve rate should remain the validation metric.
   - Retry-assisted solve rate should be reported as practical demo capability.
