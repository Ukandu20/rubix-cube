# Stable-Baselines3 Parallel PPO Trainer

## Experiment boundary

Stable-Baselines3 (SB3) is an additional trainer, not a replacement for the
custom PyTorch PPO implementation. The custom entry point, checkpoint format,
evaluation functions, and existing artifacts remain supported.

An SB3 comparison is accepted only when both runs use the same:

- dataset and curriculum configuration;
- reward and episode-limit behavior;
- one-hot observations;
- random seed and requested timestep budget;
- PPO hyperparameters listed by `scripts/compare_ppo_trainers.py`;
- deterministic project evaluator and evaluation episode count.

SB3 implementation details that cannot be made identical must be reported:

- SB3 owns rollout storage, timeout bootstrapping, advantage calculation, and
  optimizer scheduling;
- SB3 clips gradients across the policy module, while the custom trainer clips
  actor and critic towers separately;
- random-number consumption differs, so equal seeds do not imply identical
  trajectories;
- vectorized SB3 runs collect `n_envs` transitions per environment step.

## Training

Install the pinned dependencies, then run:

```powershell
python scripts/train_sb3_ppo_agent.py `
  --curriculum-config config/curriculum_config_depth_1_10.yaml `
  --total-timesteps 500000 `
  --eval-frequency 10000 `
  --eval-episodes 100
```

SB3 artifacts are written below `models/artifacts/sb3_ppo` by default, keeping
them separate from `models/artifacts/ppo`. Each SB3 `.zip` has a neighboring
`.metadata.json` file used by checkpoint discovery without loading the model.

The checkpoint set mirrors the custom trainer:

- `best_model_depth_N.zip` — best solve/timeout result at one curriculum depth;
- `best_model.zip` — curriculum-lexicographic best;
- `latest_passed_gate.zip` — latest model that passed a curriculum gate;
- `final_model.zip` — policy after the requested training budget.

To resume in place, select the same output directory and its final checkpoint.
`--total-timesteps` is the number of additional transitions:

```powershell
python scripts/train_sb3_ppo_agent.py `
  --output-dir models/artifacts/sb3_ppo/depth_1_10_onehot/v001 `
  --resume-from models/artifacts/sb3_ppo/depth_1_10_onehot/v001/final_model.zip `
  --curriculum-config config/curriculum_config_depth_1_10.yaml `
  --total-timesteps 100000
```

Resume restores the current curriculum depth, advancement events, evaluation
history, timestep counter, and checkpoint ranking state.

## Supervised actor warm start

The SB3 trainer can pretrain its independent actor MLP from labeled cube states
before collecting PPO rollouts. The critic remains at its original random
initialization, the supervised optimizer is discarded, and PPO receives its
full configured timestep budget with a fresh optimizer.

```powershell
python scripts/train_sb3_ppo_agent.py `
  --curriculum-config config/curriculum_config_depth_1_10.yaml `
  --supervised-warm-start `
  --pretrain-depth-mode frontier `
  --pretrain-max-depth 6 `
  --pretrain-depth-sampling balanced `
  --pretrain-sample-per-depth 100000 `
  --pretrain-epochs 10
```

`mastered` excludes the declared maximum depth, `frontier` includes it,
`full-curriculum` selects the configured range, and `custom` requires explicit
minimum and maximum depths. The test partition is removed from every PPO reset;
validation states remain available to PPO after supervised model selection.

Warm runs save best/last `.pt` actor checkpoints, a deterministic split
manifest, supervised metrics, transition diagnostics, and an experiment
summary beside the normal SB3 `.zip` checkpoints. All PPO sidecars retain the
warm-start checkpoint hash and data provenance. Warm start cannot be combined
with `--resume-from`, because applying behavior cloning to a resumed actor is a
different intervention experiment.

## Controlled comparison

The current custom CLI defaults differ from the SB3 module defaults. Pass every
matched value explicitly. For example:

```powershell
$common = @(
  "--curriculum-config", "config/curriculum_config_depth_1_3.yaml",
  "--total-timesteps", "100000",
  "--eval-frequency", "10000",
  "--eval-episodes", "100",
  "--seed", "42",
  "--learning-rate", "0.0003",
  "--gamma", "0.95",
  "--gae-lambda", "0.95",
  "--clip-range", "0.2",
  "--n-epochs", "10",
  "--n-steps", "2048",
  "--batch-size", "64",
  "--ent-coef", "0.01",
  "--vf-coef", "0.5",
  "--max-grad-norm", "0.5",
  "--target-kl", "0.03"
)
python scripts/train_ppo_agent.py @common --output-dir models/comparisons/custom
python scripts/train_sb3_ppo_agent.py @common --output-dir models/comparisons/sb3
python scripts/compare_ppo_trainers.py `
  models/comparisons/custom `
  models/comparisons/sb3 `
  --output reports/ppo_sb3_comparison.md
```

The comparison command exits with status 2 if seed, budget, curriculum, or
declared PPO settings differ. It reports throughput, final curriculum depth,
solve and timeout rates, solution length, inverse-move rate, and action entropy.

## Acceptance criteria

The parallel trainer is ready for experimental use when:

1. custom PPO regression tests remain green;
2. one-hot observations contain exactly one active color per sticker;
3. curriculum advancement is synchronized across every SB3 environment;
4. evaluation and checkpoint selection use project metrics;
5. resume retains curriculum and selection history;
6. both checkpoint formats load in the showcase;
7. a comparison report passes all configuration-parity checks.

SB3 should become the default only after a meaningful matched-budget run shows
an operational or learning advantage. The custom trainer is not removed by
this experiment.
