# Rubik's Cube Reinforcement Learning

This project trains and evaluates agents that solve scrambled 3×3 Rubik's Cube
states. It provides a custom PyTorch PPO trainer, a parallel Stable-Baselines3
PPO trainer, optional supervised actor warm starts, curriculum learning,
baseline solvers, and a local Streamlit showcase.

The project is a research system rather than a complete general-purpose cube
solver. Current experiments learn useful policies through scramble depth 7, but
the deepest curriculum level is not yet mastered reliably.

## What the project provides

- Correct cube state, move, and notation primitives.
- Generated labeled states stored as CSV or Parquet by scramble depth.
- A Gymnasium environment for dataset-driven training.
- Random, inverse-scramble, and breadth-first-search baselines.
- Custom PyTorch PPO and Stable-Baselines3 PPO training workflows.
- Supervised policy training and an SB3 actor warm-start workflow.
- Curriculum gates, mixed-depth sampling, checkpoint selection, and detailed
  behavior metrics.
- A Streamlit interface for animated solve attempts and local benchmarks.

## Architecture

```text
streamlit_app.py and scripts/       User-facing commands
              │
              ▼
src/showcase and src/agents/        Solving, training, evaluation, checkpoints
              │
              ├──────────────► src/curriculum/  Depth sampling and progression
              │
              ▼
src/cube/gym_environment.py         Gymnasium dataset adapter
src/cube/environment.py             Lightweight baseline/search adapter
              │
              ▼
src/cube/episode.py                 Shared transition and reward rules
src/cube/state.py, moves.py,        Framework-independent cube domain
notation.py, encoding.py
```

The important directories are:

```text
config/          Versioned curriculum definitions
data/            Local generated datasets; large contents are gitignored
models/          Local checkpoints and metrics; artifacts are gitignored
notebooks/       Exploratory analysis
reports/         Experiment reports and limitations
scripts/         Thin command-line entry points
src/agents/      Agents, trainers, metrics, and artifact persistence
src/cube/        Cube domain, observation encoding, and environment adapters
src/curriculum/  Curriculum configuration and state sampling
src/data/        Training-data generation
src/models/      Standalone supervised policy workflow
src/showcase/    Checkpoint inference, solve service, and visualization
tests/           Automated unit and integration tests
```

## Installation

Python 3.11 or 3.12 is recommended; Python 3.14 is also supported.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebooks]"
```

`pyproject.toml` defines supported dependency ranges. To reproduce the known-good
local environment as closely as possible, also apply the constraints file:

```powershell
python -m pip install -e ".[dev,notebooks]" -c constraints.txt
```

Editable installation makes `cube`, `agents`, `curriculum`, and the other
packages importable without modifying `sys.path`.

## Quality checks

Run the same checks used by continuous integration:

```powershell
ruff check .
ruff format --check .
mypy
python -m pytest --cov --cov-report=term-missing
```

For a faster local regression run:

```powershell
python -m pytest -q
```

## Data workflow

Generate labeled training states:

```powershell
rubix-generate-data --help
```

Or run the module directly:

```powershell
python -m scripts.generate_training_data --help
```

Generated files belong under `data/processed/training`. The PPO workflows use
the Parquet depth shards under `data/processed/training/parquet` by default.
Large data files are intentionally not committed.

## Training

Train the custom PPO implementation:

```powershell
rubix-train-ppo --help
```

Train the Stable-Baselines3 implementation:

```powershell
rubix-train-sb3 --help
```

Train a standalone supervised policy:

```powershell
rubix-train-supervised --help
```

An SB3 run can initialize its actor from supervised first-solution labels. For
example, the depth-1–7 frontier workflow is configured through
`scripts/train_sb3_ppo_agent.py`; use `--help` to see the current options.
Scratch initialization remains the default, and warm start cannot be combined
with checkpoint resume.

See [reports/sb3_parallel_trainer.md](reports/sb3_parallel_trainer.md) for
checkpoint semantics, controlled comparisons, and trainer differences.

## Evaluation

Evaluate a PPO checkpoint or the baseline agents with:

```powershell
rubix-evaluate-ppo --help
rubix-evaluate-baselines --help
```

Metrics include solve and timeout rates, solution length, extra moves, inverse
move behavior, and action distributions. Curriculum checkpoint selection favors
the deepest reached level before comparing solve and timeout rates.

## Streamlit showcase

```powershell
python -m streamlit run streamlit_app.py
```

The app discovers trusted local custom-PPO `.pt` and SB3 `.zip` checkpoints
under `models/artifacts`. It supports seeded scrambles, greedy solving,
reproducible stochastic retries, animation, per-depth benchmarks, and CSV
exports. Checkpoints and generated run history are not uploaded anywhere.

## Artifact layout

Runs use trainer-specific, versioned directories:

```text
models/artifacts/
  ppo/<experiment>/v001/
  sb3_ppo/<experiment>/v001/
  supervised/
```

A run normally records its configuration, selected and final checkpoints,
metrics, curriculum progress, and evaluation history. Warm-start runs also
record split manifests, checkpoint hashes, and transition diagnostics. These
large artifacts are gitignored and must be copied or regenerated separately.

## Interpretation and limitations

- A reported depth is the generated **scramble length**, not necessarily the
  optimal distance from the solved cube. Move cancellation can produce a state
  with a shorter optimal solution.
- Reaching a curriculum depth means the previous gate was passed. It does not
  mean the newly reached depth is mastered.
- A high solve rate can hide repeated-turn shortcuts or action collapse. Use
  first-move and action-distribution diagnostics alongside solve rate.
- Most historical experiments use a single seed. Trainer comparisons require
  repeated, controlled multi-seed runs before making strong conclusions.
- Local data and model artifacts are not stored in Git, so source checkout alone
  does not reproduce historical experiments.

More detail is available in [reports/ppo_limitations.md](reports/ppo_limitations.md)
and [reports/ppo_curriculum_experiment_tracker.md](reports/ppo_curriculum_experiment_tracker.md).
