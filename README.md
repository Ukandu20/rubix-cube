# Project Title

Briefly describe the purpose of this project.

## Project Structure

```text
config/                  Configuration files
data/
  raw/                   Original source data
  external/              Third-party reference data
  processed/             Cleaned or transformed data
models/
  saved_models/          Trained model files
  artifacts/             Model metadata, encoders, and related outputs
  logs/                  Training and evaluation logs
notebooks/               Exploratory notebooks
reports/
  figures/               Generated charts and visual outputs
skills/                  Project-specific reusable instructions
src/
  data/                  Data loading and preparation code
  features/              Feature engineering code
  models/                Training and inference code
  utils/                 Shared utilities
  visualization/         Plotting and reporting code
tests/                   Automated tests
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Workflow

1. Place original inputs in `data/raw`.
2. Keep reusable code in `src`.
3. Store generated datasets in `data/processed`.
4. Save trained models and related outputs under `models`.
5. Put generated charts and report assets in `reports/figures`.

## Streamlit model showcase

The local showcase generates seeded scramble sequences with lengths 1–10,
animates PPO solve attempts, and runs per-length benchmarks without loading the
large processed training datasets.

```powershell
python -m streamlit run streamlit_app.py
```

The app discovers trusted local custom-PPO `.pt` and Stable-Baselines3 `.zip`
checkpoints under `models/artifacts`.
Artifacts remain gitignored and must exist on the machine running the app.
Checkpoint curriculum metadata is displayed in the sidebar; scramble lengths
beyond the checkpoint's evaluated depths are marked as experimental.

The app reports **scramble length**, not guaranteed optimal distance. A generated
sequence can occasionally produce a state whose shortest solution is shorter
than the sequence itself.

The solve demo:

- supports exact, uniform-range, and curriculum-weighted sampling;
- runs a greedy attempt first and reproducible stochastic retries afterward;
- enforces move, attempt, and cooperative wall-clock limits;
- separates model inference and solver runtime from animation time; and
- keeps downloadable run history only in the current Streamlit session.

The benchmark compares greedy and retry-assisted PPO solve rates with the known
inverse-scramble oracle. Its summary and detailed results can be downloaded as
CSV files.

Use `python -m streamlit` rather than the global `streamlit` command so the app
runs with the same Python interpreter where the project dependencies, including
PyTorch, were installed.

## PPO trainers

The original custom PyTorch trainer remains available:

```powershell
python scripts/train_ppo_agent.py
```

Stable-Baselines3 is available as a parallel trainer with a separate artifact
namespace:

```powershell
python scripts/train_sb3_ppo_agent.py
```

See [reports/sb3_parallel_trainer.md](reports/sb3_parallel_trainer.md) for
checkpoint semantics, resume instructions, controlled-comparison commands, and
the differences that prevent bit-for-bit equivalence between trainers.
