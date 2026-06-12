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
