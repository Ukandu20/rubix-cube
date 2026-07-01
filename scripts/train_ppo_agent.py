"""Train the custom PyTorch PPO cube agent."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents.ppo_agent import (  # noqa: E402
    DEFAULT_EVAL_EPISODES,
    DEFAULT_EVAL_FREQUENCY,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TOTAL_TIMESTEPS,
    NetworkConfig,
    OBSERVATION_SIZE,
    ONE_HOT_OBSERVATION_SIZE,
    PPOConfig,
    train_ppo,
)
from cube.gym_environment import (  # noqa: E402
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_TRAINING_DATA_DIR,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train custom PPO cube agent.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_TRAINING_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--version", default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reuse the latest auto-versioned output directory for this experiment.",
    )
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--eval-frequency", type=int, default=DEFAULT_EVAL_FREQUENCY)
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help=(
            "Maximum episodes per depth for deterministic evaluation; "
            "small exhaustive datasets are evaluated once per unique state."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--observation-encoding",
        choices=("one_hot", "normalized"),
        default="one_hot",
        help="PPO model input encoding. one_hot uses 324 inputs; normalized uses 54.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument(
        "--skip-dataset-validation",
        action="store_true",
        help="Skip expensive dataset validation when using trusted generated files.",
    )
    args = parser.parse_args()

    config = PPOConfig(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        n_epochs=args.n_epochs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
    )
    network_config = NetworkConfig(
        input_dim=(
            ONE_HOT_OBSERVATION_SIZE
            if args.observation_encoding == "one_hot"
            else OBSERVATION_SIZE
        ),
        observation_encoding=args.observation_encoding,
    )
    output_dir, output_metadata = resolve_training_output_dir(
        output_dir=args.output_dir,
        output_root=args.output_root,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        observation_encoding=args.observation_encoding,
        version=args.version,
        overwrite=args.overwrite,
    )
    result = train_ppo(
        total_timesteps=args.total_timesteps,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        max_episode_steps=args.max_episode_steps,
        data_dir=args.data_dir,
        output_dir=output_dir,
        seed=args.seed,
        device=args.device,
        eval_frequency=args.eval_frequency,
        eval_episodes=args.eval_episodes,
        config=config,
        network_config=network_config,
        output_metadata=output_metadata,
        validate_dataset=not args.skip_dataset_validation,
    )
    print(f"Saved PPO artifacts to {result['output_dir']}")


def resolve_training_output_dir(
    *,
    output_dir: Path | None,
    output_root: Path,
    min_depth: int,
    max_depth: int,
    observation_encoding: str,
    version: str | None,
    overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    """Resolve the destination folder for a PPO training run."""

    if output_dir is not None:
        resolved_output_dir = Path(output_dir)
        return resolved_output_dir, {
            "output_dir": str(resolved_output_dir),
            "output_root": None,
            "experiment_name": None,
            "version": None,
            "overwrite": overwrite,
            "output_mode": "explicit",
        }

    experiment_name = build_experiment_name(
        min_depth=min_depth,
        max_depth=max_depth,
        observation_encoding=observation_encoding,
    )
    experiment_dir = Path(output_root) / experiment_name
    resolved_version = (
        str(version).strip()
        if version is not None and str(version).strip()
        else _auto_version_for(experiment_dir, overwrite=overwrite)
    )
    resolved_output_dir = experiment_dir / resolved_version
    return resolved_output_dir, {
        "output_dir": str(resolved_output_dir),
        "output_root": str(output_root),
        "experiment_name": experiment_name,
        "version": resolved_version,
        "overwrite": overwrite,
        "output_mode": "versioned",
    }


def build_experiment_name(
    *,
    min_depth: int,
    max_depth: int,
    observation_encoding: str,
) -> str:
    """Build a readable experiment name from depth range and encoding."""

    if min_depth == max_depth:
        depth_label = f"depth_{min_depth}"
    else:
        depth_label = f"depth_{min_depth}_{max_depth}"
    encoding_label = (
        "onehot"
        if observation_encoding == "one_hot"
        else _safe_name_component(observation_encoding)
    )
    return f"{depth_label}_{encoding_label}"


def _auto_version_for(experiment_dir: Path, *, overwrite: bool) -> str:
    versions = _existing_numeric_versions(experiment_dir)
    if overwrite and versions:
        return f"v{versions[-1]:03d}"
    next_version = versions[-1] + 1 if versions else 1
    return f"v{next_version:03d}"


def _existing_numeric_versions(experiment_dir: Path) -> list[int]:
    if not experiment_dir.exists():
        return []

    versions: list[int] = []
    for child in experiment_dir.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"v(\d+)", child.name)
        if match is not None:
            versions.append(int(match.group(1)))
    return sorted(versions)


def _safe_name_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("_") or "unknown"


if __name__ == "__main__":
    main()
