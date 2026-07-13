"""Train the parallel Stable-Baselines3 PPO cube agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.sb3_ppo_agent import (  # noqa: E402
    DEFAULT_SB3_OUTPUT_DIR,
    SB3PPOConfig,
    train_sb3_ppo,
)
from cube.gym_environment import DEFAULT_TRAINING_DATA_DIR  # noqa: E402
from curriculum.manager import (  # noqa: E402
    DEFAULT_CURRICULUM_CONFIG_PATH,
    load_curriculum_config,
)
from scripts.train_ppo_agent import resolve_training_output_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train parallel SB3 PPO cube agent.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_TRAINING_DATA_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_SB3_OUTPUT_DIR)
    parser.add_argument("--version")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume an SB3 .zip checkpoint in the selected output directory.",
    )
    parser.add_argument("--curriculum-config", type=Path, default=DEFAULT_CURRICULUM_CONFIG_PATH)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--eval-frequency", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--subprocess", action="store_true")
    parser.add_argument("--skip-dataset-validation", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    args = parser.parse_args()

    curriculum = load_curriculum_config(args.curriculum_config)
    output_dir, output_metadata = resolve_training_output_dir(
        output_dir=args.output_dir,
        output_root=args.output_root,
        min_depth=curriculum.min_depth,
        max_depth=curriculum.max_depth,
        observation_encoding="one_hot",
        version=args.version,
        overwrite=args.overwrite,
    )
    config = SB3PPOConfig(
        learning_rate=args.learning_rate, gamma=args.gamma,
        gae_lambda=args.gae_lambda, clip_range=args.clip_range,
        n_epochs=args.n_epochs, n_steps=args.n_steps, batch_size=args.batch_size,
        ent_coef=args.ent_coef, vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm, target_kl=args.target_kl,
    )
    result = train_sb3_ppo(
        curriculum_config=curriculum, total_timesteps=args.total_timesteps,
        data_dir=args.data_dir, output_dir=output_dir, seed=args.seed,
        device=args.device, eval_frequency=args.eval_frequency,
        eval_episodes=args.eval_episodes, n_envs=args.n_envs,
        subprocess=args.subprocess, config=config,
        validate_dataset=not args.skip_dataset_validation,
        output_metadata=output_metadata,
        resume_from=args.resume_from,
    )
    print(f"Saved SB3 PPO artifacts to {result['output_dir']}")


if __name__ == "__main__":
    main()
