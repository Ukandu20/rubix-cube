"""Train the custom PyTorch PPO cube agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents.ppo_agent import (  # noqa: E402
    DEFAULT_EVAL_EPISODES,
    DEFAULT_EVAL_FREQUENCY,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TOTAL_TIMESTEPS,
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--eval-frequency", type=int, default=DEFAULT_EVAL_FREQUENCY)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
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
    result = train_ppo(
        total_timesteps=args.total_timesteps,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        max_episode_steps=args.max_episode_steps,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        eval_frequency=args.eval_frequency,
        eval_episodes=args.eval_episodes,
        config=config,
        validate_dataset=not args.skip_dataset_validation,
    )
    print(f"Saved PPO artifacts to {result['output_dir']}")


if __name__ == "__main__":
    main()
