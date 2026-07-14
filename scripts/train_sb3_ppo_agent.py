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
from agents.sb3_supervised_warm_start import SupervisedWarmStartConfig  # noqa: E402
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
    parser.add_argument("--supervised-warm-start", action="store_true")
    parser.add_argument(
        "--pretrain-depth-mode",
        choices=("mastered", "frontier", "full-curriculum", "custom"),
        default="frontier",
    )
    parser.add_argument("--pretrain-min-depth", type=int)
    parser.add_argument("--pretrain-max-depth", type=int)
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--pretrain-batch-size", type=int, default=256)
    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-3)
    parser.add_argument("--pretrain-sample-per-depth", type=int, default=100_000)
    parser.add_argument("--pretrain-validation-fraction", type=float, default=0.10)
    parser.add_argument("--pretrain-test-fraction", type=float, default=0.10)
    parser.add_argument(
        "--pretrain-depth-sampling",
        choices=("balanced", "natural", "frontier-weighted"),
        default="balanced",
    )
    parser.add_argument("--pretrain-label-smoothing", type=float, default=0.01)
    parser.add_argument("--pretrain-early-stopping-patience", type=int, default=2)
    parser.add_argument("--pretrain-gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--pretrain-update-shared-encoder", action="store_true")
    parser.add_argument(
        "--pretrain-rollout-sample-per-depth", type=int, default=1_000
    )
    args = parser.parse_args()

    curriculum = load_curriculum_config(args.curriculum_config)
    supervised_config = None
    experiment_suffix = None
    if args.supervised_warm_start:
        supervised_config = SupervisedWarmStartConfig(
            depth_mode=args.pretrain_depth_mode,
            min_depth=args.pretrain_min_depth,
            max_depth=args.pretrain_max_depth,
            epochs=args.pretrain_epochs,
            batch_size=args.pretrain_batch_size,
            learning_rate=args.pretrain_learning_rate,
            sample_per_depth=args.pretrain_sample_per_depth,
            validation_fraction=args.pretrain_validation_fraction,
            test_fraction=args.pretrain_test_fraction,
            depth_sampling=args.pretrain_depth_sampling,
            label_smoothing=args.pretrain_label_smoothing,
            early_stopping_patience=args.pretrain_early_stopping_patience,
            gradient_clip_norm=args.pretrain_gradient_clip_norm,
            update_shared_encoder=args.pretrain_update_shared_encoder,
            rollout_sample_per_depth=args.pretrain_rollout_sample_per_depth,
        )
        selected_depths = supervised_config.selected_depths(curriculum)
        experiment_suffix = (
            f"warm_{supervised_config.depth_mode}_"
            f"{selected_depths[0]}_{selected_depths[-1]}"
        )
    output_dir, output_metadata = resolve_training_output_dir(
        output_dir=args.output_dir,
        output_root=args.output_root,
        min_depth=curriculum.min_depth,
        max_depth=curriculum.max_depth,
        observation_encoding="one_hot",
        version=args.version,
        overwrite=args.overwrite,
        experiment_suffix=experiment_suffix,
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
        supervised_config=supervised_config,
    )
    print(f"Saved SB3 PPO artifacts to {result['output_dir']}")


if __name__ == "__main__":
    main()
