"""Evaluate a saved custom PyTorch PPO cube agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents.ppo_agent import (  # noqa: E402
    DEFAULT_EVAL_EPISODES,
    evaluate_ppo_model,
    load_checkpoint,
)
from cube.gym_environment import (  # noqa: E402
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_TRAINING_DATA_DIR,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate custom PPO cube agent.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_TRAINING_DATA_DIR)
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--skip-dataset-validation",
        action="store_true",
        help="Skip expensive dataset validation when using trusted generated files.",
    )
    args = parser.parse_args()

    model, _ = load_checkpoint(args.checkpoint, device=args.device)
    results = evaluate_ppo_model(
        model,
        depths=range(args.min_depth, args.max_depth + 1),
        episodes_per_depth=args.episodes,
        max_episode_steps=args.max_episode_steps,
        data_dir=args.data_dir,
        seed=args.seed,
        device=args.device,
        validate_dataset=not args.skip_dataset_validation,
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
