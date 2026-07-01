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

from agents.behavior_logging import (  # noqa: E402
    DEFAULT_BEHAVIOR_LOG_ROOT,
    BehaviorLogConfig,
    BehaviorLogger,
    timestamp_run_id,
)
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
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help=(
            "Maximum episodes per depth; small exhaustive datasets are "
            "evaluated once per unique state."
        ),
    )
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--log-behavior", action="store_true")
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_BEHAVIOR_LOG_ROOT)
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--skip-dataset-validation",
        action="store_true",
        help="Skip expensive dataset validation when using trusted generated files.",
    )
    args = parser.parse_args()

    model, _ = load_checkpoint(args.checkpoint, device=args.device)
    behavior_logger = None
    if args.log_behavior:
        model_version = args.model_version or args.checkpoint.parent.name
        run_id = args.run_id or timestamp_run_id()
        behavior_logger = BehaviorLogger(
            BehaviorLogConfig(
                logs_root=args.logs_root,
                model_version=model_version,
                run_id=run_id,
            )
        )

    results = evaluate_ppo_model(
        model,
        depths=range(args.min_depth, args.max_depth + 1),
        episodes_per_depth=args.episodes,
        max_episode_steps=args.max_episode_steps,
        data_dir=args.data_dir,
        device=args.device,
        validate_dataset=not args.skip_dataset_validation,
        behavior_logger=behavior_logger,
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    if behavior_logger is not None:
        written_paths = behavior_logger.flush()
        for kind, paths in written_paths.items():
            for path in paths:
                print(f"Wrote {kind}: {path}")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
