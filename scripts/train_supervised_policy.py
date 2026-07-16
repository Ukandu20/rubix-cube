"""Train the supervised first-move cube policy."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from models.supervised_policy import (  # noqa: E402
    DATA_FORMATS,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    CubePolicyDataset,
    SupervisedPolicyNet,
    bfs_agent_comparison,
    evaluate_accuracy,
    evaluate_greedy_solver,
    inverse_scramble_comparison,
    load_training_rows,
    random_agent_comparison,
    save_artifacts,
    split_dataset,
    train_policy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train supervised cube policy.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--solve-cap", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--comparison-episodes", type=int, default=20)
    parser.add_argument("--bfs-episodes", type=int, default=1)
    parser.add_argument("--bfs-max-depth", type=int, default=7)
    parser.add_argument("--min-depth", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--sample-per-depth", type=int, default=None)
    parser.add_argument(
        "--data-format",
        choices=DATA_FORMATS,
        default="auto",
        help="Training data format. auto prefers depth_N.parquet when present.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    rows = load_training_rows(
        args.data_dir,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        max_rows=args.max_rows,
        sample_per_depth=args.sample_per_depth,
        seed=args.seed,
        data_format=args.data_format,
    )
    dataset = CubePolicyDataset(rows)
    train_dataset, validation_dataset = split_dataset(
        dataset,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    validation_rows = [dataset.rows[index] for index in validation_dataset.indices]

    model = SupervisedPolicyNet()
    history = train_policy(
        model,
        train_dataset,
        validation_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    accuracy = evaluate_accuracy(model, validation_dataset, batch_size=args.batch_size)
    greedy = evaluate_greedy_solver(model, validation_rows, solve_cap=args.solve_cap)
    depths = sorted({int(row["scramble_depth"]) for row in validation_rows})
    random_rows = random_agent_comparison(
        depths,
        episodes_per_depth=args.comparison_episodes,
        seed=args.seed,
    )
    inverse_rows = inverse_scramble_comparison(
        depths,
        episodes_per_depth=args.comparison_episodes,
        seed=args.seed,
    )
    bfs_rows = bfs_agent_comparison(
        depths,
        episodes_per_depth=args.bfs_episodes,
        max_depth=args.bfs_max_depth,
        seed=args.seed,
    )
    metrics = {
        "history": history,
        "validation": accuracy,
        "greedy_solver": greedy,
        "random_agent_comparison": random_rows,
        "inverse_scramble_comparison": inverse_rows,
        "bfs_agent_comparison": bfs_rows,
    }
    config = {
        "data_dir": str(args.data_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "solve_cap": args.solve_cap,
        "learning_rate": args.learning_rate,
        "validation_fraction": args.validation_fraction,
        "comparison_episodes": args.comparison_episodes,
        "bfs_episodes": args.bfs_episodes,
        "bfs_max_depth": args.bfs_max_depth,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "max_rows": args.max_rows,
        "sample_per_depth": args.sample_per_depth,
        "data_format": args.data_format,
        "loaded_examples": len(dataset),
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
    }
    paths = save_artifacts(model, args.output_dir, config, metrics)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved artifacts to {args.output_dir}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
