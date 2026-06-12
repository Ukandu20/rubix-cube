"""Train the supervised first-move cube policy."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.supervised_policy import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    CubePolicyDataset,
    SupervisedPolicyNet,
    evaluate_accuracy,
    evaluate_greedy_solver,
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
    args = parser.parse_args()

    random.seed(args.seed)
    rows = load_training_rows(args.data_dir)
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
    random_rows = random_agent_comparison(depths, seed=args.seed)
    metrics = {
        "history": history,
        "validation": accuracy,
        "greedy_solver": greedy,
        "random_agent_comparison": random_rows,
    }
    config = {
        "data_dir": str(args.data_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "solve_cap": args.solve_cap,
        "learning_rate": args.learning_rate,
        "validation_fraction": args.validation_fraction,
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
