"""Run baseline cube-agent evaluations."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents.evaluate import (  # noqa: E402
    DEFAULT_CSV_PATH,
    evaluate_bfs_agent,
    evaluate_inverse_scramble_agent,
    evaluate_random_agent,
    print_results_table,
    write_results_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cube baseline agents.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--bfs-episodes", type=int, default=1)
    parser.add_argument("--bfs-max-depth", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    results = []
    results.extend(evaluate_random_agent(episodes_per_depth=args.episodes, rng=rng))
    results.extend(
        evaluate_inverse_scramble_agent(episodes_per_depth=args.episodes, rng=rng)
    )
    results.extend(
        evaluate_bfs_agent(
            episodes_per_depth=args.bfs_episodes,
            max_depth=args.bfs_max_depth,
            rng=rng,
        )
    )

    print_results_table(results)
    if not args.no_csv:
        output_path = write_results_csv(results, args.csv)
        print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
