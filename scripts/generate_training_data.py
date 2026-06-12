"""Generate cube training data CSV files."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.training_data import (  # noqa: E402
    DEFAULT_DEPTH_COUNTS,
    DEFAULT_OUTPUT_DIR,
    generate_training_datasets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cube training data.")
    parser.add_argument("--depths", nargs="+", type=int, default=None)
    parser.add_argument("--counts", nargs="+", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--unique",
        action="store_true",
        help=(
            "Require unique scrambled states per depth. Use small counts for "
            "shallow depths because depth 1 has only 12 unique states."
        ),
    )
    args = parser.parse_args()

    depth_counts = _parse_depth_counts(args.depths, args.counts)
    output_paths = generate_training_datasets(
        depth_counts=depth_counts,
        output_dir=args.output_dir,
        rng=random.Random(args.seed),
        unique=args.unique,
    )

    for output_path in output_paths:
        print(output_path)


def _parse_depth_counts(
    depths: list[int] | None,
    counts: list[int] | None,
) -> dict[int, int]:
    if depths is None and counts is None:
        return dict(DEFAULT_DEPTH_COUNTS)
    if depths is None or counts is None:
        raise ValueError("--depths and --counts must be provided together")
    if len(depths) != len(counts):
        raise ValueError("--depths and --counts must have the same length")
    return dict(zip(depths, counts))


if __name__ == "__main__":
    main()
