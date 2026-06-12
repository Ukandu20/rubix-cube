"""Generate supervised training examples from random cube scrambles."""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional

from cube.environment import CubeEnvironment
from cube.notation import inverse_algorithm, split_algorithm


DEFAULT_DEPTH_COUNTS: Mapping[int, int] = {
    1: 1000,
    2: 2000,
    3: 5000,
    4: 10000,
    5: 10000,
}
DEFAULT_OUTPUT_DIR = Path("data/processed/training")
FIELDNAMES = (
    "sample_id",
    "scramble_depth",
    "scramble_moves",
    "state_encoded",
    "solution_moves",
    "first_solution_move",
    "is_solved",
)


@dataclass(frozen=True)
class TrainingExample:
    """One scramble/state/solution row for training data."""

    sample_id: str
    scramble_depth: int
    scramble_moves: str
    state_encoded: str
    solution_moves: str
    first_solution_move: str
    is_solved: bool

    def to_row(self) -> dict:
        return asdict(self)


def generate_training_example(
    sample_id: str,
    depth: int,
    rng: Optional[random.Random] = None,
) -> TrainingExample:
    """Generate one training example at an exact scramble depth."""

    if depth < 0:
        raise ValueError("scramble depth cannot be negative")

    random_source = rng if rng is not None else random.Random()
    env = CubeEnvironment(rng=random_source)
    scramble_moves = env.scramble(depth)
    solution_moves = inverse_algorithm(scramble_moves)
    solution_tokens = split_algorithm(solution_moves)

    return TrainingExample(
        sample_id=sample_id,
        scramble_depth=depth,
        scramble_moves=scramble_moves,
        state_encoded=env.get_state(),
        solution_moves=solution_moves,
        first_solution_move=solution_tokens[0] if solution_tokens else "",
        is_solved=env.is_solved(),
    )


def generate_depth_dataset(
    depth: int,
    count: int,
    rng: Optional[random.Random] = None,
    unique: bool = True,
    max_attempts: Optional[int] = None,
) -> list[TrainingExample]:
    """Generate examples for one depth, optionally enforcing unique states."""

    if count < 0:
        raise ValueError("count cannot be negative")
    if count == 0:
        return []

    random_source = rng if rng is not None else random.Random()
    attempt_limit = max_attempts if max_attempts is not None else max(count * 100, 1000)
    examples: list[TrainingExample] = []
    seen_states: set[str] = set()
    attempts = 0

    while len(examples) < count:
        if attempts >= attempt_limit:
            raise ValueError(
                f"could not generate {count} unique examples for depth {depth} "
                f"within {attempt_limit} attempts"
            )

        attempts += 1
        sample_id = f"depth_{depth}_{len(examples) + 1:06d}"
        example = generate_training_example(sample_id, depth, random_source)
        if unique and example.state_encoded in seen_states:
            continue

        examples.append(example)
        seen_states.add(example.state_encoded)

    return examples


def write_depth_csv(
    examples: list[TrainingExample],
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write one depth CSV and return the created path."""

    if not examples:
        raise ValueError("cannot write an empty depth dataset")

    depth = examples[0].scramble_depth
    if any(example.scramble_depth != depth for example in examples):
        raise ValueError("all examples must have the same scramble depth")

    output_path = Path(output_dir) / f"depth_{depth}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(example.to_row() for example in examples)
    return output_path


def generate_training_datasets(
    depth_counts: Mapping[int, int] = DEFAULT_DEPTH_COUNTS,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    rng: Optional[random.Random] = None,
    unique: bool = False,
) -> list[Path]:
    """Generate and write one CSV per requested scramble depth."""

    random_source = rng if rng is not None else random.Random()
    output_paths: list[Path] = []
    for depth, count in sorted(depth_counts.items()):
        examples = generate_depth_dataset(
            depth=depth,
            count=count,
            rng=random_source,
            unique=unique,
        )
        output_paths.append(write_depth_csv(examples, output_dir))
    return output_paths
