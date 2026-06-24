"""PyTorch supervised baseline for first-move cube prediction."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from agents.evaluate import (
    evaluate_bfs_agent,
    evaluate_inverse_scramble_agent,
    evaluate_random_agent,
)
from cube.environment import ACTION_TO_MOVE
from cube.moves import apply_move
from cube.state import CubeState


COLOR_ORDER = ("W", "R", "G", "Y", "O", "B")
COLOR_TO_INDEX = {color: index for index, color in enumerate(COLOR_ORDER)}
MOVE_ORDER = tuple(ACTION_TO_MOVE[index] for index in sorted(ACTION_TO_MOVE))
LABEL_TO_INDEX = {move: index for index, move in enumerate(MOVE_ORDER)}
INDEX_TO_MOVE = {index: move for move, index in LABEL_TO_INDEX.items()}
STICKER_COUNT = 54
INPUT_SIZE = STICKER_COUNT * len(COLOR_ORDER)
DEFAULT_TRAINING_ROOT = Path("data/processed/training")
DEFAULT_CSV_DATA_DIR = DEFAULT_TRAINING_ROOT / "csv"
DEFAULT_PARQUET_DATA_DIR = DEFAULT_TRAINING_ROOT / "parquet"
DEFAULT_DATA_DIR = DEFAULT_PARQUET_DATA_DIR
DEFAULT_OUTPUT_DIR = Path("models/artifacts/supervised")
DATA_FORMATS = ("auto", "csv", "parquet")
TRAINING_ROW_COLUMNS = (
    "sample_id",
    "scramble_depth",
    "scramble_moves",
    "state_encoded",
    "solution_moves",
    "first_solution_move",
)


def encode_state(state_encoded: str) -> torch.Tensor:
    """Encode one flat cube string as 54 x 6 one-hot features."""

    if len(state_encoded) != STICKER_COUNT:
        raise ValueError(f"state must contain {STICKER_COUNT} stickers")

    encoded = torch.zeros(INPUT_SIZE, dtype=torch.float32)
    for sticker_index, color in enumerate(state_encoded):
        if color not in COLOR_TO_INDEX:
            raise ValueError(f"unknown sticker color: {color!r}")
        encoded[sticker_index * len(COLOR_ORDER) + COLOR_TO_INDEX[color]] = 1.0
    return encoded


def encode_states(states: Sequence[str]) -> torch.Tensor:
    """Encode many flat cube strings into a tensor."""

    if not states:
        return torch.empty((0, INPUT_SIZE), dtype=torch.float32)
    return torch.stack([encode_state(state) for state in states])


class CubePolicyDataset(Dataset):
    """Torch dataset backed by generated training rows."""

    def __init__(self, rows: Sequence[dict]) -> None:
        if not rows:
            raise ValueError("dataset rows cannot be empty")
        self.rows = list(rows)
        self.labels = torch.tensor(
            [_label_for_move(row["first_solution_move"]) for row in self.rows],
            dtype=torch.long,
        )
        self.depths = torch.tensor(
            [int(row["scramble_depth"]) for row in self.rows],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return encode_state(self.rows[index]["state_encoded"]), self.labels[index], self.depths[index]


class SupervisedPolicyNet(nn.Module):
    """Small MLP classifier over cube state one-hot features."""

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = 256,
        output_size: int = len(MOVE_ORDER),
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def load_training_rows(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    *,
    min_depth: Optional[int] = None,
    max_depth: Optional[int] = None,
    max_rows: Optional[int] = None,
    sample_per_depth: Optional[int] = None,
    seed: int = 0,
    data_format: Literal["auto", "csv", "parquet"] = "auto",
) -> list[dict]:
    """Load training rows with optional depth filtering and sampling."""

    _validate_row_loading_options(
        min_depth=min_depth,
        max_depth=max_depth,
        max_rows=max_rows,
        sample_per_depth=sample_per_depth,
        data_format=data_format,
    )

    files = _training_data_files(Path(data_dir), data_format, min_depth, max_depth)
    if not files:
        raise ValueError(f"no depth data files found in {data_dir}")

    random_source = random.Random(seed)
    columns = tuple(TRAINING_ROW_COLUMNS)
    if max_rows is not None and sample_per_depth is None:
        rows = _sample_rows_from_files(files, max_rows, random_source, columns)
    else:
        rows = []
        for _, path in files:
            depth_rows = _load_depth_file(path, columns, sample_per_depth, random_source)
            rows.extend(depth_rows)

        if max_rows is not None and len(rows) > max_rows:
            rows = random_source.sample(rows, max_rows)

    if not rows:
        raise ValueError(f"no training rows found in {data_dir}")
    return rows


def _validate_row_loading_options(
    *,
    min_depth: Optional[int],
    max_depth: Optional[int],
    max_rows: Optional[int],
    sample_per_depth: Optional[int],
    data_format: str,
) -> None:
    if data_format not in DATA_FORMATS:
        raise ValueError(f"data_format must be one of: {', '.join(DATA_FORMATS)}")
    if min_depth is not None and min_depth < 0:
        raise ValueError("min_depth cannot be negative")
    if max_depth is not None and max_depth < 0:
        raise ValueError("max_depth cannot be negative")
    if min_depth is not None and max_depth is not None and min_depth > max_depth:
        raise ValueError("min_depth cannot be greater than max_depth")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if sample_per_depth is not None and sample_per_depth <= 0:
        raise ValueError("sample_per_depth must be positive")


def _training_data_files(
    data_dir: Path,
    data_format: str,
    min_depth: Optional[int],
    max_depth: Optional[int],
) -> list[tuple[int, Path]]:
    if data_format == "auto":
        parquet_by_depth = _depth_files_by_suffix(data_dir, ".parquet")
        csv_by_depth = _depth_files_by_suffix(data_dir, ".csv")
        depths = sorted(set(parquet_by_depth) | set(csv_by_depth))
        return [
            (
                depth,
                parquet_by_depth[depth] if depth in parquet_by_depth else csv_by_depth[depth],
            )
            for depth in depths
            if _depth_allowed(depth, min_depth, max_depth)
        ]

    suffix = ".csv" if data_format == "csv" else ".parquet"
    return [
        (depth, path)
        for depth, path in sorted(_depth_files_by_suffix(data_dir, suffix).items())
        if _depth_allowed(depth, min_depth, max_depth)
    ]


def _depth_files_by_suffix(data_dir: Path, suffix: str) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in data_dir.glob(f"depth_*{suffix}"):
        depth = _depth_from_path(path)
        if depth is not None:
            files[depth] = path
    return files


def _depth_from_path(path: Path) -> Optional[int]:
    try:
        return int(path.stem.removeprefix("depth_"))
    except ValueError:
        return None


def _depth_allowed(
    depth: int,
    min_depth: Optional[int],
    max_depth: Optional[int],
) -> bool:
    if min_depth is not None and depth < min_depth:
        return False
    if max_depth is not None and depth > max_depth:
        return False
    return True


def _load_depth_file(
    path: Path,
    columns: Sequence[str],
    sample_per_depth: Optional[int],
    random_source: random.Random,
) -> list[dict]:
    if path.suffix == ".csv":
        if sample_per_depth is not None:
            return _sample_csv_rows(path, columns, sample_per_depth, random_source)
        return list(_iter_csv_rows(path, columns))
    if path.suffix == ".parquet":
        rows = _read_parquet_rows(path, columns)
        if sample_per_depth is not None and len(rows) > sample_per_depth:
            return random_source.sample(rows, sample_per_depth)
        return rows
    raise ValueError(f"unsupported training data file type: {path.suffix}")


def _sample_rows_from_files(
    files: Sequence[tuple[int, Path]],
    sample_size: int,
    random_source: random.Random,
    columns: Sequence[str],
) -> list[dict]:
    reservoir: list[dict] = []
    seen = 0
    for _, path in files:
        if path.suffix == ".csv":
            iterator = _iter_csv_rows(path, columns)
        elif path.suffix == ".parquet":
            iterator = iter(_read_parquet_rows(path, columns))
        else:
            raise ValueError(f"unsupported training data file type: {path.suffix}")

        for row in iterator:
            seen += 1
            if len(reservoir) < sample_size:
                reservoir.append(row)
                continue
            replacement_index = random_source.randrange(seen)
            if replacement_index < sample_size:
                reservoir[replacement_index] = row
    return reservoir


def _sample_csv_rows(
    path: Path,
    columns: Sequence[str],
    sample_size: int,
    random_source: random.Random,
) -> list[dict]:
    reservoir: list[dict] = []
    for seen, row in enumerate(_iter_csv_rows(path, columns), start=1):
        if len(reservoir) < sample_size:
            reservoir.append(row)
            continue
        replacement_index = random_source.randrange(seen)
        if replacement_index < sample_size:
            reservoir[replacement_index] = row
    return reservoir


def _iter_csv_rows(path: Path, columns: Sequence[str]) -> Iterable[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield _compact_row(row, columns)


def _read_parquet_rows(path: Path, columns: Sequence[str]) -> list[dict]:
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError(
            "Parquet loading requires pandas and pyarrow. "
            "Install dependencies with `python -m pip install -r requirements.txt`."
        ) from error

    try:
        frame = pd.read_parquet(path, columns=list(columns))
    except ImportError as error:
        raise ImportError(
            "Parquet loading requires pyarrow or fastparquet. "
            "Install dependencies with `python -m pip install -r requirements.txt`."
        ) from error
    return [_compact_row(row, columns) for row in frame.to_dict(orient="records")]


def _compact_row(row: dict, columns: Sequence[str]) -> dict:
    return {column: row[column] for column in columns if column in row}


def split_dataset(
    dataset: Dataset,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[Subset, Subset]:
    """Create a random train/validation split."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    validation_size = max(1, int(len(indices) * validation_fraction))
    train_indices = indices[validation_size:]
    validation_indices = indices[:validation_size]
    if not train_indices:
        raise ValueError("training split would be empty")
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


def train_policy(
    model: nn.Module,
    train_dataset: Dataset,
    validation_dataset: Optional[Dataset] = None,
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: Optional[torch.device | str] = None,
) -> list[dict]:
    """Train the policy network and return per-epoch metrics."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    run_device = torch.device(device or "cpu")
    model.to(run_device)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for features, labels, _ in loader:
            features = features.to(run_device)
            labels = labels.to(run_device)
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size_actual = labels.shape[0]
            total_loss += loss.item() * batch_size_actual
            total_examples += batch_size_actual

        metrics = {
            "epoch": epoch,
            "train_loss": total_loss / total_examples,
        }
        if validation_dataset is not None:
            validation_metrics = evaluate_accuracy(
                model,
                validation_dataset,
                batch_size=batch_size,
                device=run_device,
            )
            metrics["validation_accuracy"] = validation_metrics["accuracy"]
        history.append(metrics)

    return history


def evaluate_accuracy(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 256,
    device: Optional[torch.device | str] = None,
) -> dict:
    """Evaluate first-move classification accuracy overall and by depth."""

    run_device = torch.device(device or "cpu")
    model.to(run_device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    correct = 0
    total = 0
    depth_correct: dict[int, int] = defaultdict(int)
    depth_total: dict[int, int] = defaultdict(int)

    with torch.no_grad():
        for features, labels, depths in loader:
            features = features.to(run_device)
            labels = labels.to(run_device)
            predictions = model(features).argmax(dim=1)
            matches = predictions.eq(labels)
            correct += int(matches.sum().item())
            total += int(labels.numel())

            for depth, match in zip(depths.tolist(), matches.cpu().tolist()):
                depth_total[int(depth)] += 1
                depth_correct[int(depth)] += int(match)

    return {
        "accuracy": correct / total if total else 0.0,
        "accuracy_by_depth": {
            str(depth): depth_correct[depth] / depth_total[depth]
            for depth in sorted(depth_total)
        },
        "examples": total,
    }


def predict_move(
    model: nn.Module,
    state_encoded: str,
    device: Optional[torch.device | str] = None,
) -> str:
    """Predict one move token from a flat cube state."""

    run_device = torch.device(device or "cpu")
    model.to(run_device)
    model.eval()
    with torch.no_grad():
        features = encode_state(state_encoded).unsqueeze(0).to(run_device)
        prediction = int(model(features).argmax(dim=1).item())
    return INDEX_TO_MOVE[prediction]


def greedy_solve(
    model: nn.Module,
    state_encoded: str,
    solve_cap: int = 30,
    device: Optional[torch.device | str] = None,
) -> dict:
    """Repeatedly predict and apply moves until solved or capped."""

    if solve_cap < 0:
        raise ValueError("solve_cap cannot be negative")

    cube = CubeState.from_flat_string(state_encoded, 3)
    moves: list[str] = []
    for _ in range(solve_cap):
        if cube.is_solved():
            return {"solved": True, "moves": moves, "move_count": len(moves)}
        move = predict_move(model, cube.to_flat_string(), device=device)
        cube = apply_move(cube, move)
        moves.append(move)

    return {
        "solved": cube.is_solved(),
        "moves": moves,
        "move_count": len(moves),
    }


def evaluate_greedy_solver(
    model: nn.Module,
    rows: Sequence[dict],
    solve_cap: int = 30,
    device: Optional[torch.device | str] = None,
) -> dict:
    """Evaluate greedy recursive solving by scramble depth."""

    depth_total: dict[int, int] = defaultdict(int)
    depth_solved: dict[int, int] = defaultdict(int)
    depth_moves: dict[int, list[int]] = defaultdict(list)

    for row in rows:
        depth = int(row["scramble_depth"])
        result = greedy_solve(model, row["state_encoded"], solve_cap, device=device)
        depth_total[depth] += 1
        if result["solved"]:
            depth_solved[depth] += 1
            depth_moves[depth].append(int(result["move_count"]))

    return {
        "greedy_solve_rate_by_depth": {
            str(depth): depth_solved[depth] / depth_total[depth]
            for depth in sorted(depth_total)
        },
        "greedy_avg_moves_by_depth": {
            str(depth): (
                sum(depth_moves[depth]) / len(depth_moves[depth])
                if depth_moves[depth]
                else 0.0
            )
            for depth in sorted(depth_total)
        },
        "solve_cap": solve_cap,
    }


def random_agent_comparison(
    depths: Iterable[int],
    episodes_per_depth: int = 20,
    seed: int = 0,
) -> list[dict]:
    """Run existing random-agent evaluator for comparison rows."""

    return evaluate_random_agent(
        depths=depths,
        episodes_per_depth=episodes_per_depth,
        rng=random.Random(seed),
    )


def inverse_scramble_comparison(
    depths: Iterable[int],
    episodes_per_depth: int = 20,
    seed: int = 0,
) -> list[dict]:
    """Run inverse-scramble oracle baseline for comparison rows."""

    return evaluate_inverse_scramble_agent(
        depths=depths,
        episodes_per_depth=episodes_per_depth,
        rng=random.Random(seed),
    )


def bfs_agent_comparison(
    depths: Iterable[int],
    episodes_per_depth: int = 1,
    max_depth: int = 7,
    seed: int = 0,
) -> list[dict]:
    """Run shallow BFS baseline for comparison rows."""

    return evaluate_bfs_agent(
        depths=depths,
        episodes_per_depth=episodes_per_depth,
        max_depth=max_depth,
        rng=random.Random(seed),
    )


def save_artifacts(
    model: nn.Module,
    output_dir: Path | str,
    config: dict,
    metrics: dict,
) -> dict[str, Path]:
    """Save model checkpoint, mappings, config, and metrics."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "policy": destination / "policy.pt",
        "label_mapping": destination / "label_mapping.json",
        "config": destination / "config.json",
        "metrics": destination / "metrics.json",
    }
    torch.save(model.state_dict(), paths["policy"])
    _write_json(paths["label_mapping"], {"label_to_index": LABEL_TO_INDEX})
    _write_json(paths["config"], config)
    _write_json(paths["metrics"], metrics)
    return paths


def _label_for_move(move: str) -> int:
    if move not in LABEL_TO_INDEX:
        raise ValueError(f"unknown solution move label: {move!r}")
    return LABEL_TO_INDEX[move]


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
