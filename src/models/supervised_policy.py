"""PyTorch supervised baseline for first-move cube prediction."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from agents.evaluate import evaluate_random_agent
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
DEFAULT_DATA_DIR = Path("data/processed/training")
DEFAULT_OUTPUT_DIR = Path("models/artifacts/supervised")


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
    """Torch dataset backed by generated training CSV rows."""

    def __init__(self, rows: Sequence[dict]) -> None:
        if not rows:
            raise ValueError("dataset rows cannot be empty")
        self.rows = list(rows)
        self.features = encode_states([row["state_encoded"] for row in self.rows])
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
        return self.features[index], self.labels[index], self.depths[index]


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


def load_training_rows(data_dir: Path | str = DEFAULT_DATA_DIR) -> list[dict]:
    """Load all depth CSV rows from a training-data directory."""

    rows: list[dict] = []
    for csv_path in sorted(Path(data_dir).glob("depth_*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no depth_*.csv files found in {data_dir}")
    return rows


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
