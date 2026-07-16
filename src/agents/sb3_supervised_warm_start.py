"""Supervised actor warm start for the Stable-Baselines3 PPO trainer."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from agents.warm_start_checkpoints import (
    ACTION_COUNT,
    ACTION_ORDER,
)
from agents.warm_start_checkpoints import (
    checkpoint_payload as _checkpoint_payload,
)
from agents.warm_start_checkpoints import (
    load_supervised_warm_start_checkpoint as load_supervised_warm_start_checkpoint,
)
from agents.warm_start_checkpoints import (
    reset_ppo_optimizer as reset_ppo_optimizer,
)
from agents.warm_start_config import (
    DepthSampling as DepthSampling,
)
from agents.warm_start_config import (
    SupervisedWarmStartConfig as SupervisedWarmStartConfig,
)
from cube.encoding import COLOR_TO_INT, validate_encoded_state
from cube.environment import MOVE_TO_ACTION
from cube.gym_environment import INVERSE_ACTION
from cube.moves import apply_move
from cube.state import CubeState
from curriculum.manager import CurriculumConfig

SPLIT_MANIFEST_NAME = "supervised_split_manifest.json"


@dataclass(frozen=True)
class WarmStartExample:
    state: str
    depth: int
    valid_actions: tuple[int, ...]
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...] = ()

    @property
    def state_hash(self) -> str:
        return hashlib.sha256(self.state.encode("utf-8")).hexdigest()


@dataclass
class PreparedWarmStartData:
    depths: tuple[int, ...]
    splits: dict[str, list[WarmStartExample]]
    manifest: dict[str, Any]
    manifest_path: Path

    @property
    def test_states_by_depth(self) -> dict[int, set[str]]:
        result: dict[int, set[str]] = defaultdict(set)
        for example in self.splits["test"]:
            result[example.depth].add(example.state)
        return dict(result)


class WarmStartDataset(Dataset):
    """Lazy one-hot dataset with soft targets for multiple valid moves."""

    def __init__(self, examples: Sequence[WarmStartExample], smoothing: float) -> None:
        self.examples = list(examples)
        self.smoothing = float(smoothing)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        example = self.examples[index]
        target = torch.full(
            (ACTION_COUNT,), self.smoothing / ACTION_COUNT, dtype=torch.float32
        )
        mass = (1.0 - self.smoothing) / len(example.valid_actions)
        for action in example.valid_actions:
            target[action] += mass
        return encode_sb3_state(example.state), target, example.depth


def encode_sb3_state(state: str) -> torch.Tensor:
    """Encode a flat cube string using the environment's SB3 color ordering."""

    if not validate_encoded_state(state):
        raise ValueError("invalid encoded cube state")
    colors = torch.tensor([COLOR_TO_INT[color] for color in state], dtype=torch.long)
    return torch.nn.functional.one_hot(colors, num_classes=6).float().reshape(-1)


def prepare_warm_start_data(
    *,
    config: SupervisedWarmStartConfig,
    curriculum: CurriculumConfig,
    data_dir: Path | str,
    output_dir: Path | str,
    seed: int,
    state_data: Mapping[int, pd.DataFrame] | None = None,
) -> PreparedWarmStartData:
    """Load, normalize, cap, split, and manifest labeled examples."""

    depths = config.selected_depths(curriculum)
    source_files = _source_files(Path(data_dir), depths)
    by_depth: dict[int, list[WarmStartExample]] = {}
    invalid_counts: dict[str, int] = defaultdict(int)
    source_hashes: dict[str, str] = {}
    for depth, path in source_files.items():
        source_hashes[str(path)] = _sha256_file(path)
        if state_data is None:
            examples, counts = _load_depth_examples(path, depth)
        else:
            if depth not in state_data:
                raise ValueError(f"normalized state_data is missing depth {depth}")
            examples, counts = _load_capped_normalized_examples(
                state_data[depth],
                path=path,
                expected_depth=depth,
                cap=config.sample_per_depth,
                seed=seed,
            )
        for name, value in counts.items():
            invalid_counts[name] += value
        by_depth[depth] = examples

    # Raw standalone preparation must reproduce curriculum normalization. During
    # training, state_data already came from CurriculumManager and is normalized.
    seen_states: set[str] = set()
    cross_depth_removed: dict[str, int] = {}
    for depth in depths:
        retained = (
            by_depth[depth]
            if state_data is not None
            else [
                example
                for example in by_depth[depth]
                if example.state not in seen_states
            ]
        )
        removed = len(by_depth[depth]) - len(retained)
        if removed:
            cross_depth_removed[str(depth)] = removed
        if not retained:
            raise ValueError(f"depth {depth} has no labeled rows after deduplication")
        if state_data is None:
            seen_states.update(example.state for example in retained)
            retained = _deterministic_cap(
                retained, config.sample_per_depth, seed=seed, depth=depth
            )
        by_depth[depth] = retained

    splits: dict[str, list[WarmStartExample]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    selected_records: dict[str, list[dict[str, Any]]] = {}
    for depth in depths:
        depth_splits = _split_depth_examples(
            by_depth[depth],
            validation_fraction=config.validation_fraction,
            test_fraction=config.test_fraction,
            seed=seed + depth,
        )
        for split, examples in depth_splits.items():
            splits[split].extend(examples)
        selected_records[str(depth)] = [
            {
                "state_hash": example.state_hash,
                "sample_ids": list(example.sample_ids),
                "partition": next(
                    name for name, values in depth_splits.items() if example in values
                ),
                "valid_actions": [
                    ACTION_ORDER[action] for action in example.valid_actions
                ],
                "group_ids": list(example.group_ids),
            }
            for example in by_depth[depth]
        ]

    _validate_split_integrity(splits, depths)
    counts = {
        split: {
            str(depth): sum(example.depth == depth for example in examples)
            for depth in depths
        }
        for split, examples in splits.items()
    }
    manifest = {
        "version": 1,
        "seed": seed,
        "selected_depths": list(depths),
        "config": asdict(config),
        "action_order": list(ACTION_ORDER),
        "source_files": source_hashes,
        "dataset_counts": {str(depth): len(by_depth[depth]) for depth in depths},
        "split_counts": counts,
        "invalid_rows_removed": dict(invalid_counts),
        "cross_depth_duplicates_removed": cross_depth_removed,
        "selected_rows": selected_records,
    }
    manifest_path = Path(output_dir) / SPLIT_MANIFEST_NAME
    _write_json(manifest_path, manifest)
    return PreparedWarmStartData(depths, splits, manifest, manifest_path)


def _source_files(data_dir: Path, depths: Sequence[int]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for depth in depths:
        parquet = data_dir / f"depth_{depth}.parquet"
        csv = data_dir / f"depth_{depth}.csv"
        path = parquet if parquet.exists() else csv
        if not path.exists():
            raise ValueError(
                f"no labeled dataset found for depth {depth} in {data_dir}"
            )
        result[depth] = path
    return result


def _load_depth_examples(
    path: Path, expected_depth: int
) -> tuple[list[WarmStartExample], dict[str, int]]:
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return _examples_from_frame(frame, path, expected_depth)


def _examples_from_frame(
    frame: pd.DataFrame,
    path: Path,
    expected_depth: int,
) -> tuple[list[WarmStartExample], dict[str, int]]:
    aliases = {
        "state": "state_encoded" if "state_encoded" in frame else "encoded_state",
        "depth": "scramble_depth" if "scramble_depth" in frame else "depth",
    }
    missing = [name for name, column in aliases.items() if column not in frame]
    if missing or "first_solution_move" not in frame.columns:
        raise ValueError(
            f"{path} is missing required labeled columns: "
            "state_encoded/encoded_state, scramble_depth/depth, first_solution_move"
        )
    aggregate: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = defaultdict(int)
    for row_index, row in frame.iterrows():
        state = str(row[aliases["state"]]).strip()
        try:
            depth = int(row[aliases["depth"]])
        except (TypeError, ValueError):
            counts["invalid_depth"] += 1
            continue
        if depth != expected_depth:
            raise ValueError(f"{path} contains a row labeled for depth {depth}")
        if not validate_encoded_state(state):
            counts["invalid_state"] += 1
            continue
        moves = _valid_moves_from_row(row)
        if not moves:
            counts["missing_label"] += 1
            continue
        unknown = set(moves) - set(MOVE_TO_ACTION)
        if unknown:
            raise ValueError(
                f"action labels do not match environment action order: {sorted(unknown)}"
            )
        sample_id = str(row.get("sample_id", f"{path.name}:{row_index}"))
        group_ids = tuple(
            f"{column}:{str(row[column]).strip()}"
            for column in ("trajectory_id", "solution_id")
            if column in frame.columns
            and pd.notna(row[column])
            and str(row[column]).strip()
        )
        record = aggregate.setdefault(
            state, {"moves": set(), "sample_ids": set(), "group_ids": set()}
        )
        record["moves"].update(moves)
        record["sample_ids"].add(sample_id)
        record["group_ids"].update(group_ids)
    examples = [
        WarmStartExample(
            state=state,
            depth=expected_depth,
            valid_actions=tuple(
                sorted(MOVE_TO_ACTION[move] for move in values["moves"])
            ),
            sample_ids=tuple(sorted(values["sample_ids"])),
            group_ids=tuple(sorted(values["group_ids"])),
        )
        for state, values in aggregate.items()
    ]
    counts["duplicate_rows_aggregated"] = (
        len(frame) - sum(counts.values()) - len(examples)
    )
    return examples, dict(counts)


def _load_capped_normalized_examples(
    frame: pd.DataFrame,
    *,
    path: Path,
    expected_depth: int,
    cap: int,
    seed: int,
) -> tuple[list[WarmStartExample], dict[str, int]]:
    """Sample a validated curriculum frame before materializing row objects."""

    if frame.empty:
        raise ValueError(f"depth {expected_depth} has no normalized states")
    if len(frame) > cap:
        rng = np.random.default_rng(seed + expected_depth)
        indices = np.sort(rng.choice(len(frame), size=cap, replace=False))
        selected = frame.iloc[indices]
    else:
        selected = frame
    examples, counts = _examples_from_frame(selected, path, expected_depth)
    return examples, counts


def _valid_moves_from_row(row: pd.Series) -> tuple[str, ...]:
    value = row.get("valid_first_solution_moves")
    if value is not None and not (isinstance(value, float) and math.isnan(value)):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "valid_first_solution_moves must be a JSON array"
                ) from exc
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise ValueError("valid_first_solution_moves must be a list")
        return tuple(str(move).strip() for move in value if str(move).strip())
    move = str(row.get("first_solution_move", "")).strip()
    return (move,) if move and move.lower() != "nan" else ()


def _deterministic_cap(
    examples: Sequence[WarmStartExample], cap: int, *, seed: int, depth: int
) -> list[WarmStartExample]:
    ordered = sorted(examples, key=lambda example: example.state_hash)
    if len(ordered) <= cap:
        return ordered
    return sorted(
        random.Random(f"{seed}:{depth}:cap").sample(ordered, cap),
        key=lambda example: example.state_hash,
    )


def _split_depth_examples(
    examples: Sequence[WarmStartExample],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[WarmStartExample]]:
    # Connected components keep any shared trajectory/solution identifier together.
    parent = list(range(len(examples)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    owner: dict[str, int] = {}
    for index, example in enumerate(examples):
        for group_id in example.group_ids:
            if group_id in owner:
                union(index, owner[group_id])
            else:
                owner[group_id] = index
    groups: dict[int, list[WarmStartExample]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[find(index)].append(example)
    values = list(groups.values())
    random.Random(seed).shuffle(values)
    if len(values) < 3:
        raise ValueError(
            "each supervised depth requires at least three independent groups"
        )
    test_target = max(1, round(len(examples) * test_fraction))
    validation_target = max(1, round(len(examples) * validation_fraction))
    result = {"train": [], "validation": [], "test": []}
    for group in values:
        if len(result["test"]) < test_target:
            result["test"].extend(group)
        elif len(result["validation"]) < validation_target:
            result["validation"].extend(group)
        else:
            result["train"].extend(group)
    if any(not result[name] for name in result):
        raise ValueError("grouped supervised split produced an empty partition")
    return result


def _validate_split_integrity(
    splits: Mapping[str, Sequence[WarmStartExample]], depths: Sequence[int]
) -> None:
    state_sets = {
        name: {example.state for example in examples}
        for name, examples in splits.items()
    }
    if any(
        state_sets[left] & state_sets[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise ValueError("supervised state partitions overlap")
    for name, examples in splits.items():
        available = {example.depth for example in examples}
        if set(depths) - available:
            raise ValueError(f"{name} split does not contain every selected depth")


def depth_sampling_weights(
    config: SupervisedWarmStartConfig,
    curriculum: CurriculumConfig,
    depths: Sequence[int],
    examples: Sequence[WarmStartExample],
) -> dict[int, float]:
    """Return exact depth probabilities for one supervised epoch."""

    if config.depth_sampling == "balanced":
        return {depth: 1.0 / len(depths) for depth in depths}
    if config.depth_sampling == "natural":
        counts = {
            depth: sum(example.depth == depth for example in examples)
            for depth in depths
        }
        total = sum(counts.values())
        return {depth: count / total for depth, count in counts.items()}
    frontier = config.effective_frontier(curriculum)
    raw = curriculum.mixed_sampling_weights.get(frontier, {})
    selected = {depth: float(raw.get(depth, 0.0)) for depth in depths}
    total = sum(selected.values())
    if total <= 0.0 or any(weight <= 0.0 for weight in selected.values()):
        raise ValueError(
            "frontier-weighted pretraining requires positive curriculum weights "
            "for every selected depth"
        )
    return {depth: weight / total for depth, weight in selected.items()}


def epoch_indices(
    examples: Sequence[WarmStartExample],
    *,
    weights: Mapping[int, float],
    strategy: DepthSampling,
    seed: int,
) -> list[int]:
    """Generate deterministic depth-aware indices for one epoch."""

    rng = random.Random(seed)
    if strategy == "natural":
        indices = list(range(len(examples)))
        rng.shuffle(indices)
        return indices
    by_depth: dict[int, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        by_depth[example.depth].append(index)
    depths = list(weights)
    probabilities = [weights[depth] for depth in depths]
    return [
        rng.choice(by_depth[rng.choices(depths, weights=probabilities, k=1)[0]])
        for _ in examples
    ]


def actor_logits(model: Any, features: torch.Tensor) -> torch.Tensor:
    latent = model.policy.mlp_extractor.policy_net(features)
    return model.policy.action_net(latent)


def evaluate_classification(
    model: Any,
    examples: Sequence[WarmStartExample],
    *,
    batch_size: int,
    smoothing: float,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate exhaustive soft-label actor metrics overall and by depth."""

    dataset = WarmStartDataset(examples, smoothing)
    loader = DataLoader(dataset, batch_size=batch_size)
    accumulators: dict[int, dict[str, Any]] = defaultdict(_metric_accumulator)
    model.policy.eval()
    offset = 0
    with torch.no_grad():
        for features, targets, depths in loader:
            features, targets = features.to(device), targets.to(device)
            logits = actor_logits(model, features)
            log_probs = torch.log_softmax(logits, dim=1)
            probabilities = log_probs.exp()
            losses = -(targets * log_probs).sum(dim=1)
            top3 = probabilities.topk(min(3, ACTION_COUNT), dim=1).indices.cpu()
            predictions = probabilities.argmax(dim=1).cpu()
            confidence = probabilities.max(dim=1).values.cpu()
            entropy = -(probabilities * log_probs).sum(dim=1).cpu()
            for local_index, depth_value in enumerate(depths.tolist()):
                example = examples[offset + local_index]
                accumulator = accumulators[int(depth_value)]
                prediction = int(predictions[local_index])
                accumulator["loss"] += float(losses[local_index].item())
                accumulator["top1"] += int(prediction in example.valid_actions)
                accumulator["top3"] += int(
                    bool(set(top3[local_index].tolist()) & set(example.valid_actions))
                )
                accumulator["confidence"] += float(confidence[local_index])
                accumulator["entropy"] += float(entropy[local_index])
                accumulator["count"] += 1
                for action in example.valid_actions:
                    accumulator["action_total"][action] += 1
                    accumulator["action_correct"][action] += int(prediction == action)
            offset += len(depths)
    by_depth = {
        str(depth): _finalize_accumulator(values)
        for depth, values in sorted(accumulators.items())
    }
    total_count = sum(values["count"] for values in accumulators.values())
    micro = _finalize_accumulator(_merge_accumulators(accumulators.values()))
    macro = {
        key: float(np.mean([values[key] for values in by_depth.values()]))
        for key in (
            "loss",
            "top_1_accuracy",
            "top_3_accuracy",
            "macro_action_accuracy",
            "mean_action_confidence",
            "policy_entropy",
        )
    }
    return {
        "row_count": total_count,
        "micro": micro,
        "macro": macro,
        "by_depth": by_depth,
    }


def _metric_accumulator() -> dict[str, Any]:
    return {
        "loss": 0.0,
        "top1": 0,
        "top3": 0,
        "confidence": 0.0,
        "entropy": 0.0,
        "count": 0,
        "action_total": defaultdict(int),
        "action_correct": defaultdict(int),
    }


def _merge_accumulators(values: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    result = _metric_accumulator()
    for value in values:
        for key in ("loss", "top1", "top3", "confidence", "entropy", "count"):
            result[key] += value[key]
        for action, count in value["action_total"].items():
            result["action_total"][action] += count
            result["action_correct"][action] += value["action_correct"][action]
    return result


def _finalize_accumulator(value: Mapping[str, Any]) -> dict[str, Any]:
    count = int(value["count"])
    action_accuracies = [
        value["action_correct"][action] / total
        for action, total in value["action_total"].items()
        if total
    ]
    return {
        "loss": value["loss"] / count if count else 0.0,
        "top_1_accuracy": value["top1"] / count if count else 0.0,
        "top_3_accuracy": value["top3"] / count if count else 0.0,
        "macro_action_accuracy": float(np.mean(action_accuracies))
        if action_accuracies
        else 0.0,
        "mean_action_confidence": value["confidence"] / count if count else 0.0,
        "policy_entropy": value["entropy"] / count if count else 0.0,
        "row_count": count,
    }


def evaluate_greedy_rollouts(
    model: Any,
    examples: Sequence[WarmStartExample],
    *,
    sample_per_depth: int,
    seed: int,
    curriculum: CurriculumConfig,
    mastered_states: set[str],
    device: torch.device,
) -> dict[str, Any]:
    """Run deterministic, capped complete actor rollouts by exact depth."""

    by_depth: dict[int, list[WarmStartExample]] = defaultdict(list)
    for example in examples:
        by_depth[example.depth].append(example)
    results: dict[str, Any] = {}
    for depth, depth_examples in sorted(by_depth.items()):
        selected = _deterministic_cap(
            depth_examples, sample_per_depth, seed=seed, depth=depth
        )
        solved_moves: list[int] = []
        timeouts = inverse_moves = actions = entered = first_top1 = first_top3 = 0
        confidences: list[float] = []
        entropies: list[float] = []
        for example in selected:
            cube = CubeState.from_flat_string(example.state, 3)
            previous_action: int | None = None
            entered_mastered = example.state in mastered_states
            for step in range(curriculum.max_episode_steps(depth)):
                features = (
                    encode_sb3_state(cube.to_flat_string()).unsqueeze(0).to(device)
                )
                with torch.no_grad():
                    probabilities = torch.softmax(actor_logits(model, features), dim=1)[
                        0
                    ]
                action = int(probabilities.argmax().item())
                if step == 0:
                    first_top1 += int(action in example.valid_actions)
                    first_top3 += int(
                        bool(
                            set(probabilities.topk(3).indices.tolist())
                            & set(example.valid_actions)
                        )
                    )
                    confidences.append(float(probabilities.max().item()))
                    entropies.append(
                        float(-(probabilities * probabilities.log()).sum().item())
                    )
                inverse_moves += int(
                    previous_action is not None
                    and action == INVERSE_ACTION[previous_action]
                )
                actions += 1
                cube = apply_move(cube, ACTION_ORDER[action])
                previous_action = action
                entered_mastered = (
                    entered_mastered
                    or cube.is_solved()
                    or cube.to_flat_string() in mastered_states
                )
                if cube.is_solved():
                    solved_moves.append(step + 1)
                    break
            else:
                timeouts += 1
            entered += int(entered_mastered)
        count = len(selected)
        results[str(depth)] = {
            "row_count": count,
            "first_move_top_1_accuracy": first_top1 / count,
            "first_move_top_3_accuracy": first_top3 / count,
            "greedy_solve_rate": len(solved_moves) / count,
            "mean_moves_to_solve": float(np.mean(solved_moves))
            if solved_moves
            else 0.0,
            "median_moves_to_solve": float(median(solved_moves))
            if solved_moves
            else 0.0,
            "timeout_rate": timeouts / count,
            "inverse_move_frequency": inverse_moves / actions if actions else 0.0,
            "policy_entropy": float(np.mean(entropies)) if entropies else 0.0,
            "average_action_confidence": float(np.mean(confidences))
            if confidences
            else 0.0,
            "mastered_region_entry_rate": entered / count,
        }
    return {"sample_cap_per_depth": sample_per_depth, "by_depth": results}


def run_supervised_warm_start(
    model: Any,
    *,
    config: SupervisedWarmStartConfig,
    curriculum: CurriculumConfig,
    data_dir: Path | str,
    output_dir: Path | str,
    seed: int,
    state_data: Mapping[int, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Pretrain the independent SB3 actor and return integration metadata."""

    started = perf_counter()
    output_path = Path(output_dir)
    prepared = prepare_warm_start_data(
        config=config,
        curriculum=curriculum,
        data_dir=data_dir,
        output_dir=output_path,
        seed=seed,
        state_data=state_data,
    )
    device = next(model.policy.parameters()).device
    actor_named = [
        (name, parameter)
        for name, parameter in model.policy.named_parameters()
        if name.startswith("mlp_extractor.policy_net.")
        or name.startswith("action_net.")
    ]
    critic_named = [
        (name, parameter)
        for name, parameter in model.policy.named_parameters()
        if name.startswith("mlp_extractor.value_net.") or name.startswith("value_net.")
    ]
    if not actor_named or not critic_named:
        raise ValueError(
            "loaded SB3 checkpoint architecture is incompatible with actor warm start"
        )
    critic_before = {
        name: parameter.detach().cpu().clone() for name, parameter in critic_named
    }
    initial_actor = {
        name: parameter.detach().cpu().clone() for name, parameter in actor_named
    }
    weights = depth_sampling_weights(
        config, curriculum, prepared.depths, prepared.splits["train"]
    )
    mastered_states = {
        example.state
        for split in prepared.splits.values()
        for example in split
        if example.depth < config.effective_frontier(curriculum)
    }
    before = {
        "classification": {
            split: evaluate_classification(
                model,
                examples,
                batch_size=config.batch_size,
                smoothing=config.label_smoothing,
                device=device,
            )
            for split, examples in prepared.splits.items()
            if split in ("train", "validation")
        },
        "rollouts": {
            split: evaluate_greedy_rollouts(
                model,
                examples,
                sample_per_depth=config.rollout_sample_per_depth,
                seed=seed,
                curriculum=curriculum,
                mastered_states=mastered_states,
                device=device,
            )
            for split, examples in prepared.splits.items()
        },
    }
    optimizer = torch.optim.Adam(
        [parameter for _, parameter in actor_named], lr=config.learning_rate
    )
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_accuracy = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    stale_epochs = 0
    train_dataset = WarmStartDataset(prepared.splits["train"], config.label_smoothing)
    last_path = output_path / "supervised_warm_start_last.pt"
    best_path = output_path / "supervised_warm_start_best.pt"
    for epoch in range(1, config.epochs + 1):
        indices = epoch_indices(
            prepared.splits["train"],
            weights=weights,
            strategy=config.depth_sampling,
            seed=seed + epoch,
        )
        loader = DataLoader(
            train_dataset, batch_size=config.batch_size, sampler=indices
        )
        model.policy.train()
        for features, targets, _ in loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = actor_logits(model, features)
            loss = -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [parameter for _, parameter in actor_named], config.gradient_clip_norm
            )
            optimizer.step()
        train_metrics = evaluate_classification(
            model,
            prepared.splits["train"],
            batch_size=config.batch_size,
            smoothing=config.label_smoothing,
            device=device,
        )
        validation_metrics = evaluate_classification(
            model,
            prepared.splits["validation"],
            batch_size=config.batch_size,
            smoothing=config.label_smoothing,
            device=device,
        )
        row = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(row)
        macro_loss = float(validation_metrics["macro"]["loss"])
        macro_accuracy = float(validation_metrics["macro"]["top_1_accuracy"])
        improved = (macro_loss, -macro_accuracy) < (best_loss, -best_accuracy)
        if improved:
            best_loss, best_accuracy, best_epoch = macro_loss, macro_accuracy, epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in actor_named
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        checkpoint_base = _checkpoint_payload(
            model=model,
            config=config,
            prepared=prepared,
            seed=seed,
            updated_names=[name for name, _ in actor_named],
            best_epoch=best_epoch,
            history=history,
            validation_metrics=validation_metrics,
            runtime=perf_counter() - started,
        )
        torch.save(checkpoint_base, last_path)
        if improved:
            torch.save(checkpoint_base, best_path)
        if stale_epochs >= config.early_stopping_patience:
            break
    if not best_state:
        raise RuntimeError("supervised warm start did not produce a best actor state")
    current = dict(model.policy.named_parameters())
    for name, value in best_state.items():
        current[name].data.copy_(value.to(current[name].device))
    for name, parameter in critic_named:
        if not torch.equal(parameter.detach().cpu(), critic_before[name]):
            raise RuntimeError(
                f"critic parameter changed during actor warm start: {name}"
            )
    actor_changed = [
        name
        for name, parameter in actor_named
        if not torch.equal(parameter.detach().cpu(), initial_actor[name])
    ]
    if not actor_changed:
        raise RuntimeError("supervised warm start did not update actor parameters")
    after_classification = {
        split: evaluate_classification(
            model,
            examples,
            batch_size=config.batch_size,
            smoothing=config.label_smoothing,
            device=device,
        )
        for split, examples in prepared.splits.items()
        if split in ("train", "validation")
    }
    after_rollouts = {
        split: evaluate_greedy_rollouts(
            model,
            examples,
            sample_per_depth=config.rollout_sample_per_depth,
            seed=seed,
            curriculum=curriculum,
            mastered_states=mastered_states,
            device=device,
        )
        for split, examples in prepared.splits.items()
    }
    final_best_payload = _checkpoint_payload(
        model=model,
        config=config,
        prepared=prepared,
        seed=seed,
        updated_names=[name for name, _ in actor_named],
        best_epoch=best_epoch,
        history=history,
        validation_metrics=after_classification["validation"],
        runtime=perf_counter() - started,
    )
    final_best_payload["pre_ppo_evaluation"] = {
        "classification": after_classification,
        "rollouts": after_rollouts,
    }
    torch.save(final_best_payload, best_path)
    reference_examples = _diagnostic_examples(
        prepared.splits["validation"],
        cap_per_depth=min(100, config.rollout_sample_per_depth),
        seed=seed,
    )
    reference_probs = policy_probabilities(model, reference_examples, device=device)
    manifest_hash = _sha256_file(prepared.manifest_path)
    checkpoint_hash = _sha256_file(best_path)
    runtime = perf_counter() - started
    provenance = {
        "initialization_mode": "supervised_actor_warm_start",
        "warm_start_checkpoint_path": str(best_path),
        "warm_start_checkpoint_hash": checkpoint_hash,
        "supervised_depth_mode": config.depth_mode,
        "supervised_depth_range": [prepared.depths[0], prepared.depths[-1]],
        "supervised_seed": seed,
        "supervised_hyperparameters": asdict(config),
        "supervised_dataset_counts": prepared.manifest["dataset_counts"],
        "supervised_split_manifest": str(prepared.manifest_path),
        "supervised_split_manifest_hash": manifest_hash,
        "supervised_best_epoch": best_epoch,
        "supervised_validation_metrics": after_classification["validation"],
        "supervised_rollout_metrics": after_rollouts,
        "updated_parameter_groups": [name for name, _ in actor_named],
        "shared_encoder_updated": False,
        "critic_head_updated": False,
        "supervised_runtime_seconds": runtime,
    }
    _write_json(
        output_path / "supervised_warm_start_metrics.json",
        {
            "before_supervised_pretraining": before,
            "after_supervised_pretraining": {
                "classification": after_classification,
                "rollouts": after_rollouts,
            },
            "training_history": history,
            "best_epoch": best_epoch,
            "depth_sampling_weights": weights,
            "provenance": provenance,
        },
    )
    return {
        "provenance": provenance,
        "metrics": {
            "before_supervised_pretraining": before,
            "after_supervised_pretraining": {
                "classification": after_classification,
                "rollouts": after_rollouts,
            },
            "training_history": history,
        },
        "test_states_by_depth": prepared.test_states_by_depth,
        "diagnostic_examples": reference_examples,
        "reference_probabilities": reference_probs,
        "mastered_states": mastered_states,
    }


def policy_probabilities(
    model: Any, examples: Sequence[WarmStartExample], *, device: torch.device
) -> torch.Tensor:
    if not examples:
        return torch.empty((0, ACTION_COUNT))
    features = torch.stack(
        [encode_sb3_state(example.state) for example in examples]
    ).to(device)
    with torch.no_grad():
        return torch.softmax(actor_logits(model, features), dim=1).cpu()


def transition_snapshot(
    model: Any,
    examples: Sequence[WarmStartExample],
    reference_probabilities: torch.Tensor,
    *,
    phase: str,
    logger_values: Mapping[str, Any] | None = None,
    curriculum: CurriculumConfig | None = None,
    mastered_states: set[str] | None = None,
) -> dict[str, Any]:
    """Capture fixed-set actor retention and available PPO update diagnostics."""

    device = next(model.policy.parameters()).device
    current = policy_probabilities(model, examples, device=device)
    reference = reference_probabilities.clamp_min(1e-8)
    current_safe = current.clamp_min(1e-8)
    kl = (
        float((reference * (reference.log() - current_safe.log())).sum(dim=1).mean())
        if len(current)
        else 0.0
    )
    classification = (
        evaluate_classification(
            model,
            examples,
            batch_size=max(1, min(256, len(examples))),
            smoothing=0.0,
            device=device,
        )
        if examples
        else {}
    )
    values = dict(logger_values or {})

    def metric(name: str) -> float | None:
        value = values.get(name)
        return None if value is None else float(value)

    snapshot = {
        "phase": phase,
        "timesteps": int(model.num_timesteps),
        "kl_from_warm_start": kl,
        "first_move_metrics": classification,
        "actor_loss": metric("train/policy_gradient_loss"),
        "critic_loss": metric("train/value_loss"),
        "clip_fraction": metric("train/clip_fraction"),
        "policy_entropy_loss": metric("train/entropy_loss"),
        "explained_variance": metric("train/explained_variance"),
        "approx_kl": metric("train/approx_kl"),
    }
    if examples and curriculum is not None:
        snapshot["greedy_rollouts"] = evaluate_greedy_rollouts(
            model,
            examples,
            sample_per_depth=max(
                sum(example.depth == depth for example in examples)
                for depth in {example.depth for example in examples}
            ),
            seed=0,
            curriculum=curriculum,
            mastered_states=mastered_states or set(),
            device=device,
        )
    return snapshot


def _diagnostic_examples(
    examples: Sequence[WarmStartExample], *, cap_per_depth: int, seed: int
) -> list[WarmStartExample]:
    by_depth: dict[int, list[WarmStartExample]] = defaultdict(list)
    for example in examples:
        by_depth[example.depth].append(example)
    result: list[WarmStartExample] = []
    for depth, values in sorted(by_depth.items()):
        result.extend(_deterministic_cap(values, cap_per_depth, seed=seed, depth=depth))
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
