"""Inference and benchmarking services for the local Streamlit showcase."""

from __future__ import annotations

import csv
import io
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol

import numpy as np
import torch
import yaml

from agents.ppo_agent import (
    ACTION_SIZE,
    OBSERVATION_SIZE,
    ActorCriticNet,
    load_checkpoint,
    preprocess_observations,
)
from cube.environment import ACTION_TO_MOVE, CubeEnvironment
from cube.moves import apply_move
from cube.notation import inverse_algorithm
from cube.state import CubeState

ScrambleMode = Literal["exact", "range", "curriculum"]
TerminationReason = Literal[
    "solved", "move_limit", "time_limit", "cycle", "attempt_limit"
]


@dataclass(frozen=True)
class CheckpointInfo:
    """Display and compatibility metadata for one local PPO checkpoint."""

    path: Path
    label: str
    compatible: bool
    error: str | None
    validated_depth: int
    checkpoint_curriculum_depth: int
    run_curriculum_depth: int
    observation_encoding: str
    timesteps: int
    trainer: str = "custom"

    def supports(self, scramble_length: int) -> bool:
        return scramble_length <= self.validated_depth


@dataclass(frozen=True)
class ActionDecision:
    """One policy decision and its complete action distribution."""

    action: int
    probabilities: tuple[float, ...]

    @property
    def move(self) -> str:
        return ACTION_TO_MOVE[self.action]

    def top_actions(self, count: int = 3) -> list[tuple[str, float]]:
        ranked = sorted(
            enumerate(self.probabilities),
            key=lambda item: item[1],
            reverse=True,
        )
        return [
            (ACTION_TO_MOVE[action], float(probability))
            for action, probability in ranked[:count]
        ]


@dataclass(frozen=True)
class SolveFrame:
    """Cube state captured after a model action."""

    state: str
    step: int
    move: str | None = None
    decision: ActionDecision | None = None


@dataclass(frozen=True)
class AttemptResult:
    """Result of one greedy or stochastic policy rollout."""

    attempt: int
    mode: Literal["greedy", "stochastic"]
    solved: bool
    termination_reason: TerminationReason
    moves: tuple[str, ...]
    frames: tuple[SolveFrame, ...]
    inference_seconds: float
    solver_seconds: float


@dataclass(frozen=True)
class RunConfig:
    """Limits and random controls for solving one scramble."""

    max_moves: int | None = None
    max_attempts: int = 3
    timeout_seconds: float = 5.0
    temperature: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.max_moves is not None and self.max_moves <= 0:
            raise ValueError("max_moves must be positive when provided")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be a positive finite number")


@dataclass(frozen=True)
class SolveResult:
    """All attempts and timing for one fixed scramble."""

    scramble_length: int
    scramble: str
    initial_state: str
    seed: int
    solved: bool
    greedy_solved: bool
    termination_reason: TerminationReason
    attempts: tuple[AttemptResult, ...]
    inference_seconds: float
    solver_seconds: float

    @property
    def attempts_used(self) -> int:
        return len(self.attempts)

    @property
    def total_moves(self) -> int:
        return sum(len(attempt.moves) for attempt in self.attempts)

    @property
    def display_attempt(self) -> AttemptResult:
        solved = next((attempt for attempt in self.attempts if attempt.solved), None)
        return solved or self.attempts[-1]


@dataclass(frozen=True)
class BenchmarkResult:
    """Detailed and aggregate benchmark records."""

    records: tuple[dict[str, object], ...]
    summary: tuple[dict[str, object], ...]


class Policy(Protocol):
    """Minimal interface used by the solver."""

    def decide(
        self,
        state: str,
        *,
        greedy: bool,
        temperature: float,
        rng: random.Random,
    ) -> ActionDecision:
        """Choose one action for an encoded cube state."""


class TorchPolicy:
    """CPU PPO policy adapter with reproducible Python-side sampling."""

    def __init__(self, model: ActorCriticNet) -> None:
        self.model = model
        self.model.eval()

    def decide(
        self,
        state: str,
        *,
        greedy: bool,
        temperature: float,
        rng: random.Random,
    ) -> ActionDecision:
        observation = np.fromiter(
            ("YOGWRB".index(sticker) for sticker in state),
            dtype=np.int8,
            count=OBSERVATION_SIZE,
        )
        with torch.no_grad():
            logits, _ = self.model(
                preprocess_observations(
                    observation,
                    "cpu",
                    observation_encoding=self.model.config.observation_encoding,
                )
            )
            probabilities = torch.softmax(logits[0] / temperature, dim=0)
        values = tuple(float(value) for value in probabilities.cpu().tolist())
        if greedy:
            action = max(range(len(values)), key=values.__getitem__)
        else:
            action = rng.choices(range(len(values)), weights=values, k=1)[0]
        return ActionDecision(action=action, probabilities=values)


def load_torch_policy(path: Path | str) -> TorchPolicy:
    """Load and validate a trusted local PPO checkpoint on CPU."""

    model, _ = load_checkpoint(path, device="cpu")
    config = model.config
    expected_input = 324 if config.observation_encoding == "one_hot" else 54
    if config.actor_output_dim != ACTION_SIZE:
        raise ValueError(
            f"incompatible actor output: expected {ACTION_SIZE}, "
            f"found {config.actor_output_dim}"
        )
    if config.input_dim != expected_input:
        raise ValueError(
            f"incompatible input size for {config.observation_encoding}: "
            f"expected {expected_input}, found {config.input_dim}"
        )
    return TorchPolicy(model)


class SB3Policy:
    """Showcase adapter for a Stable-Baselines3 categorical policy."""

    def __init__(self, model: object) -> None:
        self.model = model

    def decide(
        self,
        state: str,
        *,
        greedy: bool,
        temperature: float,
        rng: random.Random,
    ) -> ActionDecision:
        observation = np.fromiter(
            ("YOGWRB".index(sticker) for sticker in state),
            dtype=np.int64,
            count=OBSERVATION_SIZE,
        )
        encoded = np.eye(6, dtype=np.float32)[observation].reshape(1, -1)
        observation_tensor, _ = self.model.policy.obs_to_tensor(encoded)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(observation_tensor)
            logits = distribution.distribution.logits[0] / temperature
            probabilities = torch.softmax(logits, dim=0)
        values = tuple(float(value) for value in probabilities.cpu().tolist())
        action = (
            max(range(len(values)), key=values.__getitem__)
            if greedy
            else rng.choices(range(len(values)), weights=values, k=1)[0]
        )
        return ActionDecision(action=action, probabilities=values)


def load_policy(path: Path | str) -> Policy:
    """Load either checkpoint format through the common showcase interface."""

    checkpoint_path = Path(path)
    if checkpoint_path.suffix == ".pt":
        return load_torch_policy(checkpoint_path)
    if checkpoint_path.suffix == ".zip":
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:  # pragma: no cover - optional install state
            raise ImportError(
                "Loading SB3 checkpoints requires stable-baselines3"
            ) from exc
        return SB3Policy(PPO.load(checkpoint_path, device="cpu"))
    raise ValueError(f"unsupported PPO checkpoint format: {checkpoint_path.suffix}")


def discover_checkpoints(root: Path | str) -> list[CheckpointInfo]:
    """Discover compatible best/final PPO checkpoints and inspect metadata."""

    artifact_root = Path(root)
    if not artifact_root.exists():
        return []
    custom_checkpoints = [
        _inspect_checkpoint(path, artifact_root)
        for path in artifact_root.rglob("*.pt")
        if path.name in {"best_model.pt", "final_model.pt"}
    ]
    sb3_checkpoints = [
        _inspect_sb3_checkpoint(path, artifact_root)
        for path in artifact_root.rglob("*.zip")
        if path.name in {"best_model.zip", "final_model.zip"}
    ]
    return sorted(custom_checkpoints + sb3_checkpoints, key=_checkpoint_rank)


def _checkpoint_rank(checkpoint: CheckpointInfo) -> tuple:
    """Rank checkpoints by showcase usefulness rather than filename convention."""

    path_text = checkpoint.path.as_posix()
    return (
        not checkpoint.compatible,
        "depth_1_10_onehot" not in path_text,
        -checkpoint.validated_depth,
        -checkpoint.checkpoint_curriculum_depth,
        -checkpoint.run_curriculum_depth,
        -checkpoint.timesteps,
        checkpoint.path.name not in {"final_model.pt", "final_model.zip"},
        checkpoint.label,
    )


def _inspect_checkpoint(path: Path, root: Path) -> CheckpointInfo:
    compatible = True
    error: str | None = None
    validated_depth = 0
    checkpoint_depth = 0
    run_depth = 0
    encoding = "unknown"
    timesteps = 0
    try:
        checkpoint = torch.load(path, map_location="cpu")
        network = checkpoint.get("network_config", {})
        encoding = str(network.get("observation_encoding", "normalized"))
        expected_input = 324 if encoding == "one_hot" else 54
        if int(network.get("actor_output_dim", -1)) != ACTION_SIZE:
            raise ValueError("checkpoint actor does not expose 12 cube moves")
        if int(network.get("input_dim", -1)) != expected_input:
            raise ValueError("checkpoint input shape does not match its encoding")

        metadata = checkpoint.get("metadata", {})
        timesteps = int(metadata.get("timesteps", 0))
        evaluation = metadata.get("evaluation", {}).get("by_depth", {})
        evaluated = [int(depth) for depth in evaluation]
        validated_depth = max(evaluated, default=0)
        curriculum = metadata.get("curriculum", {})
        checkpoint_depth = int(curriculum.get("current_depth", validated_depth))
        run_depth = _read_run_curriculum_depth(path.parent, checkpoint_depth)
    except Exception as exc:  # local artifacts may be incomplete or legacy
        compatible = False
        error = str(exc)

    relative = path.relative_to(root).as_posix()
    return CheckpointInfo(
        path=path.resolve(),
        label=relative,
        compatible=compatible,
        error=error,
        validated_depth=validated_depth,
        checkpoint_curriculum_depth=checkpoint_depth,
        run_curriculum_depth=run_depth,
        observation_encoding=encoding,
        timesteps=timesteps,
    )


def _inspect_sb3_checkpoint(path: Path, root: Path) -> CheckpointInfo:
    """Inspect an SB3 sidecar without loading the comparatively large archive."""

    compatible = True
    error: str | None = None
    validated_depth = checkpoint_depth = run_depth = timesteps = 0
    try:
        sidecar = path.with_suffix(".metadata.json")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("trainer") != "stable_baselines3":
            raise ValueError("missing Stable-Baselines3 checkpoint metadata")
        timesteps = int(metadata.get("timesteps", 0))
        evaluated = [
            int(depth) for depth in metadata.get("evaluation", {}).get("by_depth", {})
        ]
        validated_depth = max(evaluated, default=0)
        checkpoint_depth = int(
            metadata.get("curriculum", {}).get("current_depth", validated_depth)
        )
        run_depth = _read_run_curriculum_depth(path.parent, checkpoint_depth)
    except Exception as exc:
        compatible = False
        error = str(exc)
    return CheckpointInfo(
        path=path.resolve(),
        label=path.relative_to(root).as_posix(),
        compatible=compatible,
        error=error,
        validated_depth=validated_depth,
        checkpoint_curriculum_depth=checkpoint_depth,
        run_curriculum_depth=run_depth,
        observation_encoding="one_hot",
        timesteps=timesteps,
        trainer="stable_baselines3",
    )


def _read_run_curriculum_depth(directory: Path, fallback: int) -> int:
    progress_path = directory / "curriculum_progress.json"
    if not progress_path.exists():
        return fallback
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
        return int(payload.get("current_depth", fallback))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return fallback


def load_curriculum_weights(path: Path | str) -> dict[int, dict[int, float]]:
    """Load and validate mixed curriculum weights by curriculum stage."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    raw_weights = payload["curriculum"]["mixed_sampling_weights"]
    result: dict[int, dict[int, float]] = {}
    for stage, weights in raw_weights.items():
        normalized = {int(depth): float(weight) for depth, weight in weights.items()}
        total = sum(normalized.values())
        if total <= 0:
            raise ValueError(f"curriculum stage {stage} has no positive weight")
        result[int(stage)] = {
            depth: weight / total for depth, weight in normalized.items()
        }
    return result


def sample_scramble(
    *,
    mode: ScrambleMode,
    minimum: int,
    maximum: int,
    seed: int,
    curriculum_weights: Mapping[int, Mapping[int, float]] | None = None,
    curriculum_stage: int | None = None,
) -> tuple[int, str, str]:
    """Generate a seeded scramble and return length, notation, and state."""

    if minimum < 1 or maximum > 10 or minimum > maximum:
        raise ValueError("scramble length must be within 1..10")
    rng = random.Random(seed)
    if mode == "exact":
        length = minimum
    elif mode == "range":
        length = rng.randint(minimum, maximum)
    elif mode == "curriculum":
        if not curriculum_weights:
            raise ValueError("curriculum weights are required")
        stage = curriculum_stage or maximum
        if stage not in curriculum_weights:
            raise ValueError(f"no curriculum weights for stage {stage}")
        filtered = [
            (depth, weight)
            for depth, weight in curriculum_weights[stage].items()
            if minimum <= depth <= maximum and weight > 0
        ]
        if not filtered:
            raise ValueError("selected range has no curriculum sampling weight")
        lengths, weights = zip(*filtered, strict=True)
        length = rng.choices(lengths, weights=weights, k=1)[0]
    else:
        raise ValueError(f"unknown scramble mode: {mode}")

    env = CubeEnvironment(rng=rng)
    scramble = env.scramble(length)
    return length, scramble, env.get_state()


def run_solver(
    policy: Policy,
    *,
    scramble_length: int,
    scramble: str,
    config: RunConfig,
    clock: Callable[[], float] = perf_counter,
) -> SolveResult:
    """Run greedy first, followed by reproducible stochastic attempts."""

    maximum_moves = config.max_moves or (2 * scramble_length + 1)
    initial_env = _environment_for_scramble(scramble, scramble_length, maximum_moves)
    initial_state = initial_env.get_state()
    started = clock()
    deadline = started + config.timeout_seconds
    attempts: list[AttemptResult] = []
    total_inference = 0.0

    for attempt_number in range(1, config.max_attempts + 1):
        if attempt_number > 1 and clock() >= deadline:
            break
        env = _environment_for_scramble(scramble, scramble_length, maximum_moves)
        attempt_started = clock()
        inference_seconds = 0.0
        moves: list[str] = []
        frames = [SolveFrame(state=env.get_state(), step=0)]
        seen_states = {env.get_state()}
        reason: TerminationReason = "move_limit"
        solved = env.is_solved()
        greedy = attempt_number == 1
        rng = random.Random(config.seed + attempt_number * 1_000_003)

        if solved:
            reason = "solved"
        while not solved and len(moves) < maximum_moves:
            if clock() >= deadline:
                reason = "time_limit"
                break
            inference_started = clock()
            decision = policy.decide(
                env.get_state(),
                greedy=greedy,
                temperature=config.temperature,
                rng=rng,
            )
            inference_seconds += clock() - inference_started
            if decision.action not in ACTION_TO_MOVE:
                raise ValueError(f"policy selected invalid action {decision.action}")
            if len(decision.probabilities) != ACTION_SIZE:
                raise ValueError("policy must return probabilities for all 12 actions")

            next_state, _, done, info = env.step(decision.action)
            moves.append(decision.move)
            frames.append(
                SolveFrame(
                    state=next_state,
                    step=len(moves),
                    move=decision.move,
                    decision=decision,
                )
            )
            solved = bool(info["is_solved"])
            if solved:
                reason = "solved"
                break
            if next_state in seen_states:
                reason = "cycle"
                break
            seen_states.add(next_state)
            if done:
                reason = "move_limit"
                break

        attempt_result = AttemptResult(
            attempt=attempt_number,
            mode="greedy" if greedy else "stochastic",
            solved=solved,
            termination_reason=reason,
            moves=tuple(moves),
            frames=tuple(frames),
            inference_seconds=inference_seconds,
            solver_seconds=clock() - attempt_started,
        )
        attempts.append(attempt_result)
        total_inference += inference_seconds
        if solved or reason == "time_limit":
            break

    solver_seconds = clock() - started
    solved = any(attempt.solved for attempt in attempts)
    if solved:
        final_reason: TerminationReason = "solved"
    elif solver_seconds >= config.timeout_seconds or (
        attempts and attempts[-1].termination_reason == "time_limit"
    ):
        final_reason = "time_limit"
    else:
        final_reason = "attempt_limit"
    return SolveResult(
        scramble_length=scramble_length,
        scramble=scramble,
        initial_state=initial_state,
        seed=config.seed,
        solved=solved,
        greedy_solved=bool(attempts and attempts[0].solved),
        termination_reason=final_reason,
        attempts=tuple(attempts),
        inference_seconds=total_inference,
        solver_seconds=solver_seconds,
    )


def _environment_for_scramble(
    scramble: str,
    scramble_length: int,
    max_moves: int,
) -> CubeEnvironment:
    env = CubeEnvironment(max_steps=max_moves)
    cube = CubeState.solved()
    for move in scramble.split():
        cube = apply_move(cube, move)
    env.cube = cube
    env.scramble_depth = scramble_length
    env.scramble_sequence = scramble
    env.max_steps = max_moves
    return env


def run_benchmark(
    policy: Policy,
    *,
    lengths: Sequence[int],
    trials_per_length: int,
    config: RunConfig,
    progress: Callable[[int, int], None] | None = None,
) -> BenchmarkResult:
    """Run fixed-length trials and aggregate results by scramble length."""

    if trials_per_length <= 0:
        raise ValueError("trials_per_length must be positive")
    selected = sorted(set(int(length) for length in lengths))
    if not selected or selected[0] < 1 or selected[-1] > 10:
        raise ValueError("benchmark lengths must be within 1..10")

    records: list[dict[str, object]] = []
    total = len(selected) * trials_per_length
    completed = 0
    for length in selected:
        for trial in range(trials_per_length):
            seed = config.seed + length * 100_000 + trial
            _, scramble, _ = sample_scramble(
                mode="exact",
                minimum=length,
                maximum=length,
                seed=seed,
            )
            trial_config = RunConfig(
                max_moves=config.max_moves,
                max_attempts=config.max_attempts,
                timeout_seconds=config.timeout_seconds,
                temperature=config.temperature,
                seed=seed,
            )
            result = run_solver(
                policy,
                scramble_length=length,
                scramble=scramble,
                config=trial_config,
            )
            oracle = inverse_algorithm(scramble)
            oracle_solved = _sequence_solves(scramble, oracle)
            records.append(
                {
                    "scramble_length": length,
                    "trial": trial + 1,
                    "seed": seed,
                    "scramble": scramble,
                    "solved": result.solved,
                    "greedy_solved": result.greedy_solved,
                    "termination_reason": result.termination_reason,
                    "attempts_used": result.attempts_used,
                    "total_moves": result.total_moves,
                    "solution_length": (
                        len(result.display_attempt.moves) if result.solved else None
                    ),
                    "solution_moves": " ".join(result.display_attempt.moves),
                    "cycle": any(
                        attempt.termination_reason == "cycle"
                        for attempt in result.attempts
                    ),
                    "move_limit": any(
                        attempt.termination_reason == "move_limit"
                        for attempt in result.attempts
                    ),
                    "inference_seconds": result.inference_seconds,
                    "solver_seconds": result.solver_seconds,
                    "oracle_moves": oracle,
                    "oracle_length": len(oracle.split()),
                    "oracle_solved": oracle_solved,
                }
            )
            completed += 1
            if progress is not None:
                progress(completed, total)
    return BenchmarkResult(
        records=tuple(records),
        summary=tuple(_summarize_benchmark(records, selected)),
    )


def _summarize_benchmark(
    records: Sequence[dict[str, object]],
    lengths: Sequence[int],
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for length in lengths:
        rows = [row for row in records if row["scramble_length"] == length]
        solved = [row for row in rows if bool(row["solved"])]
        summary.append(
            {
                "scramble_length": length,
                "trials": len(rows),
                "greedy_solve_rate": _mean_bool(rows, "greedy_solved"),
                "retry_solve_rate": _mean_bool(rows, "solved"),
                "average_solution_length": (
                    sum(int(row["solution_length"]) for row in solved) / len(solved)
                    if solved
                    else None
                ),
                "average_total_moves": (
                    sum(int(row["total_moves"]) for row in rows) / len(rows)
                ),
                "average_attempts": (
                    sum(int(row["attempts_used"]) for row in rows) / len(rows)
                ),
                "cycle_rate": _mean_bool(rows, "cycle"),
                "move_limit_rate": _mean_bool(rows, "move_limit"),
                "time_limit_rate": (
                    sum(row["termination_reason"] == "time_limit" for row in rows)
                    / len(rows)
                ),
                "limit_failure_rate": (
                    sum(
                        row["termination_reason"] in {"attempt_limit", "time_limit"}
                        for row in rows
                    )
                    / len(rows)
                ),
                "average_inference_ms": (
                    1000
                    * sum(float(row["inference_seconds"]) for row in rows)
                    / len(rows)
                ),
                "average_solver_ms": (
                    1000 * sum(float(row["solver_seconds"]) for row in rows) / len(rows)
                ),
                "oracle_length": length,
                "oracle_solve_rate": _mean_bool(rows, "oracle_solved"),
            }
        )
    return summary


def _mean_bool(rows: Sequence[dict[str, object]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows)


def _sequence_solves(scramble: str, solution: str) -> bool:
    cube = CubeState.solved()
    for move in (*scramble.split(), *solution.split()):
        cube = apply_move(cube, move)
    return cube.is_solved()


def benchmark_to_csv(
    benchmark: BenchmarkResult,
    *,
    summary: bool = False,
) -> str:
    """Serialize detailed or summary benchmark rows to CSV."""

    rows = benchmark.summary if summary else benchmark.records
    if not rows:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def solve_result_record(result: SolveResult, checkpoint: str) -> dict[str, object]:
    """Flatten one demo result for session history and CSV export."""

    return {
        "checkpoint": checkpoint,
        "seed": result.seed,
        "scramble_length": result.scramble_length,
        "scramble": result.scramble,
        "solved": result.solved,
        "greedy_solved": result.greedy_solved,
        "termination_reason": result.termination_reason,
        "attempts_used": result.attempts_used,
        "total_moves": result.total_moves,
        "display_moves": " ".join(result.display_attempt.moves),
        "inference_seconds": result.inference_seconds,
        "solver_seconds": result.solver_seconds,
    }


def records_to_csv(rows: Sequence[Mapping[str, object]]) -> str:
    """Serialize homogeneous session records to CSV."""

    if not rows:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()
