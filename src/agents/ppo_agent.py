"""Custom PyTorch PPO agent for the Gymnasium Rubik's Cube environment."""

from __future__ import annotations

import csv
import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Literal, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from agents.behavior_logging import BehaviorLogger, utc_timestamp
from cube.environment import ACTION_TO_MOVE
from cube.gym_environment import (
    DEFAULT_EXHAUSTIVE_STATE_THRESHOLD,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_TRAINING_DATA_DIR,
    RubixCubeSolveEnv,
    encode_state,
)
from curriculum.manager import CurriculumConfig, CurriculumManager


DEFAULT_OUTPUT_DIR = Path("models/artifacts/ppo")
DEFAULT_TOTAL_TIMESTEPS = 500_000
DEFAULT_EVAL_FREQUENCY = 10_000
DEFAULT_EVAL_EPISODES = 100
EXHAUSTIVE_STATE_THRESHOLD = DEFAULT_EXHAUSTIVE_STATE_THRESHOLD
EXHAUSTIVE_EVAL_STATE_THRESHOLD = EXHAUSTIVE_STATE_THRESHOLD
OBSERVATION_SIZE = 54
COLOR_COUNT = 6
ONE_HOT_OBSERVATION_SIZE = OBSERVATION_SIZE * COLOR_COUNT
ACTION_SIZE = 12
CHECKPOINT_VERSION = 2
ObservationEncoding = Literal["one_hot", "normalized"]


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters for the custom PPO trainer."""

    learning_rate: float = 3e-4
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    n_epochs: int = 10
    n_steps: int = 2048
    batch_size: int = 64
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03


@dataclass(frozen=True)
class NetworkConfig:
    """Actor-critic MLP shape."""

    input_dim: int = ONE_HOT_OBSERVATION_SIZE
    hidden_layers: tuple[int, ...] = (256, 256)
    actor_output_dim: int = ACTION_SIZE
    critic_output_dim: int = 1
    observation_encoding: ObservationEncoding = "one_hot"

    def __post_init__(self) -> None:
        if self.observation_encoding not in ("one_hot", "normalized"):
            raise ValueError("observation_encoding must be 'one_hot' or 'normalized'")


class ActorCriticNet(nn.Module):
    """Independent actor and critic MLPs."""

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetworkConfig()
        self.actor = _build_mlp(
            input_dim=self.config.input_dim,
            hidden_layers=self.config.hidden_layers,
            output_dim=self.config.actor_output_dim,
        )
        self.critic = _build_mlp(
            input_dim=self.config.input_dim,
            hidden_layers=self.config.hidden_layers,
            output_dim=self.config.critic_output_dim,
        )

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(observations)
        values = self.critic(observations).squeeze(-1)
        return logits, values

    def actor_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return actor parameters for branch-specific optimization controls."""

        return tuple(self.actor.parameters())

    def critic_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return critic parameters for branch-specific optimization controls."""

        return tuple(self.critic.parameters())


def _build_mlp(
    *,
    input_dim: int,
    hidden_layers: tuple[int, ...],
    output_dim: int,
) -> nn.Sequential:
    """Build one actor or critic MLP."""

    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(previous_dim, hidden_dim))
        layers.append(nn.ReLU())
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


class RolloutBuffer:
    """In-memory rollout storage for one PPO update."""

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.observations: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.depths: list[int | None] = []
        self.infos: list[dict[str, Any]] = []
        self.returns: torch.Tensor | None = None
        self.advantages: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.actions)

    def add(
        self,
        *,
        observation: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        depth: int | None,
        info: Mapping[str, Any],
    ) -> None:
        self.observations.append(np.asarray(observation, dtype=np.int8).copy())
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.depths.append(None if depth is None else int(depth))
        self.infos.append(dict(info))

    def compute_returns_and_advantages(
        self,
        *,
        last_value: float,
        last_done: bool,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and lambda returns for the stored rollout."""

        advantages = np.zeros(len(self), dtype=np.float32)
        last_gae = 0.0
        for step in reversed(range(len(self))):
            if step == len(self) - 1:
                next_value = float(last_value)
                current_done = last_done
            else:
                next_value = self.values[step + 1]
                current_done = self.dones[step]

            next_non_terminal = 0.0 if current_done else 1.0
            delta = (
                self.rewards[step]
                + gamma * next_value * next_non_terminal
                - self.values[step]
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[step] = last_gae

        values = np.asarray(self.values, dtype=np.float32)
        returns = advantages + values
        self.advantages = torch.tensor(advantages, dtype=torch.float32)
        self.returns = torch.tensor(returns, dtype=torch.float32)
        return self.returns, self.advantages

    def as_tensors(
        self,
        device: torch.device,
        observation_encoding: ObservationEncoding = "one_hot",
    ) -> dict[str, torch.Tensor]:
        if self.returns is None or self.advantages is None:
            raise ValueError("returns and advantages must be computed before batching")

        return {
            "observations": preprocess_observations(
                self.observations,
                device,
                observation_encoding=observation_encoding,
            ),
            "actions": torch.tensor(self.actions, dtype=torch.long, device=device),
            "old_log_probs": torch.tensor(
                self.log_probs, dtype=torch.float32, device=device
            ),
            "returns": self.returns.to(device),
            "advantages": self.advantages.to(device),
            "values": torch.tensor(self.values, dtype=torch.float32, device=device),
        }


def set_random_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_files_for_depths(
    min_depth: int,
    max_depth: int,
    data_dir: Path | str = DEFAULT_TRAINING_DATA_DIR,
) -> dict[int, Path]:
    """Return depth_N data paths, preferring parquet and falling back to CSV."""

    if min_depth <= 0:
        raise ValueError("min_depth must be positive")
    if max_depth < min_depth:
        raise ValueError("max_depth cannot be less than min_depth")

    root = Path(data_dir)
    state_files: dict[int, Path] = {}
    for depth in range(min_depth, max_depth + 1):
        parquet_path = root / f"depth_{depth}.parquet"
        csv_path = root / f"depth_{depth}.csv"
        state_files[depth] = csv_path if csv_path.exists() and not parquet_path.exists() else parquet_path
    return state_files


def make_training_env(
    *,
    min_depth: int = 1,
    max_depth: int = 8,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    data_dir: Path | str = DEFAULT_TRAINING_DATA_DIR,
    validate_dataset: bool = True,
    curriculum_config: CurriculumConfig | None = None,
    curriculum_manager: CurriculumManager | None = None,
    seed: int | None = None,
) -> RubixCubeSolveEnv:
    """Create a dataset-driven uniform or curriculum training environment."""

    if curriculum_manager is None and curriculum_config is not None:
        curriculum_manager = CurriculumManager(
            state_files_for_depths(
                curriculum_config.min_depth,
                curriculum_config.max_depth,
                data_dir,
            ),
            curriculum_config,
            exhaustive_state_threshold=EXHAUSTIVE_STATE_THRESHOLD,
            validate_dataset=validate_dataset,
            seed=seed,
        )
    state_files = state_files_for_depths(min_depth, max_depth, data_dir)
    if curriculum_manager is not None:
        state_files = curriculum_manager.depth_files
    depth_range = (min_depth, max_depth) if min_depth != max_depth else None
    return RubixCubeSolveEnv(
        state_files=state_files,
        scramble_depth=min_depth if depth_range is None else None,
        scramble_depth_range=depth_range,
        max_episode_steps=max_episode_steps,
        validate_dataset=validate_dataset,
        exhaustive_state_threshold=EXHAUSTIVE_STATE_THRESHOLD,
        curriculum_manager=curriculum_manager,
    )


def make_eval_env(
    *,
    depth: int,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    data_dir: Path | str = DEFAULT_TRAINING_DATA_DIR,
    validate_dataset: bool = True,
    state_data: Mapping[int, Any] | None = None,
) -> RubixCubeSolveEnv:
    """Create a fixed-depth evaluation environment."""

    return RubixCubeSolveEnv(
        state_files=state_files_for_depths(depth, depth, data_dir),
        scramble_depth=depth,
        scramble_depth_range=None,
        max_episode_steps=max_episode_steps,
        validate_dataset=validate_dataset,
        exhaustive_state_threshold=EXHAUSTIVE_STATE_THRESHOLD,
        state_data=state_data,
    )


def preprocess_observations(
    observations: np.ndarray | list[np.ndarray],
    device: torch.device | str | None = None,
    observation_encoding: ObservationEncoding = "one_hot",
) -> torch.Tensor:
    """Convert cube observations to model-ready float tensors."""

    if observation_encoding not in ("one_hot", "normalized"):
        raise ValueError("observation_encoding must be 'one_hot' or 'normalized'")

    array = np.asarray(observations)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[-1] != OBSERVATION_SIZE:
        raise ValueError(f"observations must have {OBSERVATION_SIZE} stickers")

    if observation_encoding == "one_hot":
        sticker_tensor = torch.as_tensor(array, dtype=torch.long)
        tensor = F.one_hot(sticker_tensor, num_classes=COLOR_COUNT).to(torch.float32)
        tensor = tensor.reshape(tensor.shape[0], ONE_HOT_OBSERVATION_SIZE)
    else:
        tensor = torch.as_tensor(array, dtype=torch.float32) / float(COLOR_COUNT - 1)

    if device is not None:
        tensor = tensor.to(device)
    return tensor


def select_action(
    model: ActorCriticNet,
    observation: np.ndarray,
    device: torch.device | str | None = None,
) -> tuple[int, float, float]:
    """Sample a training action and return action, log prob, and value."""

    run_device = torch.device(device or "cpu")
    model.eval()
    with torch.no_grad():
        logits, values = model(
            preprocess_observations(
                observation,
                run_device,
                observation_encoding=model.config.observation_encoding,
            )
        )
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
    return int(action.item()), float(log_prob.item()), float(values.squeeze(0).item())


def predict_action(
    model: ActorCriticNet,
    observation: np.ndarray,
    device: torch.device | str | None = None,
) -> int:
    """Choose the greedy action for deterministic evaluation."""

    run_device = torch.device(device or "cpu")
    model.eval()
    with torch.no_grad():
        logits, _ = model(
            preprocess_observations(
                observation,
                run_device,
                observation_encoding=model.config.observation_encoding,
            )
        )
    return int(torch.argmax(logits, dim=1).item())


def evaluate_actions(
    model: ActorCriticNet,
    observations: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate old rollout actions under the current policy."""

    logits, values = model(observations)
    distribution = Categorical(logits=logits)
    return distribution.log_prob(actions), distribution.entropy(), values


def collect_rollout(
    *,
    model: ActorCriticNet,
    env: RubixCubeSolveEnv,
    config: PPOConfig,
    device: torch.device,
    observation: np.ndarray | None = None,
    info: dict[str, Any] | None = None,
) -> tuple[RolloutBuffer, np.ndarray, dict[str, Any], bool, list[dict[str, Any]]]:
    """Collect one fixed-size PPO rollout from a single environment."""

    buffer = RolloutBuffer()
    if observation is None:
        observation, info = env.reset()
    if info is None:
        info = {}

    done = False
    completed_episodes: list[dict[str, Any]] = []
    episode_reward = 0.0
    episode_length = 0
    episode_inverse_moves = 0

    for _ in range(config.n_steps):
        current_observation = observation.copy()
        action, log_prob, value = select_action(model, current_observation, device)
        next_observation, reward, terminated, truncated, next_info = env.step(action)
        done = bool(terminated or truncated)

        episode_reward += float(reward)
        episode_length += 1
        episode_inverse_moves += int(next_info.get("immediate_inverse_move", False))
        buffer.add(
            observation=current_observation,
            action=action,
            reward=float(reward),
            done=done,
            log_prob=log_prob,
            value=value,
            depth=next_info.get("start_depth", info.get("start_depth")),
            info=next_info,
        )

        observation = next_observation
        info = next_info
        if done:
            completed_episodes.append(
                {
                    "reward": episode_reward,
                    "length": episode_length,
                    "solved": bool(next_info.get("is_solved", False)),
                    "timeout": next_info.get("terminated_reason") == "max_steps_reached",
                    "inverse_moves": episode_inverse_moves,
                    "start_depth": next_info.get("start_depth"),
                }
            )
            observation, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
            episode_inverse_moves = 0

    last_value = 0.0
    if not done:
        with torch.no_grad():
            _, values = model(
                preprocess_observations(
                    observation,
                    device,
                    observation_encoding=model.config.observation_encoding,
                )
            )
            last_value = float(values.squeeze(0).item())

    buffer.compute_returns_and_advantages(
        last_value=last_value,
        last_done=done,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    return buffer, observation, info, done, completed_episodes


def ppo_update(
    *,
    model: ActorCriticNet,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    config: PPOConfig,
    device: torch.device,
) -> dict[str, float]:
    """Run PPO optimization epochs over a completed rollout."""

    model.train()
    tensors = buffer.as_tensors(
        device,
        observation_encoding=model.config.observation_encoding,
    )
    advantages = tensors["advantages"]
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    sample_count = len(buffer)
    batch_size = min(config.batch_size, sample_count)
    metrics: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
        "loss": [],
    }

    for _ in range(config.n_epochs):
        permutation = torch.randperm(sample_count, device=device)
        for start in range(0, sample_count, batch_size):
            indices = permutation[start : start + batch_size]
            new_log_probs, entropy, new_values = evaluate_actions(
                model,
                tensors["observations"][indices],
                tensors["actions"][indices],
            )
            old_log_probs = tensors["old_log_probs"][indices]
            log_ratio = new_log_probs - old_log_probs
            ratio = torch.exp(log_ratio)
            batch_advantages = advantages[indices]

            unclipped = ratio * batch_advantages
            clipped = (
                torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range)
                * batch_advantages
            )
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = (tensors["returns"][indices] - new_values).pow(2).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.actor_parameters(),
                config.max_grad_norm,
            )
            nn.utils.clip_grad_norm_(
                model.critic_parameters(),
                config.max_grad_norm,
            )
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > config.clip_range)
                    .float()
                    .mean()
                )

            metrics["policy_loss"].append(float(policy_loss.item()))
            metrics["value_loss"].append(float(value_loss.item()))
            metrics["entropy"].append(float(entropy_loss.item()))
            metrics["approx_kl"].append(float(approx_kl.item()))
            metrics["clip_fraction"].append(float(clip_fraction.item()))
            metrics["loss"].append(float(loss.item()))

        if config.target_kl is not None and metrics["approx_kl"]:
            if metrics["approx_kl"][-1] > 1.5 * config.target_kl:
                break

    result = {key: mean(values) if values else 0.0 for key, values in metrics.items()}
    result["explained_variance"] = explained_variance(
        tensors["values"].detach().cpu().numpy(),
        tensors["returns"].detach().cpu().numpy(),
    )
    return result


def train_ppo(
    *,
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    min_depth: int = 1,
    max_depth: int = 10,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    data_dir: Path | str = DEFAULT_TRAINING_DATA_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
    device: torch.device | str | None = None,
    eval_frequency: int = DEFAULT_EVAL_FREQUENCY,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    config: PPOConfig | None = None,
    network_config: NetworkConfig | None = None,
    output_metadata: Mapping[str, Any] | None = None,
    validate_dataset: bool = True,
    curriculum_config: CurriculumConfig | None = None,
    curriculum_manager: CurriculumManager | None = None,
) -> dict[str, Any]:
    """Train a custom PPO agent and save final/best checkpoints."""

    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if eval_frequency <= 0:
        raise ValueError("eval_frequency must be positive")
    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive")

    training_started_at = utc_timestamp()
    training_start_time = perf_counter()
    ppo_config = config or PPOConfig()
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_random_seeds(seed)

    if curriculum_manager is None and curriculum_config is not None:
        curriculum_manager = CurriculumManager(
            state_files_for_depths(
                curriculum_config.min_depth,
                curriculum_config.max_depth,
                data_dir,
            ),
            curriculum_config,
            exhaustive_state_threshold=EXHAUSTIVE_STATE_THRESHOLD,
            validate_dataset=validate_dataset,
            seed=seed,
        )
    if curriculum_manager is not None:
        curriculum_config = curriculum_manager.config
        training_min_depth = curriculum_config.min_depth
        training_max_depth = curriculum_config.max_depth
    else:
        training_min_depth = min_depth
        training_max_depth = max_depth

    env = make_training_env(
        min_depth=training_min_depth,
        max_depth=training_max_depth,
        max_episode_steps=max_episode_steps,
        data_dir=data_dir,
        validate_dataset=validate_dataset,
        curriculum_manager=curriculum_manager,
    )
    env.action_space.seed(seed)
    observation, info = env.reset(seed=None)

    model = ActorCriticNet(network_config).to(run_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_config.learning_rate)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_solve_rate = -1.0
    best_curriculum_depth = 0
    best_timeout_rate = float("inf")
    best_by_depth: dict[int, dict[str, float | int]] = {}
    total_steps = 0
    next_eval_step = eval_frequency
    recent_episodes: deque[dict[str, Any]] = deque(maxlen=100)
    curriculum_evaluations: list[dict[str, Any]] = []

    random_baseline = evaluate_random_baseline(
        depths=range(training_min_depth, training_max_depth + 1),
        episodes_per_depth=max(1, min(eval_episodes, 20)),
        max_episode_steps=max_episode_steps,
        data_dir=data_dir,
        seed=seed,
        validate_dataset=validate_dataset,
        state_data=(
            curriculum_manager.depth_data
            if curriculum_manager is not None
            else None
        ),
    )

    while total_steps < total_timesteps:
        rollout, observation, info, _, episodes = collect_rollout(
            model=model,
            env=env,
            config=ppo_config,
            device=run_device,
            observation=observation,
            info=info,
        )
        total_steps += len(rollout)
        recent_episodes.extend(episodes)
        update_metrics = ppo_update(
            model=model,
            optimizer=optimizer,
            buffer=rollout,
            config=ppo_config,
            device=run_device,
        )
        update_metrics.update(_episode_window_metrics(recent_episodes))

        row: dict[str, Any] = {
            "timesteps": total_steps,
            **update_metrics,
        }
        if curriculum_manager is not None:
            row.update(
                {
                    "curriculum_depth": curriculum_manager.get_current_depth(),
                    "curriculum_sampling_weights": (
                        curriculum_manager.current_weights()
                    ),
                }
            )

        if total_steps >= next_eval_step or total_steps >= total_timesteps:
            evaluation_depths: range | tuple[int, ...]
            evaluation_max_steps = max_episode_steps
            evaluated_curriculum_depth = None
            if curriculum_manager is not None and curriculum_config is not None:
                evaluated_curriculum_depth = (
                    curriculum_manager.get_current_depth()
                )
                evaluation_depths = (evaluated_curriculum_depth,)
                evaluation_max_steps = curriculum_config.max_episode_steps(
                    evaluated_curriculum_depth
                )
            else:
                evaluation_depths = range(
                    training_min_depth,
                    training_max_depth + 1,
                )
            evaluation = evaluate_ppo_model(
                model,
                depths=evaluation_depths,
                episodes_per_depth=eval_episodes,
                max_episode_steps=evaluation_max_steps,
                data_dir=data_dir,
                device=run_device,
                validate_dataset=validate_dataset,
                state_data=(
                    curriculum_manager.depth_data
                    if curriculum_manager is not None
                    else None
                ),
            )
            row["evaluation"] = evaluation
            advanced_curriculum = False
            if (
                curriculum_manager is not None
                and curriculum_config is not None
                and evaluated_curriculum_depth is not None
            ):
                depth_metrics = evaluation["by_depth"][
                    str(evaluated_curriculum_depth)
                ]
                threshold = curriculum_config.advancement_thresholds[
                    evaluated_curriculum_depth
                ]
                if curriculum_manager.should_advance(depth_metrics):
                    advanced_curriculum = curriculum_manager.increase_depth(
                        timestep=total_steps,
                        metrics=depth_metrics,
                    )
                curriculum_evaluation = {
                    "timesteps": total_steps,
                    "curriculum_depth": evaluated_curriculum_depth,
                    "metrics": depth_metrics,
                    "threshold": asdict(threshold),
                    "advanced_curriculum": advanced_curriculum,
                    "next_curriculum_depth": (
                        curriculum_manager.get_current_depth()
                    ),
                }
                curriculum_evaluations.append(curriculum_evaluation)
                row["curriculum_evaluation"] = curriculum_evaluation
                _write_json(
                    output_path / "curriculum_progress.json",
                    {
                        **curriculum_manager.progress(),
                        "evaluations": curriculum_evaluations,
                    },
                )
            solve_rate = float(evaluation["overall"]["solve_rate"])
            timeout_rate = float(evaluation["overall"]["timeout_rate"])
            checkpoint_metadata = {
                "timesteps": total_steps,
                "evaluation": evaluation,
            }
            if curriculum_manager is not None:
                checkpoint_metadata["curriculum"] = (
                    curriculum_manager.progress()
                )

            if evaluated_curriculum_depth is not None:
                depth_best = best_by_depth.get(evaluated_curriculum_depth)
                improves_depth = depth_best is None or (
                    solve_rate > float(depth_best["solve_rate"])
                    or (
                        solve_rate == float(depth_best["solve_rate"])
                        and timeout_rate < float(depth_best["timeout_rate"])
                    )
                )
                if improves_depth:
                    best_by_depth[evaluated_curriculum_depth] = {
                        "depth": evaluated_curriculum_depth,
                        "solve_rate": solve_rate,
                        "timeout_rate": timeout_rate,
                        "timesteps": total_steps,
                    }
                    save_checkpoint(
                        output_path
                        / f"best_model_depth_{evaluated_curriculum_depth}.pt",
                        model=model,
                        config=ppo_config,
                        network_config=network_config or NetworkConfig(),
                        metadata={
                            **checkpoint_metadata,
                            "selection": "best_at_curriculum_depth",
                            "evaluated_curriculum_depth": (
                                evaluated_curriculum_depth
                            ),
                        },
                    )

                improves_curriculum_best = (
                    evaluated_curriculum_depth > best_curriculum_depth
                    or (
                        evaluated_curriculum_depth == best_curriculum_depth
                        and (
                            solve_rate > best_solve_rate
                            or (
                                solve_rate == best_solve_rate
                                and timeout_rate < best_timeout_rate
                            )
                        )
                    )
                )
                if improves_curriculum_best:
                    best_curriculum_depth = evaluated_curriculum_depth
                    best_solve_rate = solve_rate
                    best_timeout_rate = timeout_rate
                    save_checkpoint(
                        output_path / "best_model.pt",
                        model=model,
                        config=ppo_config,
                        network_config=network_config or NetworkConfig(),
                        metadata={
                            **checkpoint_metadata,
                            "selection": "curriculum_lexicographic",
                            "evaluated_curriculum_depth": (
                                evaluated_curriculum_depth
                            ),
                        },
                    )

                if advanced_curriculum:
                    save_checkpoint(
                        output_path / "latest_passed_gate.pt",
                        model=model,
                        config=ppo_config,
                        network_config=network_config or NetworkConfig(),
                        metadata={
                            **checkpoint_metadata,
                            "selection": "latest_passed_curriculum_gate",
                            "passed_curriculum_depth": evaluated_curriculum_depth,
                        },
                    )
            elif solve_rate > best_solve_rate:
                best_solve_rate = solve_rate
                save_checkpoint(
                    output_path / "best_model.pt",
                    model=model,
                    config=ppo_config,
                    network_config=network_config or NetworkConfig(),
                    metadata=checkpoint_metadata,
                )
            while next_eval_step <= total_steps:
                next_eval_step += eval_frequency

        history.append(row)
        print(json.dumps(row, sort_keys=True))

    final_depths: range | tuple[int, ...] = range(
        training_min_depth,
        training_max_depth + 1,
    )
    final_max_episode_steps = max_episode_steps
    if curriculum_manager is not None and curriculum_config is not None:
        final_depth = curriculum_manager.get_current_depth()
        final_depths = (final_depth,)
        final_max_episode_steps = curriculum_config.max_episode_steps(final_depth)
    final_evaluation = evaluate_ppo_model(
        model,
        depths=final_depths,
        episodes_per_depth=eval_episodes,
        max_episode_steps=final_max_episode_steps,
        data_dir=data_dir,
        device=run_device,
        validate_dataset=validate_dataset,
        state_data=(
            curriculum_manager.depth_data
            if curriculum_manager is not None
            else None
        ),
    )
    final_checkpoint_metadata = {
        "timesteps": total_steps,
        "evaluation": final_evaluation,
    }
    if curriculum_manager is not None:
        final_checkpoint_metadata["curriculum"] = curriculum_manager.progress()
    save_checkpoint(
        output_path / "final_model.pt",
        model=model,
        config=ppo_config,
        network_config=network_config or NetworkConfig(),
        metadata=final_checkpoint_metadata,
    )
    training_session = {
        "started_at": training_started_at,
        "completed_at": utc_timestamp(),
        "elapsed_seconds": perf_counter() - training_start_time,
    }
    run_config = {
        "total_timesteps": total_timesteps,
        "actual_timesteps": total_steps,
        "output_dir": str(output_path),
        "min_depth": training_min_depth,
        "max_depth": training_max_depth,
        "max_episode_steps": max_episode_steps,
        "data_dir": str(data_dir),
        "seed": seed,
        "device": str(run_device),
        "ppo": asdict(ppo_config),
        "network": _network_config_dict(network_config or NetworkConfig()),
        "curriculum": (
            curriculum_config.to_dict()
            if curriculum_config is not None
            else None
        ),
        "training_session": training_session,
    }
    if output_metadata:
        run_config.update(dict(output_metadata))
    persisted_metrics = {
        "training_session": training_session,
        "final_evaluation": final_evaluation,
        "random_baseline": random_baseline,
        "best_solve_rate": best_solve_rate,
        "checkpoint_selection": (
            {
                "strategy": "curriculum_lexicographic",
                "best_curriculum_depth": best_curriculum_depth,
                "best_solve_rate": best_solve_rate,
                "best_timeout_rate": best_timeout_rate,
                "best_by_depth": {
                    str(depth): values
                    for depth, values in sorted(best_by_depth.items())
                },
            }
            if curriculum_manager is not None
            else {
                "strategy": "highest_overall_solve_rate",
                "best_solve_rate": best_solve_rate,
            }
        ),
        "curriculum_progress": (
            {
                **curriculum_manager.progress(),
                "evaluations": curriculum_evaluations,
            }
            if curriculum_manager is not None
            else None
        ),
    }
    metrics = {
        "history": history,
        **persisted_metrics,
    }
    _write_json(output_path / "config.json", run_config)
    write_history_csv(output_path / "history.csv", history)
    _write_json(output_path / "metrics.json", persisted_metrics)
    return {
        "model": model,
        "config": run_config,
        "metrics": metrics,
        "output_dir": output_path,
    }


def evaluate_ppo_model(
    model: Any,
    *,
    depths: range | list[int] | tuple[int, ...],
    episodes_per_depth: int = DEFAULT_EVAL_EPISODES,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    data_dir: Path | str = DEFAULT_TRAINING_DATA_DIR,
    device: torch.device | str | None = None,
    validate_dataset: bool = True,
    behavior_logger: BehaviorLogger | None = None,
    state_data: Mapping[int, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a deterministic PPO policy by scramble depth."""

    return _evaluate_policy(
        model,
        depths=depths,
        episodes_per_depth=episodes_per_depth,
        max_episode_steps=max_episode_steps,
        data_dir=data_dir,
        seed=None,
        device=device,
        random_policy=False,
        validate_dataset=validate_dataset,
        behavior_logger=behavior_logger,
        state_data=state_data,
    )


def evaluate_random_baseline(
    *,
    depths: range | list[int] | tuple[int, ...],
    episodes_per_depth: int = DEFAULT_EVAL_EPISODES,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    data_dir: Path | str = DEFAULT_TRAINING_DATA_DIR,
    seed: int = 0,
    validate_dataset: bool = True,
    state_data: Mapping[int, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate uniformly random actions on the same Gymnasium environment."""

    return _evaluate_policy(
        None,
        depths=depths,
        episodes_per_depth=episodes_per_depth,
        max_episode_steps=max_episode_steps,
        data_dir=data_dir,
        seed=seed,
        random_policy=True,
        validate_dataset=validate_dataset,
        behavior_logger=None,
        state_data=state_data,
    )


def save_checkpoint(
    path: Path | str,
    *,
    model: ActorCriticNet,
    config: PPOConfig,
    network_config: NetworkConfig,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save a PPO model checkpoint."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "model_state_dict": model.state_dict(),
            "ppo_config": asdict(config),
            "network_config": _network_config_dict(network_config),
            "metadata": dict(metadata or {}),
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_checkpoint(
    path: Path | str,
    *,
    device: torch.device | str | None = None,
) -> tuple[ActorCriticNet, dict[str, Any]]:
    """Load a saved PPO model checkpoint."""

    run_device = torch.device(device or "cpu")
    checkpoint = torch.load(Path(path), map_location=run_device)
    network_config = _network_config_from_dict(checkpoint.get("network_config", {}))
    model = ActorCriticNet(network_config).to(run_device)
    state_dict = checkpoint["model_state_dict"]
    if int(checkpoint.get("checkpoint_version", 1)) < CHECKPOINT_VERSION:
        state_dict = _migrate_legacy_shared_state_dict(
            state_dict,
            network_config=network_config,
        )
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def _migrate_legacy_shared_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    network_config: NetworkConfig,
) -> dict[str, torch.Tensor]:
    """Convert a shared-trunk checkpoint into independent actor/critic towers."""

    output_layer_index = 2 * len(network_config.hidden_layers)
    migrated: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("shared."):
            suffix = key.removeprefix("shared.")
            migrated[f"actor.{suffix}"] = value
            migrated[f"critic.{suffix}"] = value
        elif key.startswith("actor."):
            suffix = key.removeprefix("actor.")
            migrated[f"actor.{output_layer_index}.{suffix}"] = value
        elif key.startswith("critic."):
            suffix = key.removeprefix("critic.")
            migrated[f"critic.{output_layer_index}.{suffix}"] = value
        else:
            migrated[key] = value
    return migrated


def explained_variance(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return the fraction of target variance explained by predictions."""

    target_variance = float(np.var(targets))
    if target_variance == 0.0:
        return 0.0
    return float(1.0 - np.var(targets - predictions) / target_variance)


def _evaluate_policy(
    model: Any,
    *,
    depths: range | list[int] | tuple[int, ...],
    episodes_per_depth: int,
    max_episode_steps: int,
    data_dir: Path | str,
    seed: int | None,
    random_policy: bool,
    validate_dataset: bool,
    behavior_logger: BehaviorLogger | None,
    device: torch.device | str | None = None,
    state_data: Mapping[int, Any] | None = None,
) -> dict[str, Any]:
    if episodes_per_depth <= 0:
        raise ValueError("episodes_per_depth must be positive")

    by_depth: dict[str, dict[str, Any]] = {}
    aggregate: list[dict[str, Any]] = []
    for depth in depths:
        env = make_eval_env(
            depth=int(depth),
            max_episode_steps=max_episode_steps,
            data_dir=data_dir,
            validate_dataset=validate_dataset,
            state_data=(
                {int(depth): state_data[int(depth)]}
                if state_data is not None
                else None
            ),
        )
        if seed is not None:
            env.action_space.seed(seed + int(depth))
        available_states = len(env.states_by_depth[int(depth)])
        selection_mode = env.state_selection_mode(int(depth))
        exhaustive_deterministic_evaluation = (
            not random_policy and selection_mode == "exhaustive_cycle"
        )
        evaluation_episodes = (
            min(episodes_per_depth, available_states)
            if exhaustive_deterministic_evaluation
            else episodes_per_depth
        )
        depth_rows: list[dict[str, Any]] = []
        for episode in range(evaluation_episodes):
            episode_seed = (
                None
                if seed is None
                else seed + int(depth) * 10_000 + episode
            )
            reset_options = (
                {"state_index": episode}
                if exhaustive_deterministic_evaluation
                else None
            )
            observation, reset_info = env.reset(
                seed=episode_seed,
                options=reset_options,
            )
            done = False
            episode_reward = 0.0
            inverse_moves = 0
            final_info: dict[str, Any] = {}
            previous_action: int | None = None
            moves_taken: list[str] = []
            episode_id = f"{behavior_logger.config.run_id}-depth{depth}-episode{episode}" if behavior_logger else ""
            episode_started_at = utc_timestamp()

            while not done:
                start_state = encode_state(observation)
                step_index = int(final_info.get("current_step", 0))
                if random_policy:
                    action = int(env.action_space.sample())
                    policy_debug = _empty_policy_debug()
                else:
                    action, policy_debug = _policy_decision(model, observation, device)
                next_observation, reward, terminated, truncated, final_info = env.step(action)
                done = bool(terminated or truncated)
                episode_reward += float(reward)
                inverse_moves += int(final_info.get("immediate_inverse_move", False))
                move = ACTION_TO_MOVE[int(action)]
                moves_taken.append(move)

                if behavior_logger is not None:
                    behavior_logger.log_step(
                        int(depth),
                        {
                            "episode_id": episode_id,
                            "step_index": step_index,
                            "model_version": behavior_logger.config.model_version,
                            "scramble_depth": int(depth),
                            "start_state": start_state,
                            "action": int(action),
                            "move": move,
                            "end_state": encode_state(next_observation),
                            "reward": float(reward),
                            "total_reward_so_far": episode_reward,
                            "moves_used_so_far": int(final_info.get("current_step", step_index + 1)),
                            "previous_action": previous_action,
                            "immediate_inverse_move": bool(
                                final_info.get("immediate_inverse_move", False)
                            ),
                            "solved_after_move": bool(final_info.get("is_solved", False)),
                            "done": done,
                            "termination_reason": final_info.get(
                                "terminated_reason",
                                "running",
                            ),
                            **policy_debug,
                        },
                    )

                previous_action = int(action)
                observation = next_observation

            solved = bool(final_info.get("is_solved", False))
            steps = int(final_info.get("current_step", 0))
            timeout = final_info.get("terminated_reason") == "max_steps_reached"
            depth_rows.append(
                {
                    "reward": episode_reward,
                    "solved": solved,
                    "timeout": timeout,
                    "steps": steps,
                    "solution_length": steps if solved else None,
                    "inverse_moves": inverse_moves,
                    "extra_moves": steps - int(depth) if solved else None,
                    "moves_taken": moves_taken,
                }
            )
            if behavior_logger is not None:
                behavior_logger.log_episode(
                    int(depth),
                    {
                        "episode_id": episode_id,
                        "model_version": behavior_logger.config.model_version,
                        "scramble_depth": int(depth),
                        "scramble_sequence": reset_info.get("scramble_moves"),
                        "start_state": reset_info.get("encoded_state"),
                        "solved": solved,
                        "total_steps": steps,
                        "total_reward": episode_reward,
                        "timeout": timeout,
                        "termination_reason": final_info.get("terminated_reason"),
                        "moves_taken": " ".join(moves_taken),
                        "moves_taken_count": len(moves_taken),
                        "seed": episode_seed,
                        "timestamp": episode_started_at,
                    },
                )

        depth_metrics = _evaluation_metrics(depth_rows)
        depth_metrics.update(
            {
                "available_states": available_states,
                "state_selection": selection_mode,
            }
        )
        by_depth[str(depth)] = depth_metrics
        aggregate.extend(depth_rows)

    return {
        "episodes_per_depth": episodes_per_depth,
        "by_depth": by_depth,
        "overall": _evaluation_metrics(aggregate),
    }


def _evaluation_selection_mode(available_states: int) -> str:
    """Return the state-selection strategy for an evaluation dataset."""

    return (
        "exhaustive_cycle"
        if available_states < EXHAUSTIVE_STATE_THRESHOLD
        else "random"
    )


def _policy_action(
    model: Any,
    observation: np.ndarray,
    device: torch.device | str | None,
) -> int:
    if hasattr(model, "predict"):
        prediction = model.predict(observation, deterministic=True)
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        return int(np.asarray(prediction).item())
    return predict_action(model, observation, device)


def _policy_decision(
    model: Any,
    observation: np.ndarray,
    device: torch.device | str | None,
) -> tuple[int, dict[str, float | None]]:
    if hasattr(model, "predict"):
        prediction = model.predict(observation, deterministic=True)
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        return int(np.asarray(prediction).item()), _empty_policy_debug()

    if isinstance(model, ActorCriticNet):
        run_device = torch.device(device or "cpu")
        model.eval()
        with torch.no_grad():
            logits, values = model(
                preprocess_observations(
                    observation,
                    run_device,
                    observation_encoding=model.config.observation_encoding,
                )
            )
            distribution = Categorical(logits=logits)
            probabilities = torch.softmax(logits, dim=1)
            action = torch.argmax(logits, dim=1)
            log_probability = distribution.log_prob(action)
            entropy = distribution.entropy()

        action_id = int(action.item())
        return action_id, {
            "action_probability": float(probabilities[0, action_id].item()),
            "log_probability": float(log_probability.item()),
            "value_estimate": float(values.squeeze(0).item()),
            "entropy": float(entropy.item()),
        }

    return predict_action(model, observation, device), _empty_policy_debug()


def _empty_policy_debug() -> dict[str, None]:
    return {
        "action_probability": None,
        "log_probability": None,
        "value_estimate": None,
        "entropy": None,
    }


def _evaluation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    solved_rows = [row for row in rows if row["solved"]]
    solution_lengths = [
        int(row["solution_length"])
        for row in solved_rows
        if row["solution_length"] is not None
    ]
    extra_moves = [
        int(row["extra_moves"])
        for row in solved_rows
        if row["extra_moves"] is not None
    ]
    total_steps = sum(int(row["steps"]) for row in rows)
    action_counts = _action_counts(rows)
    return {
        "episodes": len(rows),
        "solved_count": len(solved_rows),
        "solve_rate": len(solved_rows) / len(rows) if rows else 0.0,
        "average_reward": mean(row["reward"] for row in rows) if rows else 0.0,
        "average_solution_length": mean(solution_lengths) if solution_lengths else None,
        "median_solution_length": median(solution_lengths) if solution_lengths else None,
        "timeout_rate": (
            sum(int(row["timeout"]) for row in rows) / len(rows) if rows else 0.0
        ),
        "inverse_move_rate": (
            sum(int(row["inverse_moves"]) for row in rows) / total_steps
            if total_steps
            else 0.0
        ),
        "average_extra_moves": mean(extra_moves) if extra_moves else None,
        "action_counts": action_counts,
        "action_distribution": _action_distribution(action_counts),
    }


def _action_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {ACTION_TO_MOVE[action_id]: 0 for action_id in sorted(ACTION_TO_MOVE)}
    for row in rows:
        for move in row.get("moves_taken", []):
            if move in counts:
                counts[move] += 1
    return counts


def _action_distribution(action_counts: Mapping[str, int]) -> dict[str, float]:
    total = sum(int(count) for count in action_counts.values())
    if total == 0:
        return {move: 0.0 for move in action_counts}
    return {
        move: int(count) / total
        for move, count in action_counts.items()
    }


def _episode_window_metrics(episodes: deque[dict[str, Any]]) -> dict[str, float]:
    if not episodes:
        return {
            "mean_episode_reward": 0.0,
            "mean_episode_length": 0.0,
            "train_solve_rate": 0.0,
            "timeout_rate": 0.0,
            "inverse_move_rate": 0.0,
        }
    total_steps = sum(int(row["length"]) for row in episodes)
    return {
        "mean_episode_reward": mean(float(row["reward"]) for row in episodes),
        "mean_episode_length": mean(int(row["length"]) for row in episodes),
        "train_solve_rate": mean(float(row["solved"]) for row in episodes),
        "timeout_rate": mean(float(row["timeout"]) for row in episodes),
        "inverse_move_rate": (
            sum(int(row["inverse_moves"]) for row in episodes) / total_steps
            if total_steps
            else 0.0
        ),
    }


def _network_config_dict(config: NetworkConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["hidden_layers"] = list(config.hidden_layers)
    return payload


def _network_config_from_dict(payload: Mapping[str, Any]) -> NetworkConfig:
    input_dim = int(payload.get("input_dim", ONE_HOT_OBSERVATION_SIZE))
    if "observation_encoding" in payload:
        observation_encoding = payload["observation_encoding"]
    else:
        observation_encoding = "normalized" if input_dim == OBSERVATION_SIZE else "one_hot"

    return NetworkConfig(
        input_dim=input_dim,
        hidden_layers=tuple(payload.get("hidden_layers", (256, 256))),
        actor_output_dim=int(payload.get("actor_output_dim", ACTION_SIZE)),
        critic_output_dim=int(payload.get("critic_output_dim", 1)),
        observation_encoding=observation_encoding,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_history_csv(
    path: Path | str,
    history: list[Mapping[str, Any]],
) -> None:
    """Write PPO update history as CSV, encoding nested values as JSON."""

    rows = list(history)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"history row {index} must be a mapping")

    fieldnames: list[str] = []
    seen_fields: set[str] = set()
    for row in rows:
        for fieldname in row:
            if not isinstance(fieldname, str):
                raise TypeError("history field names must be strings")
            if fieldname not in seen_fields:
                seen_fields.add(fieldname)
                fieldnames.append(fieldname)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")

    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            key: _history_csv_value(row.get(key))
                            for key in fieldnames
                        }
                    )
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _history_csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return value
