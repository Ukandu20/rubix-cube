"""Stable-Baselines3 PPO trainer kept parallel to the custom PPO trainer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping

import gymnasium as gym
import numpy as np

from agents.behavior_logging import utc_timestamp
from agents.ppo_agent import (
    DEFAULT_EVAL_EPISODES,
    DEFAULT_EVAL_FREQUENCY,
    DEFAULT_TOTAL_TIMESTEPS,
    COLOR_COUNT,
    OBSERVATION_SIZE,
    ONE_HOT_OBSERVATION_SIZE,
    evaluate_ppo_model,
    evaluate_random_baseline,
    make_training_env,
    state_files_for_depths,
)
from cube.gym_environment import DEFAULT_MAX_EPISODE_STEPS, DEFAULT_TRAINING_DATA_DIR
from curriculum.manager import CurriculumConfig, CurriculumManager


DEFAULT_SB3_OUTPUT_DIR = Path("models/artifacts/sb3_ppo")


def _require_sb3() -> tuple[Any, Any, Any, Any]:
    """Import the optional training dependency only when the SB3 path is used."""

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError(
            "Stable-Baselines3 training requires `pip install stable-baselines3`."
        ) from exc
    return PPO, BaseCallback, DummyVecEnv, SubprocVecEnv


@dataclass(frozen=True)
class SB3PPOConfig:
    """SB3 settings corresponding to the custom trainer's PPO defaults."""

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
    hidden_layers: tuple[int, ...] = (256, 256)


class OneHotObservation(gym.ObservationWrapper):
    """Expose categorical sticker colors without imposing numeric ordering."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(ONE_HOT_OBSERVATION_SIZE,),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation, dtype=np.int64)
        if values.shape != (OBSERVATION_SIZE,):
            raise ValueError(f"expected {OBSERVATION_SIZE} cube stickers")
        return np.eye(COLOR_COUNT, dtype=np.float32)[values].reshape(-1)


class SB3RawObservationPolicy:
    """Adapt an SB3 one-hot policy to the project's raw-state evaluator/showcase."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Any:
        values = np.asarray(observation)
        single = values.ndim == 1
        if single:
            values = values.reshape(1, -1)
        if values.shape[-1] != OBSERVATION_SIZE:
            raise ValueError(f"expected observations ending in {OBSERVATION_SIZE}")
        encoded = np.eye(COLOR_COUNT, dtype=np.float32)[values.astype(np.int64)]
        encoded = encoded.reshape(values.shape[0], ONE_HOT_OBSERVATION_SIZE)
        actions, state = self.model.predict(encoded, deterministic=deterministic)
        if single:
            actions = np.asarray(actions).reshape(-1)[0]
        return actions, state


def make_sb3_env_factory(
    *,
    curriculum_config: CurriculumConfig,
    data_dir: Path | str = DEFAULT_TRAINING_DATA_DIR,
    seed: int = 42,
    rank: int = 0,
    validate_dataset: bool = True,
) -> Callable[[], gym.Env]:
    """Create one independently seeded curriculum environment for a VecEnv."""

    def factory() -> gym.Env:
        manager = CurriculumManager(
            state_files_for_depths(
                curriculum_config.min_depth,
                curriculum_config.max_depth,
                data_dir,
            ),
            curriculum_config,
            validate_dataset=validate_dataset,
            seed=seed + rank,
            warn_on_normalization=False,
        )
        env = make_training_env(
            data_dir=data_dir,
            validate_dataset=validate_dataset,
            curriculum_manager=manager,
        )
        env.action_space.seed(seed + rank)
        return OneHotObservation(env)

    return factory


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _save_sb3_checkpoint(
    model: Any,
    path: Path,
    *,
    metadata: Mapping[str, Any],
) -> None:
    """Save an SB3 archive plus inspectable project metadata beside it."""

    model.save(path)
    _write_json(path.with_suffix(".metadata.json"), dict(metadata))


def build_curriculum_callback(
    *,
    manager: CurriculumManager,
    output_dir: Path,
    eval_frequency: int,
    eval_episodes: int,
    data_dir: Path | str,
    validate_dataset: bool,
    initial_timesteps: int = 0,
    previous_evaluations: list[dict[str, Any]] | None = None,
    previous_selection: Mapping[str, Any] | None = None,
) -> Any:
    """Build a callback preserving curriculum gates and checkpoint ranking."""

    _, BaseCallback, _, _ = _require_sb3()

    class CurriculumCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            selection = dict(previous_selection or {})
            self.evaluations = list(previous_evaluations or [])
            self.best_depth = int(selection.get("best_curriculum_depth", 0))
            self.best_solve_rate = float(selection.get("best_solve_rate", -1.0))
            self.best_timeout_rate = float(
                selection.get("best_timeout_rate", float("inf"))
            )
            self.best_by_depth = {
                int(depth): (
                    float(values["solve_rate"]),
                    float(values["timeout_rate"]),
                )
                for depth, values in selection.get("best_by_depth", {}).items()
            }
            self.next_evaluation = (
                (initial_timesteps // eval_frequency) + 1
            ) * eval_frequency

        def _on_step(self) -> bool:
            if self.num_timesteps < self.next_evaluation:
                return True
            depth = manager.get_current_depth()
            policy = SB3RawObservationPolicy(self.model)
            evaluation = evaluate_ppo_model(
                policy,
                depths=(depth,),
                episodes_per_depth=eval_episodes,
                max_episode_steps=manager.config.max_episode_steps(depth),
                data_dir=data_dir,
                validate_dataset=validate_dataset,
                state_data=manager.depth_data,
            )
            metrics = evaluation["by_depth"][str(depth)]
            solve_rate = float(metrics["solve_rate"])
            timeout_rate = float(metrics["timeout_rate"])
            advanced = manager.should_advance(metrics) and manager.increase_depth(
                timestep=self.num_timesteps, metrics=metrics
            )
            if advanced:
                self.training_env.env_method(
                    "set_curriculum_depth", manager.get_current_depth()
                )
            row = {
                "timesteps": self.num_timesteps,
                "curriculum_depth": depth,
                "metrics": metrics,
                "threshold": asdict(manager.config.advancement_thresholds[depth]),
                "advanced_curriculum": advanced,
                "next_curriculum_depth": manager.get_current_depth(),
            }
            self.evaluations.append(row)
            metadata = {
                "trainer": "stable_baselines3",
                "timesteps": self.num_timesteps,
                "evaluation": evaluation,
                "curriculum": manager.progress(),
            }
            previous = self.best_by_depth.get(depth)
            if previous is None or (solve_rate, -timeout_rate) > (previous[0], -previous[1]):
                self.best_by_depth[depth] = (solve_rate, timeout_rate)
                _save_sb3_checkpoint(
                    self.model, output_dir / f"best_model_depth_{depth}.zip",
                    metadata={**metadata, "selection": "best_at_curriculum_depth"},
                )
            if (depth, solve_rate, -timeout_rate) > (
                self.best_depth, self.best_solve_rate, -self.best_timeout_rate
            ):
                self.best_depth, self.best_solve_rate = depth, solve_rate
                self.best_timeout_rate = timeout_rate
                _save_sb3_checkpoint(
                    self.model, output_dir / "best_model.zip",
                    metadata={**metadata, "selection": "curriculum_lexicographic"},
                )
            if advanced:
                _save_sb3_checkpoint(
                    self.model, output_dir / "latest_passed_gate.zip",
                    metadata={
                        **metadata,
                        "selection": "latest_passed_curriculum_gate",
                        "passed_curriculum_depth": depth,
                    },
                )
            _write_json(
                output_dir / "curriculum_progress.json",
                {**manager.progress(), "evaluations": self.evaluations},
            )
            while self.next_evaluation <= self.num_timesteps:
                self.next_evaluation += eval_frequency
            return True

    return CurriculumCallback()


def _restore_curriculum_progress(
    output_path: Path,
    manager: CurriculumManager,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Restore curriculum depth/events/evaluation history for an in-place resume."""

    progress_path = output_path / "curriculum_progress.json"
    if not progress_path.exists():
        return [], {}
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    manager.set_depth(int(payload["current_depth"]))
    manager.events = [dict(event) for event in payload.get("events", [])]
    metrics_path = output_path / "metrics.json"
    selection: dict[str, Any] = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        selection = dict(metrics.get("checkpoint_selection", {}))
    return [dict(row) for row in payload.get("evaluations", [])], selection


def train_sb3_ppo(
    *,
    curriculum_config: CurriculumConfig,
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    data_dir: Path | str = DEFAULT_TRAINING_DATA_DIR,
    output_dir: Path | str = DEFAULT_SB3_OUTPUT_DIR,
    seed: int = 42,
    device: str = "auto",
    eval_frequency: int = DEFAULT_EVAL_FREQUENCY,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    n_envs: int = 1,
    subprocess: bool = False,
    config: SB3PPOConfig | None = None,
    validate_dataset: bool = True,
    output_metadata: Mapping[str, Any] | None = None,
    resume_from: Path | str | None = None,
) -> dict[str, Any]:
    """Train SB3 PPO without mutating or depending on the custom trainer path."""

    if min(total_timesteps, eval_frequency, eval_episodes, n_envs) <= 0:
        raise ValueError("timesteps, evaluation values, and n_envs must be positive")
    PPO, _, DummyVecEnv, SubprocVecEnv = _require_sb3()
    cfg = config or SB3PPOConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manager = CurriculumManager(
        state_files_for_depths(curriculum_config.min_depth, curriculum_config.max_depth, data_dir),
        curriculum_config,
        validate_dataset=validate_dataset,
        seed=seed,
    )
    previous_evaluations, previous_selection = (
        _restore_curriculum_progress(output_path, manager)
        if resume_from is not None
        else ([], {})
    )
    factories = [
        make_sb3_env_factory(
            curriculum_config=curriculum_config, data_dir=data_dir, seed=seed,
            rank=rank, validate_dataset=validate_dataset,
        )
        for rank in range(n_envs)
    ]
    vec_env = (SubprocVecEnv if subprocess and n_envs > 1 else DummyVecEnv)(factories)
    if resume_from is None:
        model = PPO(
            "MlpPolicy", vec_env, learning_rate=cfg.learning_rate, gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda, clip_range=cfg.clip_range,
            n_epochs=cfg.n_epochs, n_steps=cfg.n_steps, batch_size=cfg.batch_size,
            ent_coef=cfg.ent_coef, vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm, target_kl=cfg.target_kl,
            seed=seed, device=device,
            policy_kwargs={
                "net_arch": {
                    "pi": list(cfg.hidden_layers),
                    "vf": list(cfg.hidden_layers),
                }
            },
        )
    else:
        model = PPO.load(Path(resume_from), env=vec_env, device=device)
    initial_timesteps = int(model.num_timesteps)
    vec_env.env_method("set_curriculum_depth", manager.get_current_depth())
    callback = build_curriculum_callback(
        manager=manager, output_dir=output_path, eval_frequency=eval_frequency,
        eval_episodes=eval_episodes, data_dir=data_dir,
        validate_dataset=validate_dataset, initial_timesteps=initial_timesteps,
        previous_evaluations=previous_evaluations,
        previous_selection=previous_selection,
    )
    started_at = utc_timestamp()
    start = perf_counter()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=resume_from is None,
        )
    finally:
        vec_env.close()
    elapsed = perf_counter() - start
    final_depth = manager.get_current_depth()
    final_evaluation = evaluate_ppo_model(
        SB3RawObservationPolicy(model), depths=(final_depth,),
        episodes_per_depth=eval_episodes,
        max_episode_steps=curriculum_config.max_episode_steps(final_depth),
        data_dir=data_dir, validate_dataset=validate_dataset,
        state_data=manager.depth_data,
    )
    metadata = {
        "trainer": "stable_baselines3", "timesteps": model.num_timesteps,
        "evaluation": final_evaluation, "curriculum": manager.progress(),
    }
    _save_sb3_checkpoint(model, output_path / "final_model.zip", metadata=metadata)
    random_baseline = evaluate_random_baseline(
        depths=range(curriculum_config.min_depth, curriculum_config.max_depth + 1),
        episodes_per_depth=max(1, min(eval_episodes, 20)), data_dir=data_dir,
        seed=seed, validate_dataset=validate_dataset,
        state_data=manager.depth_data,
    )
    run_config = {
        "trainer": "stable_baselines3", "total_timesteps": total_timesteps,
        "actual_timesteps": model.num_timesteps, "seed": seed, "device": device,
        "initial_timesteps": initial_timesteps,
        "resume_from": str(resume_from) if resume_from is not None else None,
        "n_envs": n_envs, "subprocess": subprocess, "ppo": asdict(cfg),
        "curriculum": curriculum_config.to_dict(),
        "training_session": {"started_at": started_at, "completed_at": utc_timestamp(), "elapsed_seconds": elapsed},
        **dict(output_metadata or {}),
    }
    metrics = {
        "final_evaluation": final_evaluation,
        "random_baseline": random_baseline,
        "checkpoint_selection": {
            "strategy": "curriculum_lexicographic",
            "best_curriculum_depth": callback.best_depth,
            "best_solve_rate": callback.best_solve_rate,
            "best_timeout_rate": callback.best_timeout_rate,
            "best_by_depth": {
                str(depth): {
                    "solve_rate": values[0],
                    "timeout_rate": values[1],
                }
                for depth, values in sorted(callback.best_by_depth.items())
            },
        },
        "curriculum_progress": {
            **manager.progress(),
            "evaluations": callback.evaluations,
        },
    }
    _write_json(output_path / "config.json", run_config)
    _write_json(output_path / "metrics.json", metrics)
    return {"model": model, "config": run_config, "metrics": metrics, "output_dir": output_path}
