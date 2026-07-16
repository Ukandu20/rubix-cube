"""Checkpoint schema and persistence for SB3 supervised warm starts."""

from __future__ import annotations

import platform
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from cube.environment import ACTION_TO_MOVE

OBSERVATION_ENCODING = "one_hot"
OBSERVATION_SHAPE = (54 * 6,)
ACTION_ORDER = tuple(ACTION_TO_MOVE[index] for index in sorted(ACTION_TO_MOVE))
ACTION_COUNT = len(ACTION_ORDER)
STATE_ENCODER_VERSION = "sb3_cube_one_hot_v1"
MODEL_ARCHITECTURE_VERSION = "sb3_mlp_actor_critic_v1"


def checkpoint_payload(
    *,
    model: Any,
    config: Any,
    prepared: Any,
    seed: int,
    updated_names: Sequence[str],
    best_epoch: int,
    history: Sequence[Mapping[str, Any]],
    validation_metrics: Mapping[str, Any],
    runtime: float,
) -> dict[str, Any]:
    """Build a self-describing warm-start checkpoint payload."""

    return {
        "model_state_dict": model.policy.state_dict(),
        "architecture_version": MODEL_ARCHITECTURE_VERSION,
        "observation_schema": {
            "observation_encoding": OBSERVATION_ENCODING,
            "observation_shape": list(OBSERVATION_SHAPE),
            "state_encoder_version": STATE_ENCODER_VERSION,
        },
        "action_schema": {
            "action_count": ACTION_COUNT,
            "action_order": list(ACTION_ORDER),
        },
        "selected_depth_range": [prepared.depths[0], prepared.depths[-1]],
        "depth_mode": config.depth_mode,
        "dataset_counts": prepared.manifest["dataset_counts"],
        "train_counts_by_depth": prepared.manifest["split_counts"]["train"],
        "validation_counts_by_depth": prepared.manifest["split_counts"]["validation"],
        "test_counts_by_depth": prepared.manifest["split_counts"]["test"],
        "split_manifest_reference": str(prepared.manifest_path),
        "sampling_strategy": config.depth_sampling,
        "seed": seed,
        "hyperparameters": asdict(config),
        "updated_parameter_groups": list(updated_names),
        "shared_encoder_updated": False,
        "critic_head_updated": False,
        "best_epoch": best_epoch,
        "training_history": list(history),
        "validation_metrics": dict(validation_metrics),
        "dataset_identifiers": prepared.manifest["selected_rows"],
        "dataset_hashes": prepared.manifest["source_files"],
        "runtime": runtime,
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }


def load_supervised_warm_start_checkpoint(
    model: Any, path: Path | str
) -> dict[str, Any]:
    """Validate schema compatibility and restore a supervised checkpoint."""

    payload = torch.load(
        Path(path),
        map_location=next(model.policy.parameters()).device,
        weights_only=False,
    )
    if payload.get("architecture_version") != MODEL_ARCHITECTURE_VERSION:
        raise ValueError("supervised checkpoint architecture version mismatch")
    observation = payload.get("observation_schema", {})
    if (
        observation.get("observation_encoding") != OBSERVATION_ENCODING
        or tuple(observation.get("observation_shape", ())) != OBSERVATION_SHAPE
    ):
        raise ValueError("supervised checkpoint observation schema mismatch")
    action = payload.get("action_schema", {})
    if (
        action.get("action_count") != ACTION_COUNT
        or tuple(action.get("action_order", ())) != ACTION_ORDER
    ):
        raise ValueError("supervised checkpoint action order mismatch")
    model.policy.load_state_dict(payload["model_state_dict"])
    return payload


def reset_ppo_optimizer(model: Any) -> None:
    """Discard old momentum and create a fresh full-policy optimizer."""

    model.policy.optimizer = model.policy.optimizer_class(
        model.policy.parameters(),
        lr=model.lr_schedule(1),
        **model.policy.optimizer_kwargs,
    )
