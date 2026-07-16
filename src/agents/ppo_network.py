"""Network architecture and configuration for the custom PPO agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

OBSERVATION_SIZE = 54
COLOR_COUNT = 6
ONE_HOT_OBSERVATION_SIZE = OBSERVATION_SIZE * COLOR_COUNT
ACTION_SIZE = 12
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
    """Actor-critic MLP shape and observation schema."""

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
        self.actor = build_mlp(
            input_dim=self.config.input_dim,
            hidden_layers=self.config.hidden_layers,
            output_dim=self.config.actor_output_dim,
        )
        self.critic = build_mlp(
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


def build_mlp(
    *, input_dim: int, hidden_layers: tuple[int, ...], output_dim: int
) -> nn.Sequential:
    """Build one actor or critic MLP."""

    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_layers:
        layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ReLU()))
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)
