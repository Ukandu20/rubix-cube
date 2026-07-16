"""Validated configuration for SB3 supervised actor warm starts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from curriculum.manager import CurriculumConfig

DepthMode = Literal["mastered", "frontier", "full-curriculum", "custom"]
DepthSampling = Literal["balanced", "natural", "frontier-weighted"]


@dataclass(frozen=True)
class SupervisedWarmStartConfig:
    """Settings for dataset selection and supervised optimization."""

    depth_mode: DepthMode = "frontier"
    min_depth: int | None = None
    max_depth: int | None = None
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-3
    sample_per_depth: int = 100_000
    validation_fraction: float = 0.10
    test_fraction: float = 0.10
    depth_sampling: DepthSampling = "balanced"
    label_smoothing: float = 0.01
    early_stopping_patience: int = 2
    gradient_clip_norm: float = 1.0
    update_shared_encoder: bool = False
    rollout_sample_per_depth: int = 1_000

    def __post_init__(self) -> None:
        if self.depth_mode not in ("mastered", "frontier", "full-curriculum", "custom"):
            raise ValueError("unsupported supervised depth mode")
        if self.depth_sampling not in ("balanced", "natural", "frontier-weighted"):
            raise ValueError("unsupported supervised depth sampling strategy")
        if self.depth_mode in ("mastered", "frontier") and self.max_depth is None:
            raise ValueError(
                f"pretrain_max_depth is required for {self.depth_mode!r} mode"
            )
        if self.depth_mode == "custom" and (
            self.min_depth is None or self.max_depth is None
        ):
            raise ValueError(
                "custom mode requires pretrain_min_depth and pretrain_max_depth"
            )
        if self.min_depth is not None and self.min_depth <= 0:
            raise ValueError("pretrain_min_depth must be positive")
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError("pretrain_max_depth must be positive")
        if (
            self.min_depth is not None
            and self.max_depth is not None
            and self.min_depth > self.max_depth
        ):
            raise ValueError("pretrain_min_depth cannot exceed pretrain_max_depth")
        if (
            min(
                self.epochs,
                self.batch_size,
                self.sample_per_depth,
                self.early_stopping_patience,
                self.rollout_sample_per_depth,
            )
            <= 0
        ):
            raise ValueError("supervised count and patience settings must be positive")
        if self.learning_rate <= 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("supervised learning rate and clip norm must be positive")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("pretrain_label_smoothing must be in [0, 1)")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("pretrain_validation_fraction must be in (0, 1)")
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("pretrain_test_fraction must be in (0, 1)")
        if self.validation_fraction + self.test_fraction >= 1.0:
            raise ValueError("validation and test fractions must sum to less than 1")
        if self.update_shared_encoder:
            raise ValueError(
                "the current SB3 policy has no trainable shared encoder; "
                "pretrain_update_shared_encoder is unsupported"
            )

    def selected_depths(self, curriculum: CurriculumConfig) -> tuple[int, ...]:
        """Resolve the exact supervised depths for one curriculum."""

        minimum = self.min_depth or curriculum.min_depth
        if self.depth_mode == "full-curriculum":
            minimum, maximum = curriculum.min_depth, curriculum.max_depth
        elif self.depth_mode == "mastered":
            maximum = int(self.max_depth) - 1
        else:
            maximum = int(self.max_depth)
        if minimum < curriculum.min_depth or maximum > curriculum.max_depth:
            raise ValueError("supervised depth range is outside the curriculum")
        if maximum < minimum:
            raise ValueError("selected supervised depth range is empty")
        return tuple(range(minimum, maximum + 1))

    def effective_frontier(self, curriculum: CurriculumConfig) -> int:
        """Return the deepest level emphasized by supervised sampling."""

        if self.depth_mode == "full-curriculum":
            return curriculum.max_depth
        return int(self.max_depth)
