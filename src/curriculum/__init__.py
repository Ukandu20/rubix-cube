"""Curriculum learning support for dataset-driven PPO training."""

from curriculum.manager import (
    AdvancementThreshold,
    CurriculumConfig,
    CurriculumManager,
    DEFAULT_CURRICULUM_CONFIG_PATH,
    load_curriculum_config,
)

__all__ = (
    "AdvancementThreshold",
    "CurriculumConfig",
    "CurriculumManager",
    "DEFAULT_CURRICULUM_CONFIG_PATH",
    "load_curriculum_config",
)
