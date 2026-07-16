"""Random baseline agent."""

from __future__ import annotations

import random

from cube.environment import ACTION_SIZE, CubeEnvironment


class RandomAgent:
    """Agent that chooses a legal action uniformly at random."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng if rng is not None else random.Random()

    def act(self, env: CubeEnvironment) -> int:
        """Return a random action index from the environment action space."""

        return self.rng.randrange(ACTION_SIZE)
