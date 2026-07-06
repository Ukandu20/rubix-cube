# Curriculum Learning Specification for Rubik's Cube PPO Agent

## 1. Purpose

This document defines the specification for implementing **curriculum learning** in the Rubik's Cube reinforcement learning project.

The goal of curriculum learning in this project is to train the PPO agent from **easy scramble states to harder scramble states** instead of exposing it to all possible cube states immediately.

The curriculum will be based on **scramble depth**:

```text
Depth 1 → Depth 2 → Depth 3 → Depth 4 → Depth 5 → deeper depths later
```

A scramble depth represents how many moves away a cube state is from the solved state, assuming the state was generated from a valid scramble sequence.

---

## 2. High-Level Concept

Curriculum learning should not change the PPO algorithm itself.

Instead, curriculum learning controls the **starting states** given to the agent during environment reset.

In simple terms:

```text
PPO decides how to learn.
Curriculum learning decides what difficulty level PPO trains on.
```

The PPO agent still performs its normal responsibilities:

- Collect rollouts
- Estimate rewards-to-go
- Compute advantages
- Update the actor policy
- Update the critic value function
- Apply PPO clipping
- Track entropy and KL divergence

The curriculum system controls:

- Which scramble depth the agent trains on
- How states are sampled
- When difficulty increases
- Whether earlier depths remain mixed into training
- How curriculum progress is logged

---

## 3. Curriculum Difficulty Definition

The primary difficulty measure is:

```text
scramble_depth
```

Example:

| Scramble Depth | Meaning |
|---|---|
| 1 | Cube is 1 move away from solved |
| 2 | Cube is 2 moves away from solved |
| 3 | Cube is 3 moves away from solved |
| 4 | Cube is 4 moves away from solved |
| 5 | Cube is 5 moves away from solved |

For version 1 of the project, the curriculum should support depths `1` through `5`.

Later versions can extend this to deeper scrambles.

---

## 4. Required Data Files

The project already has pre-generated unique states for depths `1–5`.

The curriculum system should use those files as the source of starting states.

Recommended data structure:

```text
data/
    depth_1.parquet
    depth_2.parquet
    depth_3.parquet
    depth_4.parquet
    depth_5.parquet
```

CSV files are acceptable for early development, but Parquet is recommended for larger datasets because it is faster and more efficient.

---

## 5. Required Dataset Columns

Each depth file should contain at least:

| Column | Required | Description |
|---|---:|---|
| `encoded_state` | Yes | The cube state as a 54-character encoded string |
| `depth` | Yes | The scramble depth of the state |

Example encoded state:

```text
BYYBYYBYYOOOOOOOOOYGGYGGYGGGWWGWWGWWRRRRRRRRRBBWBBWBBW
```

Recommended additional columns:

| Column | Required | Description |
|---|---:|---|
| `scramble_moves` | Recommended | The moves used to generate the scrambled state |
| `inverse_solution` | Recommended | The reverse solution sequence |
| `state_id` | Recommended | Unique ID for the state |
| `optimal_distance` | Optional | Known optimal distance from solved state, if available |
| `source_file` | Optional | File where the state came from |

Example row:

| encoded_state | depth | scramble_moves | inverse_solution |
|---|---:|---|---|
| `BYY...BBW` | 3 | `R U F` | `F' U' R'` |

---

## 6. Environment Requirements

The custom Gym/Gymnasium environment must support curriculum-controlled resets.

The most important requirement is:

```text
reset() must sample the starting cube state from the current curriculum depth.
```

The environment should not randomly sample from all depths unless the curriculum manager explicitly allows it.

### 6.1 Environment Responsibilities

The environment is responsible for:

- Holding the current cube state
- Applying actions/moves
- Returning observations
- Computing rewards
- Detecting solved states
- Detecting timeouts
- Detecting immediate inverse moves
- Returning episode information in `info`
- Requesting start states from the curriculum manager during `reset()`

### 6.2 Required Environment Methods

The environment should expose or internally support:

```python
reset()
step(action)
set_curriculum_depth(depth)
get_current_depth()
is_solved()
is_inverse_move(action)
```

Depending on implementation style, `set_curriculum_depth()` may belong directly to the environment or to the curriculum manager.

---

## 7. Curriculum Manager Requirements

A separate `CurriculumManager` should be created to keep the curriculum logic outside the PPO algorithm and outside the low-level cube mechanics.

### 7.1 Curriculum Manager Responsibilities

The curriculum manager is responsible for:

- Loading depth-specific datasets
- Tracking the current curriculum depth
- Sampling start states
- Applying the sampling strategy
- Increasing difficulty when performance thresholds are met
- Preventing curriculum depth from exceeding the available dataset depth
- Logging curriculum changes

### 7.2 Suggested Class Structure

```python
class CurriculumManager:
    def __init__(self, depth_files, max_depth=5):
        """
        Manages the curriculum learning schedule.

        Parameters
        ----------
        depth_files:
            Dictionary mapping depth numbers to dataset file paths.
            Example:
            {
                1: "data/depth_1.parquet",
                2: "data/depth_2.parquet",
                3: "data/depth_3.parquet",
                4: "data/depth_4.parquet",
                5: "data/depth_5.parquet",
            }

        max_depth:
            Maximum curriculum depth available for training.
        """
        self.depth_files = depth_files
        self.max_depth = max_depth
        self.current_depth = 1
        self.depth_data = self.load_depth_data()

    def load_depth_data(self):
        """
        Loads all depth datasets into memory or prepares them for lazy loading.
        """
        pass

    def sample_state(self):
        """
        Samples a starting cube state based on the current curriculum depth
        and the chosen sampling strategy.
        """
        pass

    def increase_depth(self):
        """
        Increases the current curriculum depth by 1 if the maximum depth
        has not already been reached.
        """
        if self.current_depth < self.max_depth:
            self.current_depth += 1

    def set_depth(self, depth):
        """
        Manually sets the curriculum depth.
        Useful for debugging or evaluation.
        """
        if depth < 1 or depth > self.max_depth:
            raise ValueError("Invalid curriculum depth")
        self.current_depth = depth
```

---

## 8. Curriculum Sampling Strategy

The project should support at least two sampling strategies.

---

### 8.1 Strict Curriculum Sampling

In strict curriculum sampling, the agent only trains on the current depth.

Example:

```text
Current depth = 3
Sample only from depth_3.parquet
```

Advantages:

- Simple to implement
- Easy to debug
- Clear difficulty progression

Disadvantages:

- The agent may forget earlier depths
- The agent may overfit to the current depth
- Earlier skills may become unstable

---

### 8.2 Mixed Curriculum Sampling

In mixed curriculum sampling, the agent trains mostly on the current depth but still sees earlier depths.

Example:

```text
Current depth = 3
Sample from:
- Depth 1
- Depth 2
- Depth 3
```

Recommended for this project.

Suggested sampling schedule:

| Current Curriculum Depth | Depth 1 | Depth 2 | Depth 3 | Depth 4 | Depth 5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 100% | 0% | 0% | 0% | 0% |
| 2 | 30% | 70% | 0% | 0% | 0% |
| 3 | 20% | 30% | 50% | 0% | 0% |
| 4 | 10% | 20% | 30% | 40% | 0% |
| 5 | 10% | 15% | 20% | 25% | 30% |

This strategy helps prevent the agent from forgetting easier states while learning harder ones.

---

## 9. Curriculum Progression Rules

The agent should only move to a harder depth when evaluation metrics show that it is ready.

Difficulty should not increase only because a fixed number of timesteps has passed.

### 9.1 Required Metrics

The curriculum progression system should track:

| Metric | Description |
|---|---|
| `success_rate` | Percentage of evaluation episodes solved |
| `average_episode_length` | Average number of moves used in solved episodes |
| `timeout_rate` | Percentage of episodes that ended by timeout |
| `average_reward` | Mean episode reward |
| `inverse_move_rate` | Frequency of immediate inverse moves |
| `current_depth` | Current curriculum level |

Optional PPO metrics to monitor:

|Metric|Description|
|---|---|
|`entropy`|Measures how much the policy is still exploring|
|`approx_kl`|Measures how much the policy changed during PPO update|
|`value_loss`|Critic/value-function loss|
|`policy_loss`|Actor/policy loss|
|`clip_fraction`|Percentage of PPO updates affected by clipping|

---

### 9.2 Suggested Advancement Rule

A simple rule for advancing from depth `d` to depth `d + 1`:

```text
Advance if:

success_rate >= required_success_rate
AND average_episode_length <= max_allowed_average_moves
AND timeout_rate <= max_allowed_timeout_rate
```

Suggested thresholds:

|Current Depth|Required Success Rate|Max Average Moves|Max Timeout Rate|
|---:|---:|---:|---:|
|1|90%|2|10%|
|2|85%|4|10%|
|3|80%|6|15%|
|4|75%–80%|8|15%|
|5|Stable training target|10|20%|

For a simple formula:

```python
max_allowed_average_moves = 3 * current_depth
```

---

## 10. Evaluation Requirements

Training results alone should not be used to decide curriculum progression.

A separate evaluation loop should test the current policy with reduced or no exploration.

If using Stable-Baselines3, evaluation should use:

```python
model.predict(obs, deterministic=True)
```

### 10.1 Evaluation Function

```python
def evaluate_agent(model, env, num_episodes=100):
    """
    Evaluates the PPO agent on the current curriculum depth.

    Parameters
    ----------
    model:
        Trained or partially trained PPO model.

    env:
        Evaluation environment.

    num_episodes:
        Number of episodes to evaluate.

    Returns
    -------
    dict:
        Evaluation metrics including success rate, average moves,
        timeout rate, and average reward.
    """

    solved_count = 0
    timeout_count = 0
    total_moves_for_solved = 0
    total_reward = 0
    inverse_move_count = 0
    total_steps = 0

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        moves_used = 0

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            moves_used += 1
            total_steps += 1

            if info.get("is_inverse_move", False):
                inverse_move_count += 1

        total_reward += episode_reward

        if info.get("is_solved", False):
            solved_count += 1
            total_moves_for_solved += moves_used
        else:
            timeout_count += 1

    success_rate = solved_count / num_episodes
    timeout_rate = timeout_count / num_episodes
    average_reward = total_reward / num_episodes
    inverse_move_rate = inverse_move_count / total_steps if total_steps > 0 else 0

    average_moves = (
        total_moves_for_solved / solved_count
        if solved_count > 0
        else None
    )

    return {
        "success_rate": success_rate,
        "average_moves": average_moves,
        "timeout_rate": timeout_rate,
        "average_reward": average_reward,
        "inverse_move_rate": inverse_move_rate,
    }
```

---

## 11. Reward Function Requirements

Curriculum learning does not require a new reward formula.

The existing reward system can be used.

Recommended reward components:

| Event | Reward |
|---|---:|
| Ordinary move | `-1` |
| Solved cube | `+99` |
| Immediate inverse move | `-5` or `-10` |
| Timeout/failure | `-10` |

General reward form:

```text
R_t = move_penalty + solve_bonus + inverse_penalty + timeout_penalty
```

More explicitly:

```text
R_t = -1 + B_solve + P_inverse + P_timeout
```

Where:

```text
B_solve = +99 if the cube is solved, otherwise 0
P_inverse = -5 or -10 if the action immediately reverses the previous move, otherwise 0
P_timeout = -10 if the episode times out, otherwise 0
```

---

## 12. Timeout / Max-Step Requirements

The environment should use a depth-based timeout.

The timeout should be long enough to allow reasonable exploration but short enough to prevent endless wandering.

Recommended formula:

```python
max_steps = (3 * current_depth) + 1
```

Example:

| Current Depth | Max Steps |
|---:|---:|
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |
| 4 | 9 |
| 5 | 11 |

This gives the agent slightly more room than the minimum solution length.

---

## 13. PPO Agent Requirements

The PPO implementation does not need to be rewritten for curriculum learning.

The PPO agent should receive observations and rewards normally from the environment.

PPO responsibilities remain:

- Select an action from the policy network
- Store rollouts
- Compute rewards-to-go
- Compute advantages
- Use GAE if enabled
- Update actor and critic networks
- Apply clipped objective
- Track policy entropy
- Track approximate KL divergence

Curriculum learning should be implemented outside the PPO loss function.

Correct relationship:

```text
Curriculum Manager → controls difficulty of start states
Environment → uses curriculum manager during reset
PPO Agent → learns from the experience generated by the environment
Evaluation Loop → decides whether the curriculum should advance
```

---

## 14. Stable-Baselines3 Callback Requirement

If using Stable-Baselines3, curriculum progression can be implemented with a custom callback.

### 14.1 Suggested Callback Structure

```python
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    def __init__(self, eval_env, curriculum_manager, eval_freq=10_000):
        """
        Custom callback for curriculum learning.

        Parameters
        ----------
        eval_env:
            Environment used for evaluation.

        curriculum_manager:
            CurriculumManager instance that tracks and updates difficulty.

        eval_freq:
            Number of training timesteps between evaluations.
        """
        super().__init__()
        self.eval_env = eval_env
        self.curriculum_manager = curriculum_manager
        self.eval_freq = eval_freq

    def _on_step(self):
        """
        Called repeatedly during training.
        Evaluates the model every eval_freq timesteps.
        """

        if self.num_timesteps % self.eval_freq == 0:
            metrics = evaluate_agent(
                model=self.model,
                env=self.eval_env,
                num_episodes=100,
            )

            current_depth = self.curriculum_manager.current_depth

            if should_advance_curriculum(current_depth, metrics):
                self.curriculum_manager.increase_depth()

        return True
```

---

## 15. Curriculum Advancement Helper

```python
def should_advance_curriculum(current_depth, metrics):
    """
    Decides whether the agent should move to the next curriculum depth.
    """

    success_rate = metrics["success_rate"]
    average_moves = metrics["average_moves"]
    timeout_rate = metrics["timeout_rate"]

    if average_moves is None:
        return False

    thresholds = {
        1: {"success_rate": 0.95, "max_moves": 3, "timeout_rate": 0.05},
        2: {"success_rate": 0.90, "max_moves": 6, "timeout_rate": 0.10},
        3: {"success_rate": 0.85, "max_moves": 9, "timeout_rate": 0.15},
        4: {"success_rate": 0.80, "max_moves": 12, "timeout_rate": 0.20},
        5: {"success_rate": 0.75, "max_moves": 15, "timeout_rate": 0.25},
    }

    target = thresholds[current_depth]

    return (
        success_rate >= target["success_rate"]
        and average_moves <= target["max_moves"]
        and timeout_rate <= target["timeout_rate"]
    )
```

---

## 16. Logging Requirements

The project should log both episode-level and step-level data.

This is important for debugging where the model succeeds, fails, loops, or overuses inverse moves.

---

### 16.1 Episode-Level Logs

Recommended file:

```text
logs/train_episodes.parquet
```

Recommended columns:

| Column | Description |
|---|---|
| `episode_id` | Unique episode number |
| `global_timestep` | Total training timestep at episode end |
| `curriculum_depth` | Current training depth |
| `start_state` | Initial encoded cube state |
| `final_state` | Final encoded cube state |
| `solved` | Whether the cube was solved |
| `moves_used` | Number of moves used |
| `total_reward` | Sum of rewards in episode |
| `timeout` | Whether episode ended by timeout |
| `inverse_move_count` | Number of immediate inverse moves |
| `action_sequence` | Full sequence of moves taken |

---

### 16.2 Step-Level Logs

Recommended file:

```text
logs/train_steps.parquet
```

Recommended columns:

| Column | Description |
|---|---|
| `episode_id` | Episode identifier |
| `step` | Step number inside the episode |
| `curriculum_depth` | Current curriculum depth |
| `current_state` | State before action |
| `action` | Move selected by the agent |
| `next_state` | State after action |
| `reward` | Reward received |
| `done` | Whether episode ended |
| `truncated` | Whether episode timed out |
| `is_inverse_move` | Whether action reversed the previous action |
| `is_solved` | Whether next state is solved |

---

### 16.3 Evaluation Logs

Recommended file:

```text
logs/eval_results.parquet
```

Recommended columns:

| Column | Description |
|---|---|
| `eval_id` | Evaluation run identifier |
| `global_timestep` | Training timestep when evaluation happened |
| `curriculum_depth` | Depth evaluated |
| `num_episodes` | Number of evaluation episodes |
| `success_rate` | Percentage solved |
| `average_moves` | Average moves for solved episodes |
| `timeout_rate` | Percentage timed out |
| `average_reward` | Mean reward |
| `inverse_move_rate` | Frequency of inverse moves |
| `advanced_curriculum` | Whether depth increased after evaluation |

---

## 17. Recommended Project Structure

```text
rubiks_rl_project/
│
├── data/
│   ├── depth_1.parquet
│   ├── depth_2.parquet
│   ├── depth_3.parquet
│   ├── depth_4.parquet
│   └── depth_5.parquet
│
├── env/
│   ├── rubiks_env.py
│   ├── cube.py
│   ├── moves.py
│   ├── notation.py
│   └── rewards.py
│
├── curriculum/
│   ├── curriculum_manager.py
│   └── sampling_strategy.py
│
├── agents/
│   ├── ppo_agent.py
│   └── callbacks.py
│
├── evaluation/
│   ├── evaluate_agent.py
│   └── metrics.py
│
├── logs/
│   ├── train_steps.parquet
│   ├── train_episodes.parquet
│   ├── eval_results.parquet
│   └── curriculum_progress.json
│
├── configs/
│   └── curriculum_config.yaml
│
└── train.py
```

---

## 18. Configuration Requirements

Curriculum settings should be stored in a config file instead of hardcoded.

Recommended file:

```text
configs/curriculum_config.yaml
```

Example:

```yaml
curriculum:
  min_depth: 1
  max_depth: 5
  starting_depth: 1
  sampling_strategy: "mixed"

  eval_freq: 10000
  eval_episodes: 100

  max_steps_formula: "2 * depth + 1"

  advancement_thresholds:
    1:
      success_rate: 0.90
      max_average_moves: 2
      max_timeout_rate: 0.10
    2:
      success_rate: 0.85
      max_average_moves: 4
      max_timeout_rate: 0.10
    3:
      success_rate: 0.80
      max_average_moves: 6
      max_timeout_rate: 0.15
    4:
      success_rate: 0.75
      max_average_moves: 8
      max_timeout_rate: 0.15
    5:
      success_rate: 0.75
      max_average_moves: 10
      max_timeout_rate: 0.20

  mixed_sampling_weights:
    1:
      1: 1.00
    2:
      1: 0.30
      2: 0.70
    3:
      1: 0.20
      2: 0.30
      3: 0.50
    4:
      1: 0.10
      2: 0.20
      3: 0.30
      4: 0.40
    5:
      1: 0.10
      2: 0.15
      3: 0.20
      4: 0.25
      5: 0.30
```

---

## 19. Minimum Viable Implementation

The minimum version of curriculum learning requires:

```text
1. Pre-generated depth files for depths 1–5
2. A CurriculumManager class
3. Environment reset controlled by curriculum depth
4. Depth-based max-step limit
5. PPO training loop
6. Evaluation loop
7. Advancement rule based on success rate
8. Episode-level logging
```

This is enough to correctly implement curriculum learning for the first version of the project.

---

## 20. Recommended Version 1 Training Flow

```text
1. Load depth_1 to depth_5 datasets
2. Initialize CurriculumManager at depth 1
3. Initialize RubiksCubeEnv with CurriculumManager
4. Initialize PPO agent
5. Train PPO on current curriculum depth
6. Every N timesteps, evaluate the agent
7. If success criteria are met, increase curriculum depth
8. Continue until depth 5 is reached
9. Keep training at depth 5 until performance is stable
10. Save model checkpoints and logs
```

---

## 21. System Flow Diagram

```text
Depth Datasets
     ↓
CurriculumManager
     ↓
RubiksCubeEnv.reset()
     ↓
PPO Agent Rollout Collection
     ↓
Reward + Advantage Calculation
     ↓
Actor-Critic Update
     ↓
Evaluation Loop
     ↓
Curriculum Advancement Decision
     ↓
Increase Depth or Continue Training
```

---

## 22. Acceptance Criteria

Curriculum learning is considered successfully implemented when:

- The environment can reset from a selected scramble depth.
- The curriculum manager can sample states from depth-specific files.
- The current depth starts at 1.
- The current depth increases only after evaluation success criteria are met.
- The agent can train on mixed previous/current depths.
- Evaluation metrics are logged.
- Curriculum depth changes are logged.
- The PPO algorithm remains independent of curriculum-control logic.
- The system supports depth 1 through depth 5 for version 1.

---

## 23. Key Design Rule

The most important design rule is:

```text
Do not put curriculum logic inside the PPO loss function.
```

Curriculum learning should be handled by:

```text
Dataset sampling + environment reset + evaluation-based progression
```

The PPO agent should simply learn from the states, actions, rewards, and advantages it receives.

---

## 24. Summary

For this Rubik's Cube project, curriculum learning requires a structured training system that starts the agent near the solved cube and gradually increases scramble difficulty.

The required components are:

- Depth-specific scramble datasets
- Curriculum manager
- Curriculum-aware environment reset
- Depth-based timeout rule
- Evaluation system
- Advancement criteria
- Logging system
- Normal PPO training loop

The core idea is:

```text
Start easy.
Measure performance.
Increase difficulty only when the agent is ready.
Keep earlier depths available so the agent does not forget them.
```
