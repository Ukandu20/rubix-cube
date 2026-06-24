# Updated Custom Gym Environment Specification: Rubik’s Cube PPO Solver

## 1. Environment Purpose

The environment is a custom Gym/Gymnasium-style reinforcement learning environment for training an agent to solve a 3×3 Rubik’s Cube from mixed configurations.

The agent’s main objective is:

```text
Reach the solved cube state from a pre-generated scrambled cube state using the shortest possible sequence of legal moves.
```

The environment should support PPO training, curriculum learning, random baseline evaluation, and depth-based performance tracking.

For the first version of this project, the environment will focus on cube states at depths `1–5`. These states have already been pre-generated and stored in separate CSV or parquet files. Each state is unique by its encoded cube-state string.

The environment should not randomly generate scrambles for depths `1–5` during training. Instead, it should sample from the existing pre-generated state files.

---

## 2. Environment Name

Recommended name:

```text
RubixCubeSolve-v0
```

Possible class name:

```python
class RubixCubeSolveEnv(gym.Env):
    ...
```

---

## 3. Core Reinforcement Learning Loop

The environment follows the standard RL structure:

```text
state → action → reward → next state → done
```

At each time step:

```text
1. The agent observes the current cube state.
2. The agent chooses one of 12 legal Rubik’s Cube moves.
3. The environment applies the move.
4. The environment checks whether the cube is solved.
5. The environment gives a reward.
6. The episode continues until the cube is solved or the max step limit is reached.
```

The environment handles the cube simulation and reward logic.

The PPO algorithm handles policy learning, value estimation, advantage calculation, clipping, and network updates.

---

## 4. State Source

### 4.1 Dataset-Driven State Sampling

For scramble depths `1–5`, the environment should use pre-generated cube states stored in separate files.

Example folder structure:

```text
data/
  processed/
    training/
        parquet/
            depth_1.parquet
            depth_2.parquet
            depth_3.parquet
            depth_4.parquet
            depth_5.parquet
```

or:

```text
data/
  processed/
    training/
        csv/
            depth_1.csv
            depth_2.csv
            depth_3.csv
            depth_4.csv
            depth_5.csv
```

Each file contains cube states that belong to that depth.

The environment should sample from these files during `reset()`.

The old random scramble-generation method should be kept only as an optional fallback for:

```text
debugging
future depths beyond 5
generating new datasets
testing cube move logic
```

For the main training setup, depth `1–5` should come from the pre-generated files.

---

## 5. Encoded State Format

### 5.1 Current Dataset Format

Each cube state is stored as a 54-character string.

Example:

```text
BYYBYYBYYOOOOOOOOOYGGYGGYGGGWWGWWGWWRRRRRRRRRBBWBBWBBW
```

Each character represents one sticker color on the cube.

Allowed color characters:

```text
Y = Yellow
O = Orange
G = Green
W = White
R = Red
B = Blue
```

A valid 3×3 Rubik’s Cube state must contain:

```text
54 total characters
9 Y stickers
9 O stickers
9 G stickers
9 W stickers
9 R stickers
9 B stickers
```

The order of characters is important. The environment must never sort, reorder, or group the characters except when applying valid cube moves.

---

### 5.2 Face Blocks

The 54-character string can be interpreted as six blocks of nine stickers:

```text
BYYBYYBYY | OOOOOOOOO | YGGYGGYGG | GWWGWWGWW | RRRRRRRRR | BBWBBWBBW
```

Each block represents one face-position group of the cube.

Recommended face index layout:

```text
Face 0: indices  0–8
Face 1: indices  9–17
Face 2: indices 18–26
Face 3: indices 27–35
Face 4: indices 36–44
Face 5: indices 45–53
```

Each face block follows this 3×3 layout:

```text
0 1 2
3 4 5
6 7 8
```

The full flattened sticker layout is:

```text
Face 0:
0 1 2
3 4 5
6 7 8

Face 1:
9 10 11
12 13 14
15 16 17

Face 2:
18 19 20
21 22 23
24 25 26

Face 3:
27 28 29
30 31 32
33 34 35

Face 4:
36 37 38
39 40 41
42 43 44

Face 5:
45 46 47
48 49 50
51 52 53
```

The cube move functions must use the exact same indexing convention as the dataset generator. If the dataset generator and the environment move functions use different face orders, training results will be invalid.

---

## 6. Solved State

The solved state should use the same face order as the encoded dataset.

Based on the current encoded format, a likely solved state is:

```python
SOLVED_STATE_STRING = (
    "YYYYYYYYY"
    "OOOOOOOOO"
    "GGGGGGGGG"
    "WWWWWWWWW"
    "RRRRRRRRR"
    "BBBBBBBBB"
)
```

This means:

```text
Face 0 solved color: Y
Face 1 solved color: O
Face 2 solved color: G
Face 3 solved color: W
Face 4 solved color: R
Face 5 solved color: B
```

The environment should define this explicitly:

```python
SOLVED_STATE_STRING = "YYYYYYYYYOOOOOOOOOGGGGGGGGGWWWWWWWWWRRRRRRRRRBBBBBBBBB"
```

Then decode it into the integer observation format:

```python
self.solved_state = self.decode_state(SOLVED_STATE_STRING)
```

The `is_solved()` method should compare the current cube state against this solved state.

Example:

```python
def is_solved(self) -> bool:
    return np.array_equal(self.cube_state, self.solved_state)
```

---

## 7. State Decoding

The raw dataset stores cube states as strings, but the PPO model should receive numeric observations.

The environment should decode each state string into a 54-value integer array.

Recommended color mapping:

```python
COLOR_TO_INT = {
    "Y": 0,
    "O": 1,
    "G": 2,
    "W": 3,
    "R": 4,
    "B": 5,
}

INT_TO_COLOR = {
    0: "Y",
    1: "O",
    2: "G",
    3: "W",
    4: "R",
    5: "B",
}
```

Example:

```text
BYYBYYBYY...
```

becomes:

```text
[5, 0, 0, 5, 0, 0, 5, 0, 0, ...]
```

Recommended decoder:

```python
def decode_state(encoded_state: str) -> np.ndarray:
    encoded_state = encoded_state.strip()

    if len(encoded_state) != 54:
        raise ValueError(
            f"Encoded state must contain 54 characters, got {len(encoded_state)}."
        )

    invalid_colors = set(encoded_state) - set(COLOR_TO_INT.keys())
    if invalid_colors:
        raise ValueError(f"Invalid color labels found: {invalid_colors}")

    decoded_state = np.array(
        [COLOR_TO_INT[color] for color in encoded_state],
        dtype=np.int8
    )

    return decoded_state
```

The environment may also include an encoder for debugging:

```python
def encode_state(decoded_state: np.ndarray) -> str:
    return "".join(INT_TO_COLOR[int(value)] for value in decoded_state)
```

---

## 8. Observation Space

The MVP observation should be a flat 54-value integer array.

```python
self.observation_space = spaces.Box(
    low=0,
    high=5,
    shape=(54,),
    dtype=np.int8
)
```

Each value represents the color of one sticker.

Example observation:

```python
np.array(
    [5, 0, 0, 5, 0, 0, 5, 0, 0, 1, 1, 1, ...],
    dtype=np.int8
)
```

This representation is simple and easy to debug.

---

## 9. Optional Future Observation Encoding

For later versions, the environment may support one-hot encoding.

One-hot shape:

```text
54 stickers × 6 colors = 324 values
```

Possible observation space:

```python
self.observation_space = spaces.Box(
    low=0,
    high=1,
    shape=(324,),
    dtype=np.int8
)
```

This may help neural networks because the model will not treat color `5` as numerically larger than color `1`.

Recommended MVP:

```text
Use the flat 54-integer state first.
Move to one-hot encoding later if PPO struggles.
```

---

## 10. Action Space

The environment should define 12 discrete actions.

```python
self.action_space = spaces.Discrete(12)
```

The actions correspond to the six standard Rubik’s Cube face turns and their counter-clockwise variants.

| Action ID | Move | Meaning                      |
| --------: | ---- | ---------------------------- |
|         0 | U    | Up face clockwise            |
|         1 | U'   | Up face counter-clockwise    |
|         2 | R    | Right face clockwise         |
|         3 | R'   | Right face counter-clockwise |
|         4 | F    | Front face clockwise         |
|         5 | F'   | Front face counter-clockwise |
|         6 | D    | Down face clockwise          |
|         7 | D'   | Down face counter-clockwise  |
|         8 | L    | Left face clockwise          |
|         9 | L'   | Left face counter-clockwise  |
|        10 | B    | Back face clockwise          |
|        11 | B'   | Back face counter-clockwise  |

Recommended mapping:

```python
ACTION_TO_MOVE = {
    0: "U",
    1: "U'",
    2: "R",
    3: "R'",
    4: "F",
    5: "F'",
    6: "D",
    7: "D'",
    8: "L",
    9: "L'",
    10: "B",
    11: "B'",
}
```

Inverse action mapping:

```python
INVERSE_ACTION = {
    0: 1,
    1: 0,
    2: 3,
    3: 2,
    4: 5,
    5: 4,
    6: 7,
    7: 6,
    8: 9,
    9: 8,
    10: 11,
    11: 10,
}
```

This inverse mapping is used for the immediate inverse-move penalty.

---

## 11. Dataset Loading

The environment should load the state files during initialization.

Recommended constructor parameters:

```python
def __init__(
    self,
    state_files: dict[int, str],
    scramble_depth: int = 1,
    scramble_depth_range: tuple[int, int] | None = None,
    max_episode_steps: int = 50,
    state_column: str = "state_encoded",
    file_format: str = "parquet",
    reward_config: dict | None = None,
    render_mode: str | None = None,
):
    ...
```

Example config:

```python
state_files = {
    1: "data/processed/training/parquet/depth_1.parquet",
    2: "data/processed/training/parquet/depth_2.parquet",
    3: "data/processed/training/parquet/depth_3.parquet",
    4: "data/processed/training/parquet/depth_4.parquet",
    5: "data/processed/training/parquet/depth_5.parquet",
}
```

The environment should store the loaded states by depth:

```python
self.states_by_depth = {
    1: dataframe_for_depth_1,
    2: dataframe_for_depth_2,
    3: dataframe_for_depth_3,
    4: dataframe_for_depth_4,
    5: dataframe_for_depth_5,
}
```

The environment should support both CSV and parquet:

```python
if file_path.endswith(".parquet"):
    df = pd.read_parquet(file_path)
elif file_path.endswith(".csv"):
    df = pd.read_csv(file_path)
else:
    raise ValueError("Unsupported file format. Use CSV or parquet.")
```

---

## 12. Recommended Dataset Columns

Minimum required column:

| Column          | Required | Purpose                        |
| --------------- | -------: | ------------------------------ |
| `state_encoded` |      Yes | 54-character cube state string |

Recommended optional columns:

| Column               |            Required | Purpose                                |
| -------------------- | ------------------: | -------------------------------------- |
| `scramble_depth`     | Optional but useful | Starting depth of the state            |
| `sample_id`          |            Optional | Unique ID or hash for tracking         |
| `scramble_moves`     |            Optional | Move sequence that generated the state |
| `solution_moves`     |            Optional | Known inverse solution path            |
| `encoded_state_hash` |            Optional | Useful for uniqueness validation       |

Only `state_encoded` should be used as the observation.

The agent should not receive:

```text
scramble_depth
scramble_moves
solution_moves
sample_id
```

These can be included in the `info` dictionary for logging, debugging, and evaluation.

---

## 13. Dataset Validation

Before training, each file should be validated.

Required checks:

```text
Each encoded state has exactly 54 characters.
Each state contains only Y, O, G, W, R, B.
Each color appears exactly 9 times.
There are no duplicate encoded states within the same file.
There are no duplicate encoded states across different depth files.
The solved state is not included in depth 1–5 files.
Each file contains at least one state.
```

Validation function:

```python
from collections import Counter

def validate_encoded_state(encoded_state: str) -> bool:
    encoded_state = encoded_state.strip()

    if len(encoded_state) != 54:
        return False

    allowed_colors = {"Y", "O", "G", "W", "R", "B"}

    if set(encoded_state) - allowed_colors:
        return False

    counts = Counter(encoded_state)

    for color in allowed_colors:
        if counts[color] != 9:
            return False

    return True
```

Important warning:

```text
Basic validation proves that the sticker string is well-formed.
It does not prove that the cube state is physically reachable.
```

Because your states are pre-generated by your cube engine, physical reachability depends on the correctness of that generator.

---

## 14. Reset Behavior

The `reset()` method starts a new episode by sampling one state from the pre-generated dataset.

Required reset behavior:

```text
1. Reset the current step count to 0.
2. Clear move history.
3. Clear previous action.
4. Choose a start depth.
5. Sample one encoded state from that depth file.
6. Decode the encoded state into a 54-value integer array.
7. Set the decoded array as the current cube state.
8. Return the observation and info dictionary.
```

The environment should support two depth modes.

### 14.1 Fixed Depth Mode

If `scramble_depth=3` and `scramble_depth_range=None`, then every episode starts from depth 3.

Example:

```python
env = RubixCubeSolveEnv(
    state_files=state_files,
    scramble_depth=3,
    scramble_depth_range=None,
    max_episode_steps=50
)
```

### 14.2 Depth Range Mode

If `scramble_depth_range=(1, 5)`, then each episode randomly selects one depth between 1 and 5.

Example:

```python
env = RubixCubeSolveEnv(
    state_files=state_files,
    scramble_depth_range=(1, 5),
    max_episode_steps=50
)
```

Recommended sampling strategy:

```text
Choose depth uniformly first.
Then choose a random state from that depth.
```

This prevents depth 5 from dominating training simply because it may contain many more states than depth 1.

---

## 15. Reset Pseudocode

```python
def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    self.current_step = 0
    self.move_history = []
    self.last_action = None
    self.episode_return = 0.0

    if self.scramble_depth_range is not None:
        min_depth, max_depth = self.scramble_depth_range
        self.current_depth = int(
            self.np_random.integers(min_depth, max_depth + 1)
        )
    else:
        self.current_depth = self.scramble_depth

    row = self.sample_state_from_depth(self.current_depth)

    encoded_state = row[self.state_column]

    self.cube_state = self.decode_state(encoded_state)

    observation = self.get_observation()

    info = {
        "state_source": "precomputed",
        "start_depth": self.current_depth,
        "encoded_state": encoded_state,
        "state_id": row.get("state_id", None),
        "scramble_sequence": row.get("scramble_sequence", None),
        "inverse_solution": row.get("inverse_solution", None),
        "current_step": self.current_step,
        "is_solved": self.is_solved(),
    }

    return observation, info
```

---

## 16. State Sampling Function

The environment should include a method for sampling states by depth.

```python
def sample_state_from_depth(self, depth: int):
    if depth not in self.states_by_depth:
        raise ValueError(f"No precomputed states available for depth {depth}.")

    df = self.states_by_depth[depth]

    if len(df) == 0:
        raise ValueError(f"Depth {depth} file contains no states.")

    random_index = int(self.np_random.integers(0, len(df)))

    return df.iloc[random_index]
```

This function should return the row containing the encoded state and any optional metadata.

---

## 17. Step Behavior

The `step(action)` method applies one cube move and returns the result.

Required step behavior:

```text
1. Validate that the action is between 0 and 11.
2. Check whether the action immediately reverses the previous action.
3. Apply the selected cube move to the current cube state.
4. Increment the step counter.
5. Store the action in move history.
6. Check whether the cube is solved.
7. Calculate reward.
8. Check whether the episode is terminated or truncated.
9. Return observation, reward, terminated, truncated, and info.
```

Recommended step output:

```python
observation, reward, terminated, truncated, info = env.step(action)
```

Where:

```text
terminated = True if the cube is solved.
truncated = True if the max step limit is reached without solving.
```

---

## 18. Step Pseudocode

```python
def step(self, action: int):
    if not self.action_space.contains(action):
        raise ValueError(f"Invalid action: {action}")

    immediate_inverse = (
        self.last_action is not None
        and action == INVERSE_ACTION[self.last_action]
    )

    move = ACTION_TO_MOVE[action]

    self.apply_action(action)

    self.current_step += 1
    self.move_history.append(action)

    solved = self.is_solved()

    reward = self.calculate_reward(
        solved=solved,
        immediate_inverse=immediate_inverse
    )

    terminated = solved
    truncated = False

    if self.current_step >= self.max_episode_steps and not solved:
        truncated = True
        reward += self.timeout_penalty

    self.last_action = action
    self.episode_return += reward

    observation = self.get_observation()

    info = {
        "is_solved": solved,
        "start_depth": self.current_depth,
        "current_step": self.current_step,
        "max_episode_steps": self.max_episode_steps,
        "last_action": action,
        "last_move": move,
        "immediate_inverse_move": immediate_inverse,
        "move_history": self.move_history.copy(),
        "move_history_notation": [
            ACTION_TO_MOVE[a] for a in self.move_history
        ],
        "episode_return": self.episode_return,
        "terminated_reason": (
            "solved" if terminated
            else "max_steps_reached" if truncated
            else "running"
        ),
    }

    return observation, reward, terminated, truncated, info
```

---

## 19. Reward Structure

The environment should use the following reward values:

```text
Ordinary move penalty: -1
Solved-state bonus: +100
Immediate inverse-move penalty: -5
Timeout penalty: -10
```

Recommended config:

```python
reward_config = {
    "move_penalty": -1.0,
    "solve_bonus": 100.0,
    "inverse_move_penalty": -5.0,
    "timeout_penalty": -10.0,
}
```

Reward formula:

```text
R_t = r_move + I_solved × B_solve + I_inverse × P_inverse + I_timeout × P_timeout
```

Where:

```text
r_move = -1
B_solve = +100
P_inverse = -5
P_timeout = -10

I_solved = 1 if the cube is solved after the move, otherwise 0
I_inverse = 1 if the move immediately reverses the previous move, otherwise 0
I_timeout = 1 if the episode times out without solving, otherwise 0
```

In plain language:

```text
Every move costs the agent 1 point.
Solving the cube gives the agent 100 points.
Immediately undoing the previous move costs an extra 5 points.
Failing to solve within the episode limit costs an extra 10 points.
```

---

## 20. Reward Examples

### Solving in 1 move

```text
-1 + 100 = +99
```

### Solving in 3 moves

```text
(-1 × 3) + 100 = +97
```

### Making an immediate inverse move

```text
-1 - 5 = -6
```

### Timing out after 50 moves

```text
(-1 × 50) - 10 = -60
```

### Making an inverse move that also solves

```text
-1 - 5 + 100 = +94
```

This is acceptable because solving should still be strongly rewarded.

---

## 21. Reward Calculation Pseudocode

```python
def calculate_reward(self, solved: bool, immediate_inverse: bool) -> float:
    reward = self.move_penalty

    if immediate_inverse:
        reward += self.inverse_move_penalty

    if solved:
        reward += self.solve_bonus

    return reward
```

The timeout penalty should be added inside `step()` only when the episode reaches the max step count without solving.

```python
if self.current_step >= self.max_episode_steps and not solved:
    truncated = True
    reward += self.timeout_penalty
```

---

## 22. Episode Termination

The user should be able to specify the maximum episode length.

Default:

```python
max_episode_steps = 50
```

The episode can end in two ways:

```text
1. Success termination:
   The cube reaches the solved state.

2. Failure truncation:
   The agent reaches max_episode_steps without solving.
```

Rules:

```python
terminated = solved
truncated = self.current_step >= self.max_episode_steps and not solved
```

The environment should not continue after either `terminated=True` or `truncated=True`.

---

## 23. Cube Move Engine

The environment must correctly apply all 12 actions.

Required move functions:

```python
move_U()
move_U_prime()
move_D()
move_D_prime()
move_L()
move_L_prime()
move_R()
move_R_prime()
move_F()
move_F_prime()
move_B()
move_B_prime()
```

Alternative implementation:

```python
def apply_action(self, action: int):
    move = ACTION_TO_MOVE[action]
    self.cube_state = apply_move(self.cube_state, move)
```

The cube move engine must be tested carefully before PPO training.

Required cube engine tests:

```text
Applying U then U' returns to the original state.
Applying D then D' returns to the original state.
Applying L then L' returns to the original state.
Applying R then R' returns to the original state.
Applying F then F' returns to the original state.
Applying B then B' returns to the original state.
Applying each move four times returns to the original state.
The solved state is recognized as solved.
A depth 1 state becomes solved after its correct inverse move.
A sampled dataset state remains valid after each move.
```

Important:

```text
The environment move logic must match the same sticker ordering used when the depth files were generated.
```

---

## 24. Info Dictionary

The `info` dictionary should include useful debugging and evaluation details.

Recommended fields:

```python
info = {
    "state_source": "precomputed",
    "start_depth": self.current_depth,
    "encoded_state": encoded_state,
    "state_id": state_id,
    "scramble_sequence": scramble_sequence,
    "inverse_solution": inverse_solution,
    "is_solved": solved,
    "current_step": self.current_step,
    "max_episode_steps": self.max_episode_steps,
    "last_action": action,
    "last_move": move,
    "immediate_inverse_move": immediate_inverse,
    "move_history": self.move_history.copy(),
    "move_history_notation": move_history_notation,
    "episode_return": self.episode_return,
    "terminated_reason": terminated_reason,
}
```

The `info` dictionary can contain the `inverse_solution`, but the observation must not contain it.

The policy network should only receive the cube state.

---

## 25. Rendering

The environment should support simple text rendering.

Recommended render modes:

```text
None
"text"
"human"
```

For the MVP, text rendering is enough.

Example render:

```text
Step: 4 / 50
Start depth: 3
Last move: R'
Solved: False

Face 0:
B Y Y
B Y Y
B Y Y

Face 1:
O O O
O O O
O O O

Face 2:
Y G G
Y G G
Y G G

Face 3:
G W W
G W W
G W W

Face 4:
R R R
R R R
R R R

Face 5:
B B W
B B W
B B W
```

The render method is mainly for debugging. It is not needed for training.

---

## 26. Curriculum Learning Design

Because states are stored by depth, curriculum learning can be implemented cleanly.

Recommended training phases:

```text
Phase 1: Train only on depth 1.
Phase 2: Train on depths 1–2.
Phase 3: Train on depths 1–3.
Phase 4: Train on depths 1–4.
Phase 5: Train on depths 1–5.
```

Example environment configurations:

```python
# Phase 1
env = RubixCubeSolveEnv(
    state_files=state_files,
    scramble_depth=1,
    max_episode_steps=50
)

# Phase 3
env = RubixCubeSolveEnv(
    state_files=state_files,
    scramble_depth_range=(1, 3),
    max_episode_steps=50
)

# Phase 5
env = RubixCubeSolveEnv(
    state_files=state_files,
    scramble_depth_range=(1, 5),
    max_episode_steps=50
)
```

Recommended depth sampling:

```text
Uniform by depth, then random within that depth.
```

This means:

```text
For depth range 1–5:
20% chance depth 1
20% chance depth 2
20% chance depth 3
20% chance depth 4
20% chance depth 5
```

This is better than sampling from one combined file because deeper depths may have far more states.

---

## 27. Training Hyperparameters for PPO

Suggested PPO starting values:

| Hyperparameter       | Suggested Value | Notes                                     |
| -------------------- | --------------: | ----------------------------------------- |
| `policy`             |     `MlpPolicy` | Good starting point for flat observations |
| `learning_rate`      |          `3e-4` | Standard PPO starting value               |
| `gamma`              |          `0.99` | Rewards future solving success            |
| `gae_lambda`         |          `0.95` | Common advantage estimation value         |
| `clip_range`         |           `0.2` | Standard PPO clipping value               |
| `n_epochs`           |            `10` | Optimization passes per update            |
| `n_steps`            |          `2048` | Rollout steps before update               |
| `batch_size`         |            `64` | Batch size for PPO updates                |
| `mini_batch_size`    |            `64` | Same as batch size for initial setup      |
| `max_episode_length` |            `50` | Same as environment max steps             |
| `ent_coef`           |          `0.01` | Encourages exploration early              |
| `vf_coef`            |           `0.5` | Value loss coefficient                    |
| `max_grad_norm`      |           `0.5` | Gradient clipping                         |
| `target_kl`          |          `0.03` | Optional PPO stability limit              |

Recommended starting config:

```python
ppo_config = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.03,
}
```

If the agent remains too random after learning begins, reduce entropy:

```python
ent_coef = 0.001
```

or:

```python
ent_coef = 0.0
```

---

## 28. Environment Configuration Summary

Recommended MVP environment config:

```python
env_config = {
    "state_files": {
        1: "data/cube_states/depth_1.parquet",
        2: "data/cube_states/depth_2.parquet",
        3: "data/cube_states/depth_3.parquet",
        4: "data/cube_states/depth_4.parquet",
        5: "data/cube_states/depth_5.parquet",
    },
    "state_column": "encoded_state",
    "scramble_depth": 1,
    "scramble_depth_range": None,
    "max_episode_steps": 50,
    "state_encoding": "integer_flat",
    "reward_config": {
        "move_penalty": -1.0,
        "solve_bonus": 100.0,
        "inverse_move_penalty": -5.0,
        "timeout_penalty": -10.0,
    },
    "render_mode": None,
}
```

For curriculum training:

```python
env_config = {
    "state_files": {
        1: "data/cube_states/depth_1.parquet",
        2: "data/cube_states/depth_2.parquet",
        3: "data/cube_states/depth_3.parquet",
        4: "data/cube_states/depth_4.parquet",
        5: "data/cube_states/depth_5.parquet",
    },
    "state_column": "encoded_state",
    "scramble_depth": None,
    "scramble_depth_range": (1, 5),
    "max_episode_steps": 50,
    "state_encoding": "integer_flat",
    "reward_config": {
        "move_penalty": -1.0,
        "solve_bonus": 100.0,
        "inverse_move_penalty": -5.0,
        "timeout_penalty": -10.0,
    },
    "render_mode": None,
}
```

---

## 29. Success Metrics

The environment should support evaluation by depth.

Primary metric:

```text
solve_rate = solved_episodes / total_episodes
```

Track this separately for each depth:

```text
Depth 1 solve rate
Depth 2 solve rate
Depth 3 solve rate
Depth 4 solve rate
Depth 5 solve rate
```

Secondary metrics:

```text
average episode reward
average solution length
median solution length
average steps until solved
timeout rate
immediate inverse move rate
extra moves above starting depth
```

If the stored depth represents true minimum distance from solved, then:

```text
extra_moves = agent_solution_length - start_depth
```

Example:

```text
Start depth: 3
Agent solved in: 5 moves
Extra moves: 2
```

This is useful because the agent should not only solve the cube, but solve it efficiently.

---

## 30. MVP Success Targets

Recommended first-stage success targets:

```text
Depth 1: 95%+ solve rate
Depth 2: 80%+ solve rate
Depth 3: 60%+ solve rate
Depth 4: better than random baseline
Depth 5: better than random baseline
```

The first version should not be judged by whether it can solve arbitrary full Rubik’s Cube scrambles.

The first version should prove that:

```text
The cube engine works.
The dataset-driven reset works.
The reward function works.
PPO can learn shallow cube-solving behavior.
The trained agent performs better than a random agent.
```

---

## 31. Baselines

### 31.1 Random Agent Baseline

A random agent chooses one of the 12 moves uniformly at random.

Use it to compare:

```text
solve rate
average reward
timeout rate
average solution length
inverse move rate
```

The trained PPO agent should outperform the random baseline at each tested depth.

### 31.2 Oracle/Inverse-Solution Baseline

If the dataset includes `inverse_solution`, use it only for evaluation.

Useful oracle metrics:

```text
oracle_solution_length
agent_solution_length
extra_moves_used
agent_success
```

The oracle solution should not be given to the policy network.

---

## 32. Important Implementation Warnings

### 32.1 Do Not Expose Metadata to the Agent

The observation must only include the cube state.

Do not include these in the observation:

```text
depth
scramble sequence
inverse solution
state ID
known next best move
```

These are allowed in `info`, but not in the observation.

---

### 32.2 Validate Move Logic Before PPO

Before training, test the cube engine heavily.

If the move engine is wrong, PPO training results will be meaningless.

Required tests:

```text
Every move followed by its inverse returns to the same state.
Every move applied four times returns to the same state.
The solved state is correctly recognized.
Encoded and decoded solved states match.
Dataset states remain valid after legal moves.
For any state with a known inverse solution, applying that inverse solution solves the cube.
```

---

### 32.3 Keep Reward Simple at First

Start with:

```text
-1 per move
+100 for solving
-5 for immediate inverse move
-10 for timeout
```

Avoid adding too many shaping rewards immediately.

Do not add sticker-matching reward until the simple reward has been tested.

Sticker matching can be misleading because a necessary move may temporarily make the cube look less solved.

---

## 33. Final Environment Behavior Summary

At reset:

```text
Choose a depth.
Sample a pre-generated encoded state from that depth file.
Decode the 54-character string into a 54-value integer array.
Set that as the current cube state.
Return the observation.
```

At each step:

```text
Receive one action from the agent.
Apply the corresponding Rubik’s Cube move.
Check if the move immediately reversed the previous move.
Check if the cube is solved.
Calculate reward.
End the episode if solved or if max_episode_steps is reached.
Return the next observation, reward, termination flags, and info.
```

The reward is:

```text
-1 for every move
+100 for solving
-5 for immediate inverse moves
-10 for timeout
```

The main success measure is:

```text
High solve rate with low average solution length, measured separately for depths 1–5.
```

The environment’s main job is to provide a reliable Rubik’s Cube world for PPO to learn from. PPO should handle the learning process; the environment should handle state sampling, cube movement, reward calculation, and episode control.
