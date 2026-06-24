# PPO Agent Specification: Rubik’s Cube Reinforcement Learning Solver

## 1. Agent Purpose

The PPO agent is responsible for learning how to solve a 3×3 Rubik’s Cube from scrambled states provided by the custom Gym/Gymnasium environment.

The agent’s main goal is:

```text
Learn a policy that selects Rubik’s Cube moves which maximize long-term reward by reaching the solved state in as few moves as possible.
```

The environment provides:

```text
current cube state
available actions
immediate reward
next cube state
termination signal
```

The PPO agent learns from these interactions by updating an actor-critic neural network.

The actor learns:

```text
Which move should I choose from this cube state?
```

The critic learns:

```text
How promising is this cube state?
```

The agent should be trained first on shallow scrambles from depths `1–5`, using the dataset-driven environment that samples pre-generated cube states from CSV/parquet files.

---

## 2. Relationship Between Environment and PPO Agent

The environment and PPO agent must remain separate.

The environment defines the Rubik’s Cube task:

```text
state representation
legal moves
cube transition logic
reward function
episode length
solved-state detection
timeout detection
```

The PPO agent defines the learning process:

```text
actor network
critic network
action probability distribution
rollout collection
return calculation
advantage estimation
PPO clipping
policy update
value update
entropy regularization
evaluation
model saving
```

The environment reward does **not** follow the PPO policy.

Instead:

```text
The PPO policy learns to maximize the reward defined by the environment.
```

The correct flow is:

```text
Environment gives rewards.
PPO uses those rewards to improve the policy.
```

---

## 3. Environment Interface Expected by the Agent

The PPO agent expects the environment to follow the standard Gym/Gymnasium pattern.

At reset:

```python
observation, info = env.reset()
```

At each step:

```python
next_observation, reward, terminated, truncated, info = env.step(action)
```

The PPO agent receives:

```text
observation = current cube state
reward = immediate reward after action
terminated = True if solved
truncated = True if max episode length reached
info = debugging/evaluation metadata
```

The agent should treat an episode as finished when:

```python
done = terminated or truncated
```

---

## 4. Observation Input

The MVP environment uses a flat 54-value integer observation.

Each value represents one sticker color.

Example:

```python
observation = np.array(
    [5, 0, 0, 5, 0, 0, 5, 0, 0, 1, 1, 1, ...],
    dtype=np.int8
)
```

Observation shape:

```python
(54,)
```

Observation space:

```python
spaces.Box(low=0, high=5, shape=(54,), dtype=np.int8)
```

The PPO model should internally convert this input to a floating-point tensor before passing it through the neural network.

Recommended preprocessing:

```python
observation = observation.astype(np.float32)
```

Optional normalization:

```python
observation = observation / 5.0
```

This maps sticker values from:

```text
0–5
```

to:

```text
0.0–1.0
```

Recommended MVP input:

```text
54-value normalized float vector
```

Later improvement:

```text
324-value one-hot encoded sticker vector
```

---

## 5. Action Output

The action space has 12 discrete actions.

The actor network outputs a probability distribution over these 12 actions.

Action mapping:

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

The actor does not output a move directly as text.

It outputs probabilities:

```text
π(a | s)
```

Example actor output:

```text
U:  0.05
U': 0.25
D:  0.04
D': 0.03
L:  0.08
L': 0.06
R:  0.22
R': 0.09
F:  0.04
F': 0.03
B:  0.07
B': 0.04
```

During training, the action should usually be sampled from this distribution.

During evaluation, the action can be chosen greedily:

```python
action = argmax(action_probabilities)
```

---

## 6. PPO Agent Goal

The PPO agent should learn a policy that maximizes expected discounted return.

The policy is written as:

```text
πθ(a_t | s_t)
```

Where:

```text
π = policy
θ = policy network parameters
s_t = cube state at time step t
a_t = selected cube move at time step t
```

The objective is:

```text
Choose actions that maximize long-term reward.
```

For this project, that means:

```text
Solve the cube.
Use fewer moves.
Avoid immediate inverse moves.
Avoid timeout.
```

The agent is not directly trying to imitate the scramble sequence or inverse solution. It is learning from rewards.

---

## 7. Immediate Reward from the Environment

The environment provides one immediate reward after each move.

Use lowercase `r_t` for immediate reward.

For this project:

```text
r_t = -1 + I_solved(100) + I_inverse(-5) + I_timeout(-10)
```

Where:

```text
-1 = ordinary move penalty
+100 = solved-state bonus
-5 = immediate inverse-move penalty
-10 = timeout penalty
```

Indicator variables:

```text
I_solved = 1 if the cube is solved after the move, otherwise 0
I_inverse = 1 if the move immediately reverses the previous move, otherwise 0
I_timeout = 1 if the episode times out without solving, otherwise 0
```

Examples:

```text
Solves in 1 move: -1 + 100 = +99

Solves in 3 moves:
(-1 × 3) + 100 = +97 total episode reward

Immediate inverse move:
-1 - 5 = -6

Timeout after 50 moves:
(-1 × 50) - 10 = -60
```

This reward is defined by the environment, not by PPO.

---

## 8. Discounted Return Used by PPO

The PPO agent does not optimize only the immediate reward.

It uses the discounted return-to-go.

Use uppercase `G_t` for discounted return.

For finite episodes:

```text
G_t = ∑ from k=0 to T-t of γ^k r_{t+k}
```

Expanded:

```text
G_t = r_t + γr_{t+1} + γ²r_{t+2} + γ³r_{t+3} + ...
```

Where:

```text
G_t = discounted return from time step t
γ = discount factor
r_t = immediate reward from environment
T = final time step of the episode
```

Recommended discount factor:

```python
gamma = 0.99
```

Why this matters:

```text
Earlier moves may receive -1 immediately, but if they lead to solving the cube later, PPO should still learn that those moves were useful.
```

Example:

```text
Move 1 reward: -1
Move 2 reward: -1
Move 3 reward: +99
```

Then:

```text
G_0 = -1 + γ(-1) + γ²(99)
```

With `γ = 0.99`:

```text
G_0 = -1 + 0.99(-1) + 0.99²(99)
G_0 ≈ 95.04
```

So PPO can give credit to the first move even though its immediate reward was negative.

---

## 9. Critic Value Function

The critic estimates the expected future return from a given cube state.

The critic outputs:

```text
Vφ(s_t)
```

Where:

```text
V = value function
φ = critic network parameters
s_t = current cube state
```

Example:

```text
V(s_t) = 72.5
```

This means:

```text
The critic predicts that from this cube state, the agent can expect about 72.5 future reward.
```

The critic does not choose actions.

It only estimates how good or promising the current cube state is.

---

## 10. Advantage Estimation

The advantage tells PPO whether an action was better or worse than expected.

Simple advantage formula:

```text
Â_t = G_t - V(s_t)
```

Where:

```text
Â_t = advantage estimate
G_t = discounted return-to-go
V(s_t) = critic value estimate
```

Interpretation:

```text
If Â_t > 0:
The action performed better than expected.
PPO should make this action more likely in similar states.

If Â_t < 0:
The action performed worse than expected.
PPO should make this action less likely in similar states.
```

Example:

```text
G_t = 95
V(s_t) = 70

Â_t = 95 - 70 = 25
```

The action was better than expected.

Another example:

```text
G_t = -20
V(s_t) = 30

Â_t = -20 - 30 = -50
```

The action was worse than expected.

---

## 11. Generalized Advantage Estimation

In practice, PPO usually uses Generalized Advantage Estimation, also called GAE.

GAE provides a smoother advantage estimate by combining immediate temporal-difference errors across multiple future steps.

Temporal-difference error:

```text
δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

GAE advantage:

```text
Â_t = ∑ from k=0 to T-t of (γλ)^k δ_{t+k}
```

Where:

```text
γ = discount factor
λ = GAE lambda
δ_t = temporal-difference error
```

Recommended value:

```python
gae_lambda = 0.95
```

Interpretation:

```text
λ close to 1:
Uses longer-term reward information, but may be noisier.

λ close to 0:
Uses shorter-term value estimates, but may be more biased.
```

Recommended MVP:

```text
Use GAE with gamma = 0.99 and gae_lambda = 0.95.
```

---

## 12. Actor-Critic Architecture

The PPO agent should use an actor-critic neural network.

There are two acceptable designs:

```text
1. Shared feature extractor with separate actor and critic heads.
2. Separate actor and critic networks.
```

For the MVP, use a shared feature extractor.

Recommended architecture:

```text
Input: 54 values

Shared hidden layer 1: 256 units
Activation: ReLU

Shared hidden layer 2: 256 units
Activation: ReLU

Actor head:
12 output logits

Critic head:
1 scalar value
```

The actor head outputs logits, not probabilities directly.

The logits are passed through a softmax distribution to create action probabilities.

The critic head outputs one number:

```text
V(s)
```

Recommended MVP architecture:

```python
network_architecture = {
    "input_dim": 54,
    "hidden_layers": [256, 256],
    "activation": "ReLU",
    "actor_output_dim": 12,
    "critic_output_dim": 1,
}
```

---

## 13. Larger Optional Architecture

If the 54-value encoding is too limited, a larger network can be tested.

Possible larger architecture:

```text
Input: 54 values

Hidden layer 1: 512 units
Hidden layer 2: 256 units
Hidden layer 3: 128 units

Actor head: 12 logits
Critic head: 1 value
```

Config:

```python
network_architecture = {
    "input_dim": 54,
    "hidden_layers": [512, 256, 128],
    "activation": "ReLU",
    "actor_output_dim": 12,
    "critic_output_dim": 1,
}
```

Recommendation:

```text
Start with [256, 256].
Only increase the model size if the smaller model clearly underfits.
```

---

## 14. Optional One-Hot Input Architecture

If the environment later switches to one-hot state encoding, the input dimension becomes:

```text
54 stickers × 6 colors = 324 values
```

Then the architecture becomes:

```python
network_architecture = {
    "input_dim": 324,
    "hidden_layers": [512, 256],
    "activation": "ReLU",
    "actor_output_dim": 12,
    "critic_output_dim": 1,
}
```

This may improve learning because color labels are treated as categories instead of numeric magnitudes.

Recommended progression:

```text
Version 1: 54-value integer/normalized input
Version 2: 324-value one-hot input
```

---

## 15. Action Selection

During training, the agent should sample actions from the policy distribution.

Training action selection:

```python
action_distribution = Categorical(logits=actor_logits)
action = action_distribution.sample()
log_prob = action_distribution.log_prob(action)
entropy = action_distribution.entropy()
```

The sampled action is sent to the environment:

```python
next_obs, reward, terminated, truncated, info = env.step(action)
```

During evaluation, the agent can use deterministic action selection:

```python
action = argmax(action_probabilities)
```

Recommended:

```text
Training: stochastic sampling
Evaluation: deterministic argmax
```

This allows the agent to explore during learning and behave consistently during evaluation.

---

## 16. Rollout Collection

PPO trains using batches of experience called rollouts.

A rollout contains multiple transitions collected from the environment.

Each transition should store:

```text
state
action
reward
next_state
terminated
truncated
done
log probability of action under old policy
critic value estimate
start depth
info metadata
```

Recommended rollout buffer fields:

```python
rollout_buffer = {
    "observations": [],
    "actions": [],
    "rewards": [],
    "dones": [],
    "log_probs": [],
    "values": [],
    "depths": [],
    "infos": [],
}
```

The rollout size is controlled by:

```python
n_steps = 2048
```

This means PPO collects 2048 environment steps before performing a policy update.

If using multiple parallel environments, total rollout samples are:

```text
n_steps × num_envs
```

Example:

```text
n_steps = 2048
num_envs = 4

Total rollout samples = 8192
```

---

## 17. PPO Clipping Objective

PPO updates the policy using a clipped objective.

First, calculate the probability ratio:

```text
ratio_t = π_new(a_t | s_t) / π_old(a_t | s_t)
```

Using log probabilities:

```text
ratio_t = exp(log_prob_new - log_prob_old)
```

Then calculate the clipped policy objective:

```text
L_clip = min(
    ratio_t × Â_t,
    clip(ratio_t, 1 - ε, 1 + ε) × Â_t
)
```

Where:

```text
ε = clipping range
```

Recommended:

```python
clip_range = 0.2
```

This means PPO prevents the new policy from moving too far away from the old policy in one update.

With `clip_range = 0.2`, the ratio is clipped to:

```text
0.8 to 1.2
```

Interpretation:

```text
PPO wants to improve the policy, but not so aggressively that training becomes unstable.
```

---

## 18. Actor Loss

The actor loss is based on the negative clipped objective.

Because optimizers usually minimize loss, the objective is negated:

```text
actor_loss = -mean(L_clip)
```

The actor should be updated to:

```text
increase probability of actions with positive advantage
decrease probability of actions with negative advantage
avoid overly large policy updates
```

The actor is responsible for learning better cube moves.

---

## 19. Critic Loss

The critic should learn to predict the return from each state.

The value loss compares:

```text
predicted value V(s_t)
target return G_t
```

Recommended critic loss:

```text
critic_loss = mean((G_t - V(s_t))²)
```

This is mean squared error.

The critic loss is weighted by:

```python
vf_coef = 0.5
```

The critic helps reduce noisy policy updates by estimating whether a state is better or worse than expected.

---

## 20. Entropy Bonus

Entropy encourages exploration.

The entropy bonus discourages the policy from becoming too certain too early.

This is important because the Rubik’s Cube action space can easily produce local loops or repeated bad patterns.

Entropy term:

```text
entropy_bonus = mean(policy_entropy)
```

The PPO total loss subtracts entropy because higher entropy is encouraged:

```text
total_loss = actor_loss + vf_coef × critic_loss - ent_coef × entropy
```

Recommended starting value:

```python
ent_coef = 0.01
```

If the agent remains too random after learning starts, reduce it:

```python
ent_coef = 0.001
```

or:

```python
ent_coef = 0.0
```

---

## 21. Total PPO Loss

The full PPO loss combines:

```text
actor loss
critic loss
entropy bonus
```

Recommended formula:

```text
total_loss = actor_loss + vf_coef × critic_loss - ent_coef × entropy
```

Where:

```text
actor_loss = policy loss from clipped PPO objective
critic_loss = value function loss
entropy = policy entropy
vf_coef = value loss coefficient
ent_coef = entropy coefficient
```

Recommended coefficients:

```python
vf_coef = 0.5
ent_coef = 0.01
```

---

## 22. Optimization Process

After collecting rollout data, PPO should run several optimization epochs over the same rollout.

Recommended:

```python
n_epochs = 10
```

Each epoch should divide the rollout into mini-batches.

Recommended mini-batch size:

```python
batch_size = 64
```

Training process:

```text
1. Collect rollout data.
2. Calculate returns.
3. Calculate advantages.
4. Normalize advantages.
5. Shuffle rollout samples.
6. Split samples into mini-batches.
7. For each mini-batch:
   - Recalculate log probabilities using current policy.
   - Recalculate values using current critic.
   - Calculate PPO ratio.
   - Calculate clipped actor loss.
   - Calculate critic loss.
   - Calculate entropy.
   - Calculate total loss.
   - Backpropagate.
   - Clip gradients.
   - Update network parameters.
8. Repeat for n_epochs.
9. Collect new rollout data with the updated policy.
```

Recommended advantage normalization:

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

---

## 23. Gradient Clipping

Gradient clipping should be used to stabilize training.

Recommended value:

```python
max_grad_norm = 0.5
```

Purpose:

```text
Prevent very large gradients from causing unstable network updates.
```

Example:

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=0.5
)
```

---

## 24. Learning Rate

Recommended starting learning rate:

```python
learning_rate = 3e-4
```

This is a common PPO starting value.

If training is unstable:

```text
Reduce to 1e-4.
```

If training is too slow but stable:

```text
Try 5e-4.
```

Recommended MVP:

```python
learning_rate = 3e-4
```

---

## 25. PPO Hyperparameter Configuration

Recommended MVP PPO config:

```python
ppo_config = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "n_epochs": 10,
    "n_steps": 2048,
    "batch_size": 64,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.03,
}
```

Environment-related config:

```python
env_config = {
    "max_episode_steps": 50,
    "scramble_depth_range": (1, 5),
    "move_penalty": -1.0,
    "solve_bonus": 100.0,
    "inverse_move_penalty": -5.0,
    "timeout_penalty": -10.0,
}
```

Network config:

```python
network_config = {
    "input_dim": 54,
    "hidden_layers": [256, 256],
    "activation": "ReLU",
    "actor_output_dim": 12,
    "critic_output_dim": 1,
}
```

---

## 26. Training Curriculum

The PPO agent should not start by training on all difficulties equally unless early tests show it can handle that.

Recommended curriculum:

```text
Phase 1: Train on depth 1 only.
Phase 2: Train on depths 1–2.
Phase 3: Train on depths 1–3.
Phase 4: Train on depths 1–4.
Phase 5: Train on depths 1–5.
```

The environment should sample depth uniformly within the active range.

Example:

```text
For depths 1–3:

1/3 chance of depth 1
1/3 chance of depth 2
1/3 chance of depth 3
```

This prevents deeper state files from dominating training simply because they contain more states.

---

## 27. Curriculum Promotion Criteria

The agent should move to the next curriculum phase only after reaching a target solve rate.

Suggested promotion thresholds:

```text
Depth 1 only:
Move on when solve rate ≥ 95%

Depths 1–2:
Move on when depth 1 solve rate ≥ 95%
and depth 2 solve rate ≥ 80%

Depths 1–3:
Move on when depth 1 solve rate ≥ 95%
and depth 2 solve rate ≥ 80%
and depth 3 solve rate ≥ 60%

Depths 1–5:
Final MVP training phase
```

If training is unstable, lower the thresholds temporarily, but do not skip depth-wise evaluation.

---

## 28. Training Loop Pseudocode

High-level PPO training loop:

```python
initialize environment
initialize actor_critic_model
initialize optimizer
initialize rollout_buffer

for update in range(num_updates):

    rollout_buffer.clear()

    for step in range(n_steps):

        state_tensor = preprocess(observation)

        action, log_prob, value = model.get_action_and_value(state_tensor)

        next_observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        rollout_buffer.add(
            observation=observation,
            action=action,
            reward=reward,
            done=done,
            log_prob=log_prob,
            value=value,
            info=info
        )

        observation = next_observation

        if done:
            observation, info = env.reset()

    returns, advantages = compute_gae_and_returns(
        rollout_buffer,
        gamma=0.99,
        gae_lambda=0.95
    )

    normalize advantages

    for epoch in range(n_epochs):

        shuffle rollout data

        for minibatch in rollout_buffer.iter_minibatches(batch_size=64):

            new_log_probs, entropy, new_values = model.evaluate_actions(
                minibatch.observations,
                minibatch.actions
            )

            ratio = exp(new_log_probs - minibatch.old_log_probs)

            unclipped_objective = ratio * minibatch.advantages

            clipped_objective = clip(
                ratio,
                1 - clip_range,
                1 + clip_range
            ) * minibatch.advantages

            actor_loss = -mean(min(unclipped_objective, clipped_objective))

            critic_loss = mean((minibatch.returns - new_values) ** 2)

            total_loss = (
                actor_loss
                + vf_coef * critic_loss
                - ent_coef * entropy.mean()
            )

            optimizer.zero_grad()
            total_loss.backward()
            clip gradients
            optimizer.step()

    evaluate agent periodically
    save model if performance improves
```

---

## 29. Stable-Baselines3 Implementation Option

For the MVP, the project may use Stable-Baselines3 PPO instead of implementing PPO manually.

This is recommended if the priority is to get a working training pipeline quickly.

Example setup:

```python
from stable_baselines3 import PPO

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=0.03,
    verbose=1,
)

model.learn(total_timesteps=500_000)
model.save("models/ppo_rubiks_depth_1")
```

Recommended approach:

```text
Use Stable-Baselines3 PPO for the first working version.
Implement custom PPO later if deeper learning documentation or full control is needed.
```

This reduces the chance of debugging PPO math and cube logic at the same time.

---

## 30. Custom Policy Architecture in Stable-Baselines3

If using Stable-Baselines3, the network architecture can be configured like this:

```python
policy_kwargs = {
    "net_arch": {
        "pi": [256, 256],
        "vf": [256, 256],
    }
}
```

Then:

```python
model = PPO(
    policy="MlpPolicy",
    env=env,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
)
```

This gives the actor and critic separate MLP branches.

---

## 31. Evaluation Strategy

The agent should be evaluated separately from training.

During evaluation:

```text
Use deterministic actions.
Disable exploration sampling.
Do not update the model.
Run fixed numbers of test episodes.
Report metrics by scramble depth.
```

Recommended evaluation episodes:

```text
100 episodes per depth for quick checks.
1000 episodes per depth for more reliable evaluation.
```

Evaluation depths:

```text
Depth 1
Depth 2
Depth 3
Depth 4
Depth 5
```

Evaluation function should track:

```text
solve rate
average solution length
median solution length
average reward
timeout rate
immediate inverse move rate
average extra moves above starting depth
```

---

## 32. Evaluation Pseudocode

```python
def evaluate_agent(model, env, depths, episodes_per_depth=100):

    results = {}

    for depth in depths:

        env.set_depth(depth)

        solved_count = 0
        total_reward = 0
        solution_lengths = []
        timeout_count = 0
        inverse_move_count = 0

        for episode in range(episodes_per_depth):

            obs, info = env.reset()
            done = False
            episode_reward = 0

            while not done:

                action, _ = model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated
                episode_reward += reward

                if info["immediate_inverse_move"]:
                    inverse_move_count += 1

            if info["is_solved"]:
                solved_count += 1
                solution_lengths.append(info["current_step"])
            else:
                timeout_count += 1

            total_reward += episode_reward

        results[depth] = {
            "solve_rate": solved_count / episodes_per_depth,
            "average_reward": total_reward / episodes_per_depth,
            "average_solution_length": mean(solution_lengths)
                if solution_lengths else None,
            "timeout_rate": timeout_count / episodes_per_depth,
            "inverse_move_count": inverse_move_count,
        }

    return results
```

---

## 33. Success Metrics

The main metric is:

```text
solve_rate
```

Defined as:

```text
solve_rate = solved_episodes / total_episodes
```

Track solve rate by depth:

```text
Depth 1 solve rate
Depth 2 solve rate
Depth 3 solve rate
Depth 4 solve rate
Depth 5 solve rate
```

Secondary metrics:

```text
average reward
average solution length
median solution length
timeout rate
immediate inverse move rate
policy entropy
value loss
policy loss
KL divergence
explained variance
```

Project-specific metric:

```text
extra_moves = agent_solution_length - start_depth
```

This is useful if your stored depth represents the true minimum number of moves from the solved state.

---

## 34. MVP Success Targets

Recommended first-stage targets:

```text
Depth 1: 95%+ solve rate
Depth 2: 80%+ solve rate
Depth 3: 60%+ solve rate
Depth 4: better than random baseline
Depth 5: better than random baseline
```

The MVP should not be judged by whether it solves arbitrary full Rubik’s Cube scrambles.

The MVP should prove:

```text
The environment works.
The reward function works.
PPO learns better-than-random behavior.
The agent improves as curriculum difficulty increases.
The agent can solve shallow scrambles efficiently.
```

---

## 35. Random Agent Baseline

A random agent should be evaluated before PPO training.

The random agent chooses one of the 12 actions uniformly.

Random baseline pseudocode:

```python
action = env.action_space.sample()
```

Track the same metrics:

```text
solve rate
average reward
average solution length
timeout rate
inverse move rate
```

The PPO agent should be compared against this baseline at each depth.

This is especially important because shallow depths may sometimes be solved by random chance.

---

## 36. Oracle Baseline

If the dataset includes an `inverse_solution` column, use it for evaluation only.

The oracle baseline gives one known solution path back to the solved state.

Use it to calculate:

```text
oracle solution length
agent solution length
extra moves used
agent solved successfully
```

The oracle solution must not be included in the PPO observation.

Allowed:

```text
Use inverse_solution in info for logging and evaluation.
```

Not allowed:

```text
Feed inverse_solution into the policy network.
```

---

## 37. Logging Requirements

The training script should log both PPO metrics and cube-specific metrics.

PPO metrics:

```text
policy loss
value loss
entropy
approximate KL divergence
clip fraction
explained variance
learning rate
total timesteps
```

Environment metrics:

```text
episode reward
episode length
solve status
start depth
timeout status
immediate inverse move count
solution length
```

Evaluation metrics:

```text
solve rate by depth
average solution length by depth
timeout rate by depth
average reward by depth
extra moves by depth
```

Recommended logging tools:

```text
console logs for MVP
CSV logs for experiment tracking
TensorBoard for PPO training curves
Weights & Biases optional for later experiments
```

---

## 38. Model Saving

The training system should save models periodically.

Recommended save conditions:

```text
Save every fixed number of timesteps.
Save whenever evaluation solve rate improves.
Save final model after training completes.
```

Suggested file structure:

```text
models/
  ppo/
    depth_1/
      best_model.zip
      final_model.zip

    depth_1_2/
      best_model.zip
      final_model.zip

    depth_1_5/
      best_model.zip
      final_model.zip
```

Recommended checkpoint metadata:

```text
training timestep
active curriculum phase
evaluation solve rate
average solution length
environment config
PPO config
date/time
random seed
```

---

## 39. Reproducibility Requirements

To make results reproducible, the training setup should control random seeds.

Set seeds for:

```text
Python random
NumPy
PyTorch
Gym/Gymnasium environment
action space
observation sampling
```

Example:

```python
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

obs, info = env.reset(seed=seed)
env.action_space.seed(seed)
```

Recommended:

```text
Run at least 3 different seeds before trusting final performance claims.
```

A single lucky seed can give misleading results.

---

## 40. Recommended Project File Structure

Suggested code organization:

```text
src/
  cube/
    moves.py
    state.py
    notation.py
    validators.py

  envs/
    rubiks_cube_env.py

  agents/
    ppo_agent.py
    networks.py
    rollout_buffer.py

  training/
    train_ppo.py
    evaluate_ppo.py
    curriculum.py
    callbacks.py

  configs/
    env_config.yaml
    ppo_config.yaml

data/
  cube_states/
    depth_1.parquet
    depth_2.parquet
    depth_3.parquet
    depth_4.parquet
    depth_5.parquet

models/
  ppo/

logs/
  tensorboard/
  csv/
```

If using Stable-Baselines3, the custom files can be simpler:

```text
src/
  envs/
    rubiks_cube_env.py

  training/
    train_ppo_sb3.py
    evaluate_ppo.py
    curriculum.py
    callbacks.py

  configs/
    env_config.yaml
    ppo_config.yaml
```

---

## 41. Recommended YAML Config

Example PPO config:

```yaml
ppo:
  policy: MlpPolicy
  learning_rate: 0.0003
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  n_epochs: 10
  n_steps: 2048
  batch_size: 64
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  target_kl: 0.03

network:
  input_dim: 54
  hidden_layers: [256, 256]
  activation: ReLU
  actor_output_dim: 12
  critic_output_dim: 1

training:
  total_timesteps: 500000
  eval_frequency: 10000
  eval_episodes_per_depth: 100
  save_best_model: true
  seed: 42

curriculum:
  enabled: true
  phases:
    - name: depth_1
      depth_range: [1, 1]
      promotion_solve_rate: 0.95

    - name: depths_1_2
      depth_range: [1, 2]
      promotion_solve_rate: 0.80

    - name: depths_1_3
      depth_range: [1, 3]
      promotion_solve_rate: 0.60

    - name: depths_1_5
      depth_range: [1, 5]
      promotion_solve_rate: null
```

---

## 42. Key Implementation Warnings

### 42.1 Do Not Put PPO Logic Inside the Environment

The environment should not calculate:

```text
advantage
returns-to-go
policy loss
critic loss
entropy bonus
PPO clipping ratio
gradient updates
```

These belong to the PPO agent or training script.

The environment only calculates:

```text
immediate reward
next state
done/truncated status
info metadata
```

---

### 42.2 Do Not Put Environment Reward Inside the Actor

The actor should not manually calculate:

```text
-1 for move
+100 for solved
-5 for inverse
-10 for timeout
```

That belongs in the environment.

The actor only receives the resulting reward through the rollout data.

---

### 42.3 Do Not Expose the Inverse Solution to the Policy

The policy input should only be the cube state.

Do not give the policy:

```text
inverse_solution
scramble_sequence
state depth
known next best move
oracle solution
```

Those are allowed in `info`, logging, and evaluation only.

---

### 42.4 Watch for Shallow-Depth Overfitting

The agent may memorize common depth-1 or depth-2 patterns.

To reduce this risk:

```text
use all unique states
evaluate on held-out states if possible
track performance by depth
increase depth gradually
compare against random and oracle baselines
```

---

### 42.5 Monitor Inverse Move Rate

Because the environment penalizes immediate inverse moves by `-5`, the agent should gradually reduce direct undo behavior.

Track:

```text
inverse_move_rate = inverse_moves / total_moves
```

If inverse move rate remains high:

```text
training may be unstable
policy may not understand cube structure
entropy may be too high
reward signal may be too sparse
```

If inverse move rate becomes almost zero but solve rate also stays low:

```text
the inverse penalty may be discouraging useful exploration too strongly
```

Current penalty:

```text
-5
```

This is reasonable for MVP.

---

## 43. Final PPO Agent Behavior Summary

At training time:

```text
The agent observes the current cube state.
The actor outputs probabilities over 12 moves.
The agent samples a move.
The environment applies the move and returns reward.
The transition is stored in the rollout buffer.
After enough rollout steps, PPO calculates returns and advantages.
PPO updates the actor and critic using the clipped PPO objective.
The process repeats over many training updates.
```

At evaluation time:

```text
The agent observes the current cube state.
The actor selects the highest-probability move.
The environment applies the move.
The episode continues until solved or timed out.
Metrics are recorded by scramble depth.
```

The PPO agent’s main objective is:

```text
Maximize solve rate while minimizing solution length and timeout rate.
```

For this Rubik’s Cube project, the PPO agent should be considered successful when it consistently solves shallow pre-generated states better than a random baseline and shows depth-by-depth improvement through curriculum training.
