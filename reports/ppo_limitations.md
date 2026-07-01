# PPO Depth-1 Limitations and Mitigations

## Historical Observation

The custom PPO agent learned better-than-random depth-1 behavior, but it did not learn a reliable exact inverse-move policy.

The strongest depth-1 PPO-only run so far used:

```text
total_timesteps: 1,000,000
learning_rate: 0.0001
ent_coef: 0.0001
target_kl: 0.01
episodes_per_eval: 500
```

Observed results:

```text
best solve rate: 55.4%
final solve rate: 49.2%
random baseline solve rate: 5.0%
```

These historical results confirmed that the PPO loop, environment reward, and
dataset sampling produced a learnable signal, but the policy fell short of the
depth-1 target.

## Limitation

For depth-1 states, the optimal behavior is to choose the exact inverse of the scramble move. The trained policy often learns a coarser shortcut: identify or prefer a face and repeatedly turn that face. This can solve some one-move scrambles in multiple turns, but it does not produce the correct first move consistently.

That behavior explains why solve rate can improve while exact first-move accuracy remains low.

## Action-Subset Collapse

The `depth_1_onehot/v002` run shows a stronger version of the same limitation. Its final deterministic evaluation reached an `83.4%` solve rate, but several legal moves were never selected at all:

```text
unused actions: B', D, F', L, L', R, U'
used actions: B, D', F, R', U
```

This is not caused by missing legal actions in the environment. The action space still exposes all 12 quarter-turn moves, and the random baseline uses all 12. It is also not explained by the depth-1 dataset, which contains one example for each single-move scramble and one corresponding first solution move for each action.

Greedy `argmax` evaluation makes the collapse look absolute because an action
can retain nonzero probability without ever becoming the top-scoring choice.
However, the low training entropy in later runs confirms that the underlying
stochastic policy also collapses. The learned policy can additionally exploit
cube equivalences, such as using `U U U` instead of selecting `U'` directly.
This can solve some states while still avoiding the correct one-move inverse
action.

The `inverse_move_rate` metric needs careful interpretation here. A value of `0.0` does not necessarily mean the policy learned a better solver. It may mean the policy learned to avoid immediate undo pairs, while also suppressing useful inverse-labeled actions. The current inverse penalty only applies when an action immediately reverses the previous action; it does not penalize selecting an inverse-labeled action on the first move. Even so, the penalty may indirectly reinforce habits that avoid direct inverse choices after the policy begins favoring repeated same-direction turns.

## Resolved Input Issue

The original PPO input encoded sticker colors as normalized numeric IDs:

```text
Y=0.0, O=0.2, G=0.4, W=0.6, R=0.8, B=1.0
```

This imposes fake ordering and distance relationships between colors. Rubik's Cube sticker colors are categorical labels, not scalar quantities.

For example, there is no meaningful reason that `G` should be numerically closer to `W` than to `B`.

## Implemented: One-Hot Observations

New PPO runs use one-hot sticker observations by default:

```text
54 stickers * 6 colors = 324 input values
```

This removes artificial color magnitude and gives the actor-critic network categorical sticker information directly.

## Other Implemented Mitigations

### Complete Small-Dataset Coverage

Small training datasets now cycle exhaustively, with an independent cycle per
depth. Deterministic evaluation treats the requested episode count as a maximum
and evaluates each state exactly once when the dataset is small. A request for
1,000 depth-1 episodes therefore evaluates the 12 unique states once instead of
repeating identical decisions.

### Policy-Behavior Metrics

Evaluation now records action counts and action distributions. This makes
action-subset collapse visible instead of reporting only aggregate solve rate.

### Scaled Rewards

The reward schedule was divided by 100 while preserving relative incentives:

```text
move penalty:              -0.01
solve bonus:               +1.00
immediate inverse penalty: -0.05
timeout penalty:           -0.10
```

Advantage normalization keeps the actor signal comparable while smaller returns
reduce critic loss and critic-gradient magnitude.

### Separate Actor and Critic Networks

The original network shared an MLP trunk because feature reuse reduces model
size and can improve sample efficiency when actor and critic gradients are
compatible. In this project, value-regression gradients were much larger than
policy gradients and repeatedly changed the features used by the actor.

Actor and critic now use independent MLP towers and their gradients are clipped
separately. Legacy shared-trunk checkpoints are migrated by copying the old
shared layers into both towers, preserving their policy and value predictions.

## Remaining Limitations

### Three-Turn Inverse Shortcut and Missing Metrics

A quarter turn repeated three times is equivalent to its inverse. The policy
can solve a depth-1 state with `U U U` instead of selecting `U'`. One direction
per face is therefore sufficient to solve every depth-1 state within three
moves, so solve rate does not require all 12 actions or the correct first move.

Evaluation still needs:

```text
first_move_accuracy
per-state predicted move table
per-face accuracy
move-direction accuracy
```

These metrics are not yet used for checkpoint selection.

### Entropy Collapse and Overtraining

Later training frequently reduces policy entropy until only a small action
subset has meaningful probability. Training can continue beyond the best
checkpoint and degrade solve rate. Early stopping, entropy scheduling, and
automatic restoration of the best checkpoint are not yet implemented.

### KL Stopping Granularity

The target-KL check currently runs after an optimization epoch. A minibatch can
exceed the target substantially before the update stops. Per-minibatch KL
stopping remains to be implemented.

### No Supervised Warm Start

The dataset includes `first_solution_move`, but PPO does not initialize its
actor from those labels. Supervised actor pretraining followed by PPO
fine-tuning remains a planned experiment.
