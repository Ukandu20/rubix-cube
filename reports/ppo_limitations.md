# PPO Depth-1 Limitation Notes

## Current Observation

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

This confirms the PPO loop, environment reward, and dataset sampling are producing a learnable signal, but the policy still falls short of the depth-1 target.

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

The likely cause is policy collapse under deterministic evaluation. Evaluation uses the greedy `argmax` action, so an action can have nonzero probability during training but still appear zero times in `action_counts` if it is never the top-scoring action for any evaluated state. The learned policy can also exploit cube equivalences, such as using `U U U` instead of selecting `U'` directly. This can solve some states while still avoiding the correct one-move inverse action.

The `inverse_move_rate` metric needs careful interpretation here. A value of `0.0` does not necessarily mean the policy learned a better solver. It may mean the policy learned to avoid immediate undo pairs, while also suppressing useful inverse-labeled actions. The current inverse penalty only applies when an action immediately reverses the previous action; it does not penalize selecting an inverse-labeled action on the first move. Even so, the penalty may indirectly reinforce habits that avoid direct inverse choices after the policy begins favoring repeated same-direction turns.

## Likely Input Issue

The original PPO input encoded sticker colors as normalized numeric IDs:

```text
Y=0.0, O=0.2, G=0.4, W=0.6, R=0.8, B=1.0
```

This imposes fake ordering and distance relationships between colors. Rubik's Cube sticker colors are categorical labels, not scalar quantities.

For example, there is no meaningful reason that `G` should be numerically closer to `W` than to `B`.

## Mitigation

New PPO runs should use one-hot sticker observations by default:

```text
54 stickers * 6 colors = 324 input values
```

This removes artificial color magnitude and gives the actor-critic network categorical sticker information directly.

## Follow-Up Metrics

Future PPO evaluation should add:

```text
first_move_accuracy
per-state predicted move table
per-face accuracy
move-direction accuracy
```

If one-hot PPO still plateaus, the next practical experiment is a supervised warm start from `first_solution_move`, followed by PPO fine-tuning.
