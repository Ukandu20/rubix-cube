# Custom PPO vs Stable-Baselines3 PPO

This was a 512-timestep, seed-42 validation run on the depth-1–3 curriculum.
It validates trainer integration and metric parity; it is not a learning-quality
benchmark and must not be used to choose the default trainer.

Configuration parity: **passed**

| Metric | Custom PPO | SB3 PPO |
|---|---:|---:|
| actual_timesteps | 512 | 512 |
| elapsed_seconds | 10.217520500067621 | 3.7048086000140756 |
| timesteps_per_second | 50.11000467252417 | 138.19877226533504 |
| curriculum_depth | 1 | 1 |
| solve_rate | 0.3333333333333333 | 0.16666666666666666 |
| timeout_rate | 0.6666666666666666 | 0.8333333333333334 |
| average_solution_length | 1 | 2 |
| inverse_move_rate | 0.5 | 0.0 |
| action_entropy | 0.9057885085687165 | 0.13269142083987176 |

Outcome deltas are comparable under the declared controls.

This report validates the comparison pipeline. Learning conclusions require a
sufficiently large, predeclared training budget and multiple seeds.

## Recommendation

Retain both trainers and keep the custom PPO trainer as the experimental
baseline. SB3's higher throughput in this validation run is promising, but the
run is too short and its deterministic action distribution collapsed more
strongly. Do not designate SB3 as the default until a multi-seed, production-
budget comparison shows that the throughput advantage translates into equal or
better curriculum progression and solve performance.
