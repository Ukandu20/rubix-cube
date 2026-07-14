# Supervised Actor Warm Start Specification for PPO Curriculum Training

## 1. Purpose

Add an optional supervised warm-start phase that pretrains the PPO actor using labeled Rubik’s Cube states before reinforcement-learning training begins.

Each supervised row contains:

* the encoded cube state;
* the exact scramble or solution depth;
* one or more valid first solution moves.

The warm-start feature is intended to reduce the sparse-reward bottleneck encountered after depth 5 by giving the actor useful action preferences before PPO must discover successful trajectories through exploration.

Scratch initialization must remain the default so controlled A/B experiments can compare:

* PPO trained from random initialization;
* PPO initialized from mastered depths;
* PPO initialized from the current frontier;
* PPO initialized from the full configured curriculum.

---

## 2. Scope

The supervised warm start applies only before PPO training begins.

It must:

* initialize the actor using supervised state-action examples;
* preserve the configured PPO timestep budget;
* retain complete provenance in all resulting checkpoints;
* support controlled experiments using different supervised depth ranges;
* provide supervised and rollout-based evaluation before PPO starts.

It must not:

* replace PPO training;
* reduce the PPO timestep budget;
* silently include depths outside the selected supervision range;
* overwrite scratch initialization as the default;
* use held-out test states during supervised or PPO training.

---

## 3. Command-Line Interface

Add the following CLI options:

```text
--supervised-warm-start
--pretrain-depth-mode
--pretrain-min-depth
--pretrain-max-depth
--pretrain-epochs
--pretrain-batch-size
--pretrain-learning-rate
--pretrain-sample-per-depth
--pretrain-validation-fraction
--pretrain-test-fraction
--pretrain-depth-sampling
--pretrain-label-smoothing
--pretrain-early-stopping-patience
--pretrain-gradient-clip-norm
--pretrain-update-shared-encoder
```

### 3.1 `--supervised-warm-start`

Enables actor pretraining.

Default:

```text
false
```

When disabled:

* no supervised dataset is loaded;
* no pretraining metrics are created;
* no warm-start checkpoint is saved;
* the PPO actor and critic use their normal initialization.

---

### 3.2 `--pretrain-depth-mode`

Controls which depths are used for supervised training.

Supported values:

```text
mastered
frontier
full-curriculum
custom
```

Definitions:

#### `mastered`

Use only previously mastered depths.

Example:

```text
Current curriculum frontier: 6
Supervised depths: 1–5
```

#### `frontier`

Use mastered depths plus the current target depth.

Example:

```text
Current curriculum frontier: 6
Supervised depths: 1–6
```

#### `full-curriculum`

Use every depth in the configured curriculum.

Example:

```text
Configured curriculum: 1–10
Supervised depths: 1–10
```

#### `custom`

Use the explicit range provided through:

```text
--pretrain-min-depth
--pretrain-max-depth
```

---

### 3.3 Default Hyperparameters

Use the following initial defaults:

```text
epochs: 10
batch size: 256
learning rate: 1e-3
sample cap per depth: 100,000
validation fraction: 0.10
test fraction: 0.10
depth sampling: balanced
label smoothing: 0.01
early-stopping patience: 2
gradient clipping norm: 1.0
```

These must remain configurable.

---

## 4. Preconditions

Supervised warm start requires:

* one-hot observations;
* labeled state-action rows;
* an explicit and validated action ordering;
* a valid depth field;
* deterministic dataset sampling;
* distinct training, validation, and test partitions.

Reject warm start when:

* observations are normalized continuous vectors without an approved supervised preprocessing path;
* action labels do not match the current environment action ordering;
* the requested depth range has no labeled rows;
* the training and evaluation splits cannot be separated safely;
* the loaded checkpoint architecture is incompatible with the current PPO architecture.

---

## 5. Observation and Action Compatibility

Store the following schema information in the warm-start checkpoint:

```text
observation_encoding
observation_shape
action_count
action_order
state_encoder_version
model_architecture_version
```

Example action ordering:

```text
[U, U', D, D', L, L', R, R', F, F', B, B']
```

Checkpoint loading must fail when the current environment uses a different action order.

A checkpoint loading successfully at the PyTorch level is not sufficient. Schema compatibility must also be validated.

---

## 6. Dataset Loading

For each selected depth:

1. load the corresponding labeled dataset;
2. validate the required columns;
3. remove invalid rows;
4. deduplicate by encoded cube state;
5. group related rows when they belong to the same source trajectory;
6. preserve all known valid solution moves;
7. deterministically cap the stored rows;
8. create depth-specific train, validation, and test splits.

Required columns:

```text
encoded_state
depth
first_solution_move
```

Preferred additional columns:

```text
valid_first_solution_moves
trajectory_id
solution_id
scramble_sequence
optimal_solution_length
source_file
```

---

## 7. Deterministic Sampling

For a depth with at most the configured sample cap:

```text
use every valid unique row
```

For a depth larger than the sample cap:

```text
sample exactly pretrain_sample_per_depth rows
```

Sampling must:

* use the PPO seed;
* be repeatable;
* occur after deduplication;
* record the selected row identifiers;
* save a split manifest.

The manifest should allow the exact supervised dataset to be reconstructed.

---

## 8. Train, Validation, and Test Splits

Do not use one global random row split.

Create the split independently within each exact depth.

Recommended default:

```text
80% training
10% validation
10% final test
```

The split must be grouped by:

* encoded cube state;
* trajectory ID when available;
* solution ID when available.

Rows from the same trajectory must not appear across multiple partitions.

The final test partition must not be used for:

* supervised optimization;
* early stopping;
* hyperparameter selection;
* PPO environment resets;
* curriculum training;
* model selection.

The validation partition may be used for:

* early stopping;
* selecting the best supervised epoch;
* monitoring overfitting;
* comparing pretraining hyperparameters.

---

## 9. Depth Sampling Strategies

The number of stored rows and the number of gradient updates contributed by a depth must be treated as separate concerns.

Support the following depth-sampling strategies.

### 9.1 `balanced`

Select each configured depth with equal probability:

[
P(d)=\frac{1}{D}
]

Then sample an example from that depth.

This should be the default for controlled experiments.

---

### 9.2 `natural`

Sample examples according to the number of available training rows.

This reflects the natural dataset distribution but may allow deeper datasets to dominate.

---

### 9.3 `frontier-weighted`

Assign greater probability to the current curriculum frontier.

Example for frontier depth 6:

```text
Depths 1–3: 15% combined
Depths 4–5: 30% combined
Depth 6: 45%
Depth 7 preview: 10%
```

Exact weights must be configurable or derived from a documented weighting rule.

---

## 10. Supervised Labels

### 10.1 Single Demonstrated Move

When only one move is available, treat it as:

```text
demonstrated_first_move
```

Do not assume it is necessarily the only correct move.

The supervised objective may use cross-entropy with optional label smoothing.

[
L_{\text{BC}}
=============

-\log \pi_\theta(a^* \mid s)
]

---

### 10.2 Multiple Valid First Moves

When multiple optimal first moves are known, use a target probability distribution over all valid moves.

For a valid-action set (A^*(s)):

[
y(a)=
\begin{cases}
\frac{1}{|A^*(s)|}, & a \in A^*(s) \
0, & \text{otherwise}
\end{cases}
]

The loss becomes:

[
L_{\text{multi}}
================

-\sum_a y(a)\log \pi_\theta(a\mid s)
]

This prevents valid alternative solution moves from being treated as incorrect.

---

### 10.3 Label Smoothing

Support configurable label smoothing:

[
y_{\text{smooth}}
=================

(1-\epsilon)y
+
\frac{\epsilon}{|\mathcal A|}
]

Default:

```text
epsilon = 0.01
```

Label smoothing may be disabled with:

```text
--pretrain-label-smoothing 0
```

---

## 11. Model Architecture Handling

Before implementing actor-only pretraining, determine whether the PPO network uses:

* fully separate actor and critic networks;
* a shared feature extractor;
* partially shared hidden layers.

The specification must record which parameter groups are updated.

---

### 11.1 Separate Actor and Critic Networks

When the networks are independent:

* update actor parameters;
* freeze critic parameters;
* confirm critic parameters remain bit-for-bit unchanged.

---

### 11.2 Shared Encoder

When the actor and critic share an encoder, support two modes.

#### Actor-head-only warm start

Update only actor-exclusive layers.

Advantages:

* critic inputs and outputs remain unchanged;
* clean actor-only experiment.

Disadvantages:

* the shared representation does not learn supervised cube features.

#### Shared-encoder warm start

Update:

* the shared encoder;
* the actor head.

Freeze:

* critic-exclusive parameters.

This may improve actor representations, but the critic’s outputs can change even if its head weights remain unchanged.

Checkpoint metadata must clearly record:

```text
shared_encoder_updated: true or false
critic_head_updated: false
```

---

## 12. Optimizer Handling

Use a separate optimizer for supervised pretraining.

After pretraining:

1. discard the supervised optimizer;
2. create a fresh PPO optimizer;
3. initialize it using the pretrained model parameters;
4. do not carry supervised optimizer momentum into PPO.

This prevents optimizer-state contamination between supervised learning and reinforcement learning.

---

## 13. Training Procedure

For every supervised epoch:

1. construct depth-aware minibatches;
2. perform a forward pass through the actor;
3. calculate the supervised loss;
4. backpropagate only through permitted parameter groups;
5. apply gradient clipping;
6. update supervised parameters;
7. evaluate on the validation partition;
8. record metrics by exact depth;
9. apply early stopping when required.

Default maximum epochs:

```text
10
```

Default early-stopping patience:

```text
2
```

The best checkpoint should be selected using validation macro loss or validation macro accuracy.

Restore the best supervised checkpoint before PPO begins.

---

## 14. Saved Checkpoints

Save:

```text
supervised_warm_start_last.pt
supervised_warm_start_best.pt
```

PPO must initialize from:

```text
supervised_warm_start_best.pt
```

unless explicitly configured otherwise.

Each checkpoint must include:

```text
model_state_dict
architecture_version
observation_schema
action_schema
selected_depth_range
depth_mode
dataset_counts
train_counts_by_depth
validation_counts_by_depth
test_counts_by_depth
split_manifest_reference
sampling_strategy
seed
hyperparameters
updated_parameter_groups
best_epoch
training_history
validation_metrics
dataset_identifiers
dataset_hashes
runtime
software_versions
```

---

## 15. Supervised Evaluation Metrics

Record the following for training and validation.

### 15.1 Per-Depth Metrics

For every exact depth:

```text
loss
top-1 accuracy
top-3 accuracy
macro action accuracy
mean action confidence
policy entropy
row count
```

---

### 15.2 Aggregate Metrics

Report both micro and macro metrics.

Micro accuracy:

[
\text{MicroAccuracy}
====================

\frac{\text{total correct predictions}}
{\text{total predictions}}
]

Macro accuracy:

[
\text{MacroAccuracy}
====================

\frac{1}{D}
\sum_{d=1}^{D}
\text{Accuracy}_d
]

Macro metrics prevent large depth datasets from hiding poor performance at smaller depths.

---

## 16. Pre-PPO Greedy Rollout Evaluation

After supervised pretraining and before PPO training, evaluate the actor using complete greedy rollouts.

Run the evaluation separately on:

* supervised training states;
* validation states;
* final held-out test states.

Report by exact depth:

```text
first-move top-1 accuracy
first-move top-3 accuracy
greedy solve rate
mean moves to solve
median moves to solve
timeout rate
inverse-move frequency
policy entropy
average action confidence
mastered-region entry rate
```

This distinguishes classification performance from actual solving performance.

A high first-move accuracy must not be interpreted as successful cube-solving behaviour without rollout evaluation.

---

## 17. PPO Transition Diagnostics

Record metrics at the following points:

```text
before supervised pretraining
after supervised pretraining
after the first PPO rollout
after the first PPO update
after each curriculum evaluation
```

Monitor:

```text
solve rate
policy entropy
KL divergence
clip fraction
actor loss
critic loss
explained variance
timeout rate
first-move accuracy
greedy solve rate
```

This is necessary to detect whether the first PPO updates destroy the supervised initialization.

---

## 18. Curriculum Integration

Supervised warm start occurs once before PPO begins unless a separate future specification introduces repeated frontier pretraining.

The PPO curriculum must continue to use its normal advancement rules.

The warm start may expose the actor to:

* mastered depths only;
* the current frontier;
* the full future curriculum.

The selected mode must be recorded and clearly distinguished in experiment names.

Example:

```text
ppo_scratch_depth_1_10_seed_42
ppo_warm_mastered_1_5_depth_1_10_seed_42
ppo_warm_frontier_1_6_depth_1_10_seed_42
ppo_warm_full_1_10_depth_1_10_seed_42
```

---

## 19. Required Experiment Matrix

At minimum, run the following controlled experiments using the same:

* PPO seed;
* PPO timestep budget;
* environment configuration;
* curriculum thresholds;
* reward function;
* evaluation states;
* network architecture.

### Experiment A: Scratch

```text
Supervised depths: none
Initialization: random
```

### Experiment B: Mastered-Depth Warm Start

```text
Supervised depths: 1–5
PPO curriculum: 1–10
```

Purpose:

Determine whether shallow supervision improves general representation and retention.

### Experiment C: Frontier Warm Start

```text
Supervised depths: 1–6
PPO curriculum: 1–10
```

Purpose:

Determine whether direct supervision at depth 6 overcomes the immediate sparse-reward bottleneck.

### Experiment D: Full-Curriculum Warm Start

```text
Supervised depths: 1–10
PPO curriculum: 1–10
```

Purpose:

Determine whether advance exposure to all future depths accelerates the entire curriculum.

---

## 20. Comparison Metrics

Compare experiments using:

```text
pretraining runtime
total experiment runtime
first-move accuracy by depth
pre-PPO greedy solve rate
PPO timesteps to reach depth 6
PPO timesteps to reach each later depth
curriculum advancement timing
solve rate by exact depth
timeout rate by exact depth
mean episode length
final greedy solve rate
final stochastic solve rate
policy entropy
critic explained variance
catastrophic forgetting on mastered depths
```

The main sparse-reward metric should be:

```text
PPO timesteps required to reach the first stable success threshold at depth 6
```

---

## 21. Experiment Fairness

Supervised pretraining adds computation.

Therefore, report results in two ways:

### Equal PPO timesteps

All experiments receive the same PPO timestep budget.

This isolates whether the warm start improves PPO sample efficiency.

### Equal total computation

Include supervised runtime and supervised gradient updates in the comparison.

This evaluates whether warm start provides an overall computational advantage.

Do not claim the warm-start method is more computationally efficient using only equal PPO timestep results.

---

## 22. Metadata and Provenance

All PPO checkpoints created after warm start must retain:

```text
initialization_mode
warm_start_checkpoint_path
warm_start_checkpoint_hash
supervised_depth_mode
supervised_depth_range
supervised_seed
supervised_hyperparameters
supervised_dataset_counts
supervised_split_manifest
supervised_best_epoch
supervised_validation_metrics
supervised_rollout_metrics
updated_parameter_groups
```

This information must also appear in:

* experiment configuration;
* metrics output;
* curriculum logs;
* final evaluation report;
* experiment tracker.

---

## 23. Test Plan

### 23.1 Default Scratch Behaviour

Confirm that without `--supervised-warm-start`:

* no labeled dataset is loaded;
* no supervised optimizer is created;
* no supervised checkpoint is saved;
* no supervised fields are incorrectly recorded;
* actor and critic use standard initialization.

---

### 23.2 Depth Selection

Confirm:

* `mastered` selects only mastered depths;
* `frontier` includes the current frontier;
* `full-curriculum` uses the full configured range;
* `custom` respects the explicit minimum and maximum;
* no unselected depth contributes rows.

---

### 23.3 Dataset Sampling

Confirm:

* all rows are used below the cap;
* exactly the configured number is sampled above the cap;
* sampling is deterministic for a fixed seed;
* different seeds produce different valid samples;
* sampling occurs after deduplication.

---

### 23.4 Split Integrity

Confirm:

* train, validation, and test states do not overlap;
* related trajectory rows remain in one partition;
* every configured depth contributes to each split when enough rows exist;
* the final test set is never used during PPO training.

---

### 23.5 Depth-Aware Sampling

Confirm:

* balanced sampling gives equal expected depth probability;
* natural sampling follows dataset frequency;
* frontier-weighted sampling follows configured weights;
* minibatches contain valid examples from selected depths.

---

### 23.6 Parameter Updates

Confirm:

* permitted actor parameters change;
* frozen parameters remain unchanged;
* critic head parameters remain unchanged when required;
* shared-encoder changes match the selected configuration;
* updated parameter names are recorded.

---

### 23.7 Checkpoint Compatibility

Confirm:

* best and last checkpoints load successfully;
* architecture mismatches are rejected;
* observation-shape mismatches are rejected;
* action-order mismatches are rejected;
* later PPO checkpoints retain warm-start provenance.

---

### 23.8 Training Behaviour

Confirm:

* supervised loss decreases on a small fixture dataset;
* validation metrics are recorded by depth;
* macro and micro metrics are calculated correctly;
* early stopping restores the best epoch;
* gradient clipping is applied;
* label smoothing behaves as configured.

---

### 23.9 Rollout Evaluation

Confirm:

* pre-PPO greedy rollout evaluation runs;
* solve rate is reported separately from first-move accuracy;
* metrics are grouped by exact depth;
* final test states remain held out.

---

### 23.10 Regression Tests

Run:

```text
PPO unit tests
curriculum tests
supervised-policy tests
dataset-splitting tests
checkpoint tests
CLI tests
experiment-tracker tests
environment compatibility tests
```

---

## 24. Initial Recommended Configuration

For the first sparse-reward experiment:

```text
PPO curriculum: depths 1–10
Supervised depth mode: frontier
Supervised depths: 1–6
Depth sampling: balanced
Sample cap per depth: 100,000
Epochs: 10
Batch size: 256
Learning rate: 1e-3
Validation fraction: 0.10
Test fraction: 0.10
Label smoothing: 0.01
Early-stopping patience: 2
Gradient clipping norm: 1.0
Restore best checkpoint: true
```

Run this against:

```text
scratch initialization
mastered-depth warm start using depths 1–5
full-curriculum warm start using depths 1–10
```

---

## 25. Success Criteria

The supervised warm start should be considered useful only when it demonstrates one or more of the following without materially damaging previously mastered depths:

* higher depth-6 solve rate before PPO;
* fewer PPO timesteps required to pass the depth-6 advancement threshold;
* fewer timeouts at depths 6 and above;
* improved final greedy solve rate;
* reduced variance across random seeds;
* stable retention of depths 1–5;
* improved curriculum progression after depth 5.

Supervised classification accuracy alone is not sufficient evidence of success.

---

## 26. Assumptions

* Depth labels represent true or reliably verified solution depths.
* Each encoded state is deterministic and hashable.
* Labeled moves use the same action ordering as the environment.
* Multiple valid solution moves may exist at higher depths.
* The 100,000-row cap applies independently to each depth.
* The cap controls stored examples, while depth sampling controls gradient influence.
* Supervised pretraining adds computation and does not reduce PPO timesteps.
* Actor-only pretraining is the initial implementation.
* Critic pretraining may be evaluated later as a separate ablation.
* Existing versioning and output-directory options distinguish all experiment variants.

---

## 27. Final Design Decision

The recommended initial implementation is:

[
\boxed{
\text{Opt-in actor warm start with stratified splits, balanced depth sampling, held-out rollout evaluation, and full checkpoint provenance.}
}
]

Low-depth examples should remain available because they teach reliable completion behaviour.

The current frontier should receive explicit attention because it addresses the sparse-reward bottleneck.

Future curriculum depths should be included only in the full-curriculum experiment so their effect can be separated from general warm-start benefits.
