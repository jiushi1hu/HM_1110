# Five-reservoir HAPPO and MAPPO

Daily operation of the WDD, BHT, XLD, XJB and THR reservoir cascade, based on
PKU-MARL HARL. HAPPO uses five independent categorical actors; MAPPO uses one
identity-conditioned actor shared by all five reservoir agents. Both use one
centralized critic.
The environment resolves releases upstream to downstream with zero travel delay.
HAPPO learns one shared team advantage from `0.5 * sum(local rewards) +
0.5 * global generation reward`, counting the global term once. MAPPO retains
its individual 50/50 local/global advantages. The legacy team reward remains
available for evaluation, daily CSVs and historical plots.

The comparison entry point runs shared-actor MAPPO on the same
physical environment. HAPPO retains five independent actors and sequential
likelihood corrections; MAPPO retains its shared actor and individual advantages.
Their learning objectives now differ, so the figures compare these configured
methods rather than isolating an algorithm-only effect.

## Run HAPPO and MAPPO with three comparison figures

```bash
cd /Users/hujingxin/Documents/Code_python/HARL-main

PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B examples/train_cascade_compare.py \
  --episodes 300 --seeds 1 2 3 --gamma 0.995 --eval-interval 1 --diagnostics \
  --plot-python /Library/Frameworks/Python.framework/Versions/3.9/bin/python3
```

The new entry point defaults to `--algorithms happo mappo`. It trains fresh
models for each algorithm and seed, then opens exactly three comparison figures:
training return/generation, fixed-scenario evaluator return, and ecological-flow/
guaranteed-output violation rates. Every metric panel contains separate HAPPO
(solid red) and MAPPO (dashed blue) seed means and sample-standard-deviation bands. The seed
sets are never averaged across algorithms. The command above keeps results in
memory and creates no JSONL, CSV or checkpoint outputs; figures can be saved
with the Matplotlib window's Save button.

For an independent run, add `--algorithms happo` or `--algorithms mappo` to that
command. `examples/train_cascade.py` remains the original HAPPO-only entry point.
The comparison entry point accepts the same data, physical, training and
evaluation overrides as the HAPPO entry point. An identical override applies to
both algorithms. Running both doubles the configured training budget: the
300-episode budget is per algorithm and per seed.

To retain results and automatically save the three comparison PNGs, add a fresh
comparison stem, for example `--episode-log runs/comparison_seeds123_run1.jsonl`.
The stem itself is not a JSONL output. The launcher creates separate
`comparison_seeds123_run1_happo.jsonl` and
`comparison_seeds123_run1_mappo.jsonl` logs and their per-seed daily CSVs.
`--save-checkpoints` additionally saves algorithm-specific checkpoint directories.
All output paths for both algorithms are checked before training begins.
Existing files are not overwritten.

Saved comparison logs can be reopened without running policies again:

```bash
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 examples/plot_cascade_compare.py \
  --happo-log runs/comparison_seeds123_run1_happo.jsonl \
  --mappo-log runs/comparison_seeds123_run1_mappo.jsonl
```

The plotter checks matching data hashes, physics, service thresholds, training
and evaluation years, scenario inflows/initial conditions, seeds, hidden widths,
optimizer settings, discount and episode/update budgets. Actor sharing is the
explicit architectural difference and is recorded separately. Incompatible logs are
rejected. Failed or missing evaluations remain gaps. The two evaluation figures
use fixed training scenarios; final held-out results remain separate evaluations
in each algorithm's log. Existing team-return definitions and physical violation
counts are preserved.

## MAPPO implementation and comparison scope

The integration was based on the local PKU-MARL HARL reference at
`/Users/hujingxin/Documents/Code_python/kelong/HARL`, commit
`b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`, particularly
`harl/algorithms/actors/mappo.py`, `harl/runners/on_policy_ma_runner.py`,
the stochastic-policy/value networks, and `harl/configs/algos_cfgs/mappo.yaml`.
The reference repository is not a runtime dependency and was not modified.

The reference supports both shared and independent actors. HAPPO retains its
five independent actors. MAPPO now uses one actor network and one Adam optimizer,
accessed through five lightweight reservoir interfaces. Each interface appends a
five-component one-hot reservoir identity to its local observation, so the shared
network can distinguish reservoirs with different physical characteristics.
With daily limits enabled, physical observations and buffers remain 15-dimensional,
MAPPO's actor input is 20-dimensional, and HAPPO's actor input remains 15-dimensional.
Both centralized critics receive 38 system features. HAPPO has one scalar team
value output; MAPPO retains five mixed-return outputs and one legacy-return
output. The physical model, environment observations, local/global reward
components and service thresholds have not changed. Both algorithms use the
existing hidden widths `[64, 128, 256, 256]`, ReLU/LayerNorm layers, 100-command
categorical distributions, ValueNorm, GAE, forced
action masks, and uniform Actor row weighting. The shared YAML remains
`harl/configs/algos_cfgs/happo_cascade.yaml`. Its `share_param: false` continues to
describe HAPPO; the MAPPO runner and comparison launcher explicitly enable sharing
only for MAPPO and record the effective setting. Random seeds and hydrological
window schedules remain paired; actor parameter tensors and their initialization
are no longer expected to match across these different architectures.

MAPPO pools the five reservoirs' samples, preserving each row's own mixed-return
advantage, old action likelihood, identity and controllability.
It performs one optimizer step per joint minibatch, rather than training the same
network five times through the individual agent loop. With the default single
minibatch and five epochs, this is five shared-actor steps per update, over
54,750 reservoir-agent rows. MAPPO uses the PPO clipped surrogate plus entropy
regularization. It has
no HAPPO sequential likelihood correction and no command-imitation objective.
Its shared buffer factor slot is checked to be all ones and is not multiplied
into the loss. The critic update and reservoir collection/evaluation lifecycle
are reused. Both algorithms use PPO probability-ratio clipping and gradient-norm
clipping. With `--diagnostics`, KL divergence is measured after updates for
reporting; it does not stop epochs or reject optimizer steps.

This is a task-adapted, parameter-sharing MAPPO comparison. HARL's generic
EP default uses a common scalar reward/value. HAPPO now uses one team reward,
one scalar team value, and one advantage normalized once across the common
rollout rows and shared by all five actors. MAPPO retains individual mixed
rewards and six value outputs. This task also
retains this repository's full-row loss averaging, rather than introducing the
reference's active-count normalization. Actor policy and entropy losses have no
additional episode-age discount multiplier. Therefore,
the comparison now includes both the update-rule difference and policy parameter
sharing and the explicitly requested learning-objective difference. It does not
isolate the sequential-correction effect alone or reproduce every generic HARL
default. Each figure identifies the different learning objectives.

MAPPO checkpoints record their algorithm identity, sharing flag and reservoir IDs,
and save one actor state. Loading reconstructs one frozen shared network with five
identity-conditioned interfaces for inference/replay. HAPPO retains its five-state
checkpoint layout. Strict source/config fingerprints remain enabled:
older checkpoints created before the integration require their matching source
version for replay. No training-resume support has been added. The diagnostic
CSV's compatibility column named `happo_factor` is always one for MAPPO.
Per-reservoir diagnostics still have five rows; shared optimizer statistics are
marked `shared_actor_info_repeated` and must not be summed as five independent
optimizer lifecycles. Actor architecture metadata records the network count,
optimizer count, raw input dimension and identity-augmented input dimension.

## Current experiment

- Data cover 1965--2024: **55 training years and five evaluation years**.
- Evaluation years, in sequence order: **1998, 1999, 1970, 1997, 2006**.
- Training years are the full complement; inflow scaling uses training data only.
- Each episode contains **1,825 daily decisions**, with **six parallel workers**.
- Each reservoir has **100 discrete actions** by default.
- Fixed training-only scenarios are evaluated at **Round 0 and after each update**.
- The final trained policy is evaluated after all training seeds finish.
- Training uses a reproducible schedule that targets balanced year-day coverage, with distinct windows within each parallel batch.
- Service-threshold multipliers are configured in the environment YAML or CLI and recorded in each run's metadata.
- Actor losses use uniform temporal weighting, with forced decisions masked. GAE, critic returns and return accounting use `gamma=0.995`; physical energy and violation counts remain undiscounted. A soft cost applies to each guaranteed-output violation.

The standard budget is 300 completed episodes per seed: 50 HAPPO updates and
547,500 daily environment transitions. Each update collects six complete
episodes under one frozen policy, giving buffers of shape `[1825, 6, ...]`.
Episode counts must be divisible by the number of workers.

## HAPPO-only launch from VS Code

```bash
cd /Users/hujingxin/Documents/Code_python/HARL-main

PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B examples/train_cascade.py \
  --episodes 300 --seeds 1 2 3 --gamma 0.995 --eval-interval 1 --diagnostics \
  --episode-log runs/five_year_periodic_eval_seeds123_run2.jsonl \
  --plot-python /Library/Frameworks/Python.framework/Versions/3.9/bin/python3
```

The episode horizon, worker count and action count already default to
1,825 / 6 / 100. They can be stated explicitly with
`--episode-days 1825 --parallel-envs 6 --action-bins 100`.
Use a new log filename for every run; existing files are never overwritten.
For console-only execution, replace `--plot-python ...` with `--no-plot`.

Launch the file-backed entry point above so spawned workers can re-import
the main module. Training resume is not implemented; start a fresh
run to use changed source or settings. Optional evaluation checkpoints are
described below. Metrics are saved only when an
`--episode-log` path is supplied. With plotting enabled, all three figures are
automatically saved beside the log before their windows open:

```text
runs/five_year_periodic_eval_seeds123_run2_training.png
runs/five_year_periodic_eval_seeds123_run2_evaluator_return.png
runs/five_year_periodic_eval_seeds123_run2_violation_rates.png
```

`--no-plot` skips automatic figure creation and display; saved logs can still
be plotted or exported afterward.

`--episode-log` writes run, episode and update summaries to JSONL, and compact
daily metrics to one CSV per seed. For the command above the daily files are:

```text
runs/five_year_periodic_eval_seeds123_run2_seed1_steps.csv
runs/five_year_periodic_eval_seeds123_run2_seed2_steps.csv
runs/five_year_periodic_eval_seeds123_run2_seed3_steps.csv
```

Each completed daily transition produces one CSV row containing 15 scalar
values: phase, seed, episode, one-based episode step, source action date,
collected training steps, team reward, generation reward, guaranteed-output
penalty, ecological-flow penalty, generation (MWh), guaranteed-output shortfall
(MW and MWh), and ecological-flow shortfall (m3/s and m3).

Generation and physical deficits are summed over the five reservoirs. Penalties
use the weights and reservoir means from the legacy team reward. Daily CSV rewards
are undiscounted; apply `gamma**(episode_step-1)` to recover the discounted
episode total. CSV values use 10 significant digits; internal training and
summary calculations retain full precision. An unavailable service measurement
is an empty cell, while a measured zero deficit is written as zero.

Training and final evaluation share each seed's CSV, distinguished by the
`phase` column. Six environments produce six rows per collector step. A
300-episode seed produces 547,500 training rows and 1,825 evaluation rows. Each
row is flushed to disk as it is produced. The table can be opened in Excel or
filtered in Python using `seed`, `episode` and `episode_step`. Existing full
trajectory logs retain their original format; restart with a new output name
to use compact logging. Periodic evaluations add only summary records to JSONL,
so they do not expand the daily CSVs.

Optional final-policy XLD dispatch tables are described under **Xiluodu dispatch
diagnostics** below. They are separate from these team metrics and do not add
per-reservoir rows to the training CSVs.

To reopen training, evaluator-return and violation-rate curves from a saved or currently running log:

```bash
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 examples/plot_cascade_parallel.py \
  --episode-log runs/five_year_periodic_eval_seeds123_run2.jsonl
```

To export all three figures without opening windows, add
`--output result/periodic_training.png --evaluator-output result/periodic_evaluator_return.png --violations-output result/periodic_violation_rates.png --no-show`.
With only `--output NAME.png`, the additional figures are named
`NAME_evaluator_return.png` and `NAME_violation_rates.png`. The plot command streams the JSONL summaries.
The evaluator and violation-rate figures are available as soon as Round 0 has been recorded;
training curves become available after the first update. Running logs show
available seeds and label their mean accordingly. Means use only common
observed rounds/steps. A failed evaluation remains a gap, including in the
seed mean; surviving seeds or windows are not silently reweighted.
Each subplot displays one across-seed mean curve and a shaded +/-1 sample
standard deviation band. Individual seed curves are hidden; all per-seed values
remain in the logs. No smoothing or standard-error conversion is applied.

Older periodic logs, including `five_year_periodic_eval_seeds123_run1.jsonl`,
contain penalty totals but omit the violation counts required for these rates.
The plotter still opens their training and return curves, with an explicit
warning that violation-rate curves are unavailable. Requesting
`--violations-output` for such a log reports an error. Penalty magnitudes cannot
reconstruct violation frequencies, and historical intermediate models were
not saved; start a new run to record rates at every evaluation round.

Check data, calendars and reset observations for all training windows and the
evaluation sequence without training:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B examples/train_cascade.py \
  --gamma 0.995 --check-data-only
```

`--skip-final-evaluation` reserves held-out data during tuning.
`--log-interval N` controls training diagnostic reporting in update rounds.
`--eval-interval N` controls periodic policy evaluation independently. It must
be positive and defaults to 1. Round 0 and the last update are always evaluated;
for example, a 50-update run with interval 7 evaluates rounds 0, 7, 14, 21, 28,
35, 42, 49 and 50.
`--evaluation-years` accepts exactly five unique years and uses the remaining
55 for training. `--help` describes the supported flags. The CLI exposes only
the current multi-year training pipeline. These checks do not predict whether
all later daily transitions will be feasible under a particular policy.

## Configuration and networks

`harl/configs/algos_cfgs/happo_cascade.yaml` contains the complete training,
network, optimizer and split settings. There is no separate collection overlay.
`harl/configs/envs_cfgs/cascade_reservoir.yaml` contains physical assumptions and
reward settings. CLI arguments override their corresponding YAML values.

| Setting | Default |
| --- | --- |
| Episode horizon / parallel environments | 1,825 days / 6 |
| Discrete actions per reservoir | 100 |
| Episodes per seed / seeds | 300 / 1, 2, 3, 4, 5 |
| Encoder / additional hidden head | `[64, 128, 256]` / 256 |
| Stored hidden widths / activation | `[64, 128, 256, 256]` / ReLU |
| Actor / critic learning rate | 0.0003 / 0.0003 |
| Actor / critic optimization epochs | 5 / 5 |
| Actor / critic minibatches per epoch | 1 / 1 |
| Gamma / GAE lambda | 0.995 / 0.95 |
| Local guaranteed-output violation cost | 1.0 per actual violating reservoir-day, before 50/50 reward mixing |
| PPO clip / entropy coefficient | 0.2 / 0.001 |
| Maximum gradient norm | 0.5 |
| ValueNorm / per-actor advantage normalization | Enabled / enabled |
| Daily water-level limits | New action-date CSV, enabled |
| Periodic evaluation interval / scenarios | Every update / 3 fixed training-only windows |
| Training-window sampling | `coverage_balanced`; `uniform` restores the previous sampler |
| Guaranteed-output / ecological-flow threshold scale | Environment YAML values; CLI overrides are available |

Both the default configuration and launch commands select gamma=0.995 for
GAE, critic returns and return accounting. Actor policy and entropy losses use
uniform rollout rows with policy active masks; no extra gamma**t factor is
applied to either loss. The full-row averaging denominator remains unchanged.
The local learning reward includes `guaranteed_output_violation_weight` times the existing
tolerance-filtered violation indicator. This supplements relative deficit cost:
the latter measures severity, whereas the new term targets frequency. Setting
that weight to 0 restores the previous local reward formula. Constraints remain
soft; actions are not replaced or projected to satisfy service requirements.
The legacy team return in existing plots remains generation minus mean relative
deficits and does not include this learning-only term. Compare physical violation
rates, energy and final storage across configurations; returns with different
gamma values cannot be compared directly. A separate lambda experiment can use
`--gae-lambda 1`; it is not the default.

HAPPO actors optimize the sequential clipped policy surrogate with entropy
regularization, using the same normalized team advantage from actual trajectories.
The team reward is `0.5 * sum(local rewards) + 0.5 * global generation reward`.
It is the existing `system_reward`, not the sum of five mixed rewards: summing
those mixed rewards would count the global term five times. GAE uses the scalar
team value, gamma 0.995 and lambda 0.95. The raw team advantage is normalized
once over all common rollout rows before the sequential actor loop, and each
actor receives an identical copy. Its own controllability mask and HAPPO factor
are applied afterward in the loss. No per-actor re-normalization is performed.
The model-assisted command imitation objective and its candidate-cost data flow
have been removed. Service requirements influence learning through the reward;
there is no preferred-command target or additional physics-derived actor loss.

Use `--guaranteed-output-violation-weight 0` to disable the added reward frequency
cost. Combining that option with `--gamma 0.99` recovers the previous discount
and local reward formula. Effective settings are recorded in run metadata.
Do not count the frequency-cost detail twice: it is already part of the local
guarantee penalty component. Older runs with command imitation enabled retain
their original methodology and do not establish performance for this version.

With daily limits enabled, each actor observes 15 local/neighbor values and
the critic receives 38 centralized state values:

```text
Each actor: 15 -> 64 -> 128 -> 256 -> 256 -> 100 categorical logits
HAPPO critic: 38 -> 64 -> 128 -> 256 -> 256 -> 1 team value
MAPPO critic: 38 -> 64 -> 128 -> 256 -> 256 -> 6 values
```

Each network has separate weights. Hidden layers use
`Linear -> ReLU -> LayerNorm`; logits and the final value remain linear.
HAPPO uses one centralized network and one ValueNorm channel for the team
return. All five actors use value channel 0 and the same once-normalized GAE,
including when `normalize_shared_advantages` is false. MAPPO retains one
centralized network whose outputs 0--4 estimate WDD/BHT/XLD/XJB/THR mixed
returns and output 5 estimates the legacy team return. MAPPO ValueNorm tracks
each output separately, and its existing normalization flag retains its prior
per-reservoir behavior. The diagnostic team value and GAE fields identify
their reward basis explicitly; HAPPO no longer trains a legacy value head.

Both HAPPO and MAPPO update ValueNorm once per completed rollout batch, using
all critic targets in `returns[:-1]`, before the critic epochs. GAE, actor
advantages and frozen diagnostics first use the old moments; all actor updates
finish before this statistics update. HAPPO retains one common team advantage;
MAPPO retains five individual mixed-return advantages. MAPPO updates all six
value channels together, with separate mean and variance statistics per channel.
The new moments stay fixed across all five configured critic epochs and their
minibatches. Previously the moments were updated once per critic minibatch, or
`epochs * minibatches` times per rollout. The metadata field
`value_normalization_update` records the schedule, or `disabled` without ValueNorm.

The ValueNorm beta is retained; both algorithms' EMA clocks now count rollout
batches rather than reused optimization minibatches. This is not PopArt: critic output
weights are not rescaled when the moments change. Post-update values use the
new moments, while frozen pre-update values retain their original reward-unit
interpretation. Clipped value loss combined with ValueNorm remains unsupported.
This change defines the normalization schedule; improved performance has not
been established.

`--no-daily-level-limits` selects global operating bounds only, with input
dimensions 11/18.

The actor's 15 features, in order, are:

1. Own storage fraction, standardized current external inflow and previous release fraction.
2. Immediate upstream storage fraction, standardized external inflow and previous release fraction.
3. Immediate downstream storage fraction and room below its next-state upper storage limit.
4. Own current-state lower and upper storage fractions.
5. Own next-state lower and upper water-level fractions.
6. Day-of-year sine/cosine and fraction of the episode remaining.

Water-level fractions use the reservoir's global operating elevation span;
storage fractions use its global operating storage span. The current-state
limits come from the previous executed action's CSV row, with global bounds
at reset. The centralized critic receives storage, current external inflows,
previous releases, three time features and all five reservoirs' four limit
features. The actor observation layout is `local_neighbors_next_day_limits_v2`.

## Optional counterfactual action credit for HAPPO

The default `--happo-credit-assignment team_gae` retains the shared team-GAE
actor update described above. The opt-in `counterfactual` mode tests whether
action-specific feedback improves guaranteed-output decisions while keeping
the same shared `system_reward`, reservoir dynamics, and scalar centralized
value function. This is an experimental, model-assisted HAPPO variant; it is
not a claim of improved reliability or an unchanged standard HAPPO estimator.

For every collected state, reservoir, and categorical command, the variant
keeps the other four originally sampled commands fixed and computes

```text
Q_i(k) = system_reward(s, k, a_other) + gamma * V_frozen(s_next(k))
baseline_i = sum_k pi_old_i(k | local_observation_i) * Q_i(k)
credit_i = Q_i(actual_sampled_command_i) - baseline_i
```

The current-day reward and entire downstream cascade transition are evaluated
with the physical model. The next-state value uses the collection critic and
its original ValueNorm moments; a terminal transition has zero continuation
value. The candidate next state contains that command's resulting storages and
releases, while retaining the actual next day's forcing, bounds and time
features. All commands, including a command that fails a service requirement,
remain available. There is no automatic command replacement or hard service
mask.

The complete old categorical distribution, original sampled commands and
physical starting state are frozen during collection. Counterfactual credit
is computed once after the complete rollout and before actor or critic
optimization. Each actor receives its own credit array, divided by one common
RMS scale over controllable rollout/agent rows. No mean is subtracted, so a
below-baseline selected command retains negative credit. Forced choices have
zero credit. The existing PPO clipping, entropy term, controllability masks and
sequential HAPPO likelihood factors are retained. The critic still fits the
existing team GAE or discounted-return targets, separately from the new
one-step actor estimator.

The frozen critic supplies an estimate of future water value, not a verified
long-term optimum. Model error, limited actor observations and noisy value
estimates can still lead to poor decisions. Counterfactual alternatives are
used for credit estimation and do not alter collected physical trajectories.
MAPPO retains its original estimator; a comparison request combining MAPPO and
counterfactual HAPPO is rejected so that the experimental estimator is not
silently attributed to MAPPO. Use separate run logs for controlled comparisons.

For a short inspection run with the original 1,825-day horizon:

```bash
cd /Users/hujingxin/Documents/Code_python/HARL-main
RUN_DIR="runs/counterfactual_credit_$(date +%Y%m%d_%H%M%S)_$$"
mkdir "$RUN_DIR"
set -o pipefail

PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B examples/train_cascade.py \
  --happo-credit-assignment counterfactual --counterfactual-batch-size 4096 \
  --episodes 12 --episode-days 1825 --parallel-envs 6 --seeds 1 \
  --gamma 0.995 --eval-interval 1 --log-interval 1 \
  --trajectory-diagnostics --save-checkpoints --checkpoint-interval 1 \
  --episode-log "$RUN_DIR/happo_counterfactual.jsonl" \
  --plot-python /Library/Frameworks/Python.framework/Versions/3.9/bin/python3 \
  2>&1 | tee "$RUN_DIR/console.log"
```

For the full matched budget, use `--episodes 300 --seeds 1 2 3`. Keep the
baseline's other settings unchanged and launch the baseline separately with
`--happo-credit-assignment team_gae` and a different output path. The option
`--counterfactual-batch-size` limits candidate transitions per inference chunk;
it does not reduce the number of categorical alternatives evaluated. Exact
enumeration adds substantial computation, especially with 100 commands and
five reservoirs. It does not add environment transitions to the training-step
counter.

With `--episode-log`, counterfactual mode automatically creates a compressed
`<log_stem>_seedN_counterfactual_credit.csv.gz`. It records five rows per
collected day, with the full old action probabilities and candidate Q values,
selected-command Q, expected baseline, raw and RMS-scaled actor credit, best
candidate, controllability, and separate team-GAE advantage. Rows are written
and flushed before optimization. Per-update JSONL summaries report credit
statistics, computation time and selected-command Q reconciliation with the
actual rollout. Existing physical/learning traces label the actor estimator
separately from the scalar critic's GAE target. Evaluation and held-out replay
remain frozen-policy measurements and do not calculate actor training credit.

Checkpoint settings preserve the chosen credit mode and estimator metadata;
the critic continues to represent the shared team return. Evaluation replay
does not resume optimization or reconstruct the original counterfactual actor
advantages. A paused process retains the implementation it already loaded;
new source changes apply to a newly launched experiment.

## Episode calendar and evaluation

Training windows start on January 1 and contain exactly 1,825 consecutive
source days. Leap days are retained, so the horizon can finish one or two
days before its fifth calendar anniversary. Every executed day must belong
to training data; the default split provides 39 eligible windows.

Storage, previous releases, GAE and discount clocks carry across calendar
boundaries. Time-to-go is scaled by the complete episode horizon. Final storage
is subject to the final action's daily bounds; it need not equal initial storage.

Periodic evaluation measures the initial policy at Round 0 and the updated
policy after each configured interval. Under the default split, fixed scenarios
are **2007--2011 (dry), 1991--1995 (median), and 1973--1977 (wet)**. Each contains
exactly 1,825 consecutive days, ending December 30 of its fifth source year.
The three scenarios are selected once by minimum, median and maximum mean
total external inflow among complete training-only windows, with start year
breaking ties. Selection uses no RNG. Their source dates, inflow hashes and
initial storage/levels are recorded in `run_configuration.periodic_evaluation`.
All seeds and rounds use those same scenarios and initial conditions. Labels
describe training-window flow ranks, not the held-out year classifications.

Each scenario uses a fresh environment and deterministic argmax actions from
all five actors under `torch.no_grad()`. Random-number states and every actor
module's training/evaluation mode are restored afterward. Evaluation samples
never enter collection buffers or optimizer updates. Training progress counts
only collector steps. These diagnostics do not rank/select policies and do not
use any held-out source days.

One `periodic_evaluation` JSONL record per seed/round preserves the three
individual window results plus their arithmetic mean: `evaluator_return`,
`generation_MWh`/`generation_TWh`, `guaranteed_output_penalty`,
`ecological_flow_penalty`, `final_storage_m3` and `final_level_m`. Storage/level
vectors follow WDD/BHT/XLD/XJB/THR order. Return and penalties use the unchanged
configured discount and reward formula. The record also identifies
`evaluation_round`, `seed`, `training_steps`, status and successful-window count.
Physical infeasibility is recorded with its window, failed date/step and reason;
partial metrics are labelled separately. If any window fails, the aggregate
metrics are null and the evaluator curve leaves a gap. Other windows are still
checked and training continues. Unexpected programming errors propagate.

The additional unsmoothed figure is titled **Evaluator Return vs. Evaluation
Round**. Each seed's recorded return is the mean over its three fixed scenarios;
the displayed curve and band show only the across-seed mean and +/-1 sample SD. The
default 50-update experiment records rounds 0--50. This measures performance
and does not impose convergence or alter rewards to obtain a plateau.

A separate two-panel figure shows **Ecological-flow violation rate** and
**Guaranteed-output violation rate** against evaluation round. Each rate is
the number of reservoir-days with a positive shortfall divided by all
reservoir-days with an active requirement, pooled over the three fixed windows.
For the default five active reservoirs, each seed/round has a denominator of
`3 * 1825 * 5 = 27,375` reservoir-days per constraint. Counts use the existing
shortfall tolerance and are undiscounted, independent of gamma and penalty
weights. This is violation frequency, not relative shortfall severity or the
fraction of days with any reservoir violating a requirement.

Periodic records and individual windows now retain
`guaranteed_output_violation_rate` and `ecological_flow_violation_rate` as
fractions in [0, 1], their `*_violation_count` and
`*_active_reservoir_days` fields, and per-window `service_shortfalls` summaries.
The figure converts rates to percent and shows one across-seed mean curve
and +/-1 sample SD without smoothing. Failed windows leave aggregate rates
null and curve gaps; no active requirement also yields a null rate.

Final independent evaluation joins the five representative years in their configured order,
carries storage between them, and uses the same episode horizon. Their forcing
and date-specific daily bounds are retained. This is a constructed representative sequence,
rather than a continuous historical record. The evaluator scores the final
policy without ranking intermediate policies. A physical failure is logged
explicitly with partial progress before the existing final-evaluation error is
re-raised; successful final evaluation behavior is unchanged.

`--episode-days` also supports explicitly shortened collection checks. Periodic
scenarios still retain their full 1,825-day horizon. The collector checks'
actual dates and horizon are recorded; their metrics do not describe a
full-length experiment.

## Training-window coverage

`--window-sampling coverage_balanced` is the default. It targets equal numbers
of simulated training days per calendar year, rather than equal probabilities
for overlapping five-year windows. The coverage matrix uses actual source
dates, including leap days and truncated final calendar years. It never adds
held-out days or joins nonconsecutive years to form training episodes.

The sampler computes constrained window probabilities, converts them to a
fixed-budget integer schedule and shuffles that schedule using a private RNG
for each seed. Parallel batches contain distinct window starts. Every window
is included when the episode budget permits; short runs cannot guarantee full
coverage. There are no additional training episodes or optimizer updates.

Exact equal coverage is generally impossible under the fixed five-year horizon
and split gaps. Logs retain the residual imbalance instead of claiming perfect
balance. `run_configuration.training_window_sampling` and per-seed
`training_sampling_plan` records describe the target probabilities and planned
coverage. Each `training_update.training_sampling_coverage` reports cumulative
window counts, year appearances and actual simulated days per year. Final
coverage is also recorded in `run_complete.training_sampling_coverage_per_seed`.
These counts include successfully collected full training batches; periodic
evaluations and partial failed batches are excluded. A failed batch is logged
as `training_sampling_aborted` and is not replaced to satisfy the schedule.

For comparison, `--window-sampling uniform` preserves the previous independent,
uniform draws with replacement, using the same per-seed random sequence. It
allows duplicate windows within a batch. Both modes leave the fixed periodic
evaluation windows and final held-out evaluation unchanged. There is no new
wet/normal/dry classification or hydrological-class stratification.

For a G40/E60 sampler experiment using the current Actor formulation and
discount, use:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B examples/train_cascade.py \
  --episodes 300 --action-bins 200 --seeds 1 2 3 --gamma 0.995 --eval-interval 1 \
  --window-sampling coverage_balanced \
  --guaranteed-output-scale 0.4 --ecological-flow-scale 0.6 --diagnostics \
  --episode-log runs/five_year_bins200_g40_e60_coverage_balanced_seeds123_run1.jsonl \
  --plot-python /Library/Frameworks/Python.framework/Versions/3.9/bin/python3
```

## Service-threshold sensitivity

`--guaranteed-output-scale` and `--ecological-flow-scale` apply positive
multipliers to the reference thresholds. Their defaults come from the environment
YAML. A value of **1.0** uses the full reference requirement; **0.40** uses 40%
of that requirement. Multipliers always
apply directly to the original CSV, so reductions do not compound. The reference
CSV remains unchanged; effective and
base values, multipliers and the input file hash are recorded in the run configuration
and service-requirement metadata. Adjusted values are labelled
`sensitivity_adjusted`, rather than paper-reported requirements.

The two CLI values are multipliers, not absolute MW or m3/s values. For example,
`--guaranteed-output-scale 0.4 --ecological-flow-scale 0.4` gives XLD thresholds
of `5345.7 * 0.4 = 2138.28 MW` and `1160 * 0.4 = 464 m3/s`.

For a 40% / 40% sensitivity experiment, set these keys in
`harl/configs/envs_cfgs/cascade_reservoir.yaml`:

```yaml
guaranteed_output_scale: 0.4
ecological_flow_scale: 0.4
```

Remove the corresponding CLI arguments to use those defaults; explicit CLI
values always override YAML. Individual reservoir reference values are in
`data/cascade_reservoir_new/reservoir_service_requirements.csv`, in columns
`guaranteed_output_mw` and `ecological_flow_m3s`. If that CSV is changed to hold
custom absolute effective thresholds, set both scales to 1 (in YAML and any CLI
overrides) to avoid multiplying them again. Editing the CSV redefines the
reference inputs; preserve accurate source provenance when doing so. Changing
thresholds does not require editing the reward function or physics code.

For a separate 50% / 70% sensitivity experiment, use the following explicit overrides, keeping
200 actions, seeds 1/2/3 and the existing training settings:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B examples/train_cascade.py \
  --episodes 300 --action-bins 200 --seeds 1 2 3 --gamma 0.995 --eval-interval 1 \
  --guaranteed-output-scale 0.5 --ecological-flow-scale 0.7 --diagnostics \
  --episode-log runs/five_year_bins200_g50_e70_mean_sd_seeds123_run1.jsonl \
  --plot-python /Library/Frameworks/Python.framework/Versions/3.9/bin/python3
```

| Reservoir | Effective guaranteed output (MW) | Effective ecological flow (m3/s) |
| --- | ---: | ---: |
| WDD | 1,145 | 630 |
| BHT | 2,283.75 | 812 |
| XLD | 2,672.85 | 812 |
| XJB | 1,375 | 840 |
| THR | 2,495 | 4,200 |

The effective thresholds are used consistently by training, periodic evaluation,
final held-out evaluation, reward penalties and violation counts. Penalty weights,
reward formula, reservoir physics and hard operating bounds are unchanged. All
requirements remain active, so the periodic rate denominator remains 27,375
reservoir-days per seed/round. Use both scales equal to 1 for the reference setup.
Use both equal to 0.9 to reproduce the previous relaxed-threshold experiment.
Use both equal to 0.75 to reproduce the previous 75% experiment.
Changing thresholds changes the optimization and reporting problem: lower
violations or higher returns under relaxed requirements alone do not establish
improvement against the original requirements. Physical generation remains in
the same units for comparison.

## Xiluodu dispatch diagnostics

Add `--xld-diagnostics` to either training entry point to inspect the final
policy's XLD guaranteed-output violations. No `--episode-log` is required:
without a log, diagnostics remain in memory and print a terminal summary without
creating CSVs, checkpoints, JSONL or extra figures. For the current comparison:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B examples/train_cascade_compare.py \
  --algorithms happo mappo --episodes 300 --seeds 1 2 3 --gamma 0.995 \
  --eval-interval 1 --diagnostics --xld-diagnostics \
  --plot-python /Library/Frameworks/Python.framework/Versions/3.9/bin/python3
```

The terminal separately reports HAPPO/MAPPO and fixed-training/held-out scopes.
For each scope it prints the XLD violation rate and three exhaustive groups:

- `actor_missed_feasible_discrete`: a configured XLD command could meet the
  requirement in the reached state, but the actual policy did not select it.
- `discrete_grid_gap`: a continuous command could meet it, but none of the
  configured discrete commands could.
- `same_state_continuous_unattainable`: no continuous command could meet it in
  that state; capacity, water, dispatch and head labels are also reported.

Group fractions use XLD **violation days** as their denominator. The XLD violation
rate uses all active XLD days, not all five reservoirs' days. Counts are pooled
across complete windows and seeds, rather than averaging percentages. Incomplete
evaluation coverage is explicitly reported without a pooled rate. Each window's
summary remains in the returned evaluation records; the final pooled data is
available as `xld_dispatch_summary` in the in-memory plot payload and optional log.

Diagnostics observe only the last scheduled periodic evaluation (three fixed
training-only windows) and the final held-out sequence. They do not add training
or evaluation episodes, inspect training-worker trajectories or repeat daily
searches at earlier evaluation rounds. A standard run observes 7,300 XLD days per
seed and algorithm, or 5,475 when final held-out evaluation is skipped.

For daily CSV evidence, additionally provide a new `--episode-log` path. A
separate 200-command, 40% requirement sensitivity example is:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B examples/train_cascade.py \
  --episodes 300 --episode-days 1825 --parallel-envs 6 \
  --action-bins 200 --seeds 1 2 3 --gamma 0.995 --eval-interval 1 \
  --window-sampling coverage_balanced \
  --guaranteed-output-scale 0.4 --ecological-flow-scale 0.4 \
  --diagnostics --xld-diagnostics \
  --episode-log runs/five_year_bins200_g40_e40_xld_seeds123_run1.jsonl \
  --plot-python /Library/Frameworks/Python.framework/Versions/3.9/bin/python3
```

With a log, the additional files are named `<log_stem>_seed1_xld_dispatch.csv`, and likewise
for seeds 2 and 3, beside the JSONL log. Filter their classification column to
review violations. Scalar records include storage and daily bounds, actual
inflow and upstream release, selected action, feasible release limits, turbine
flow, spill, effective head, power, guarantee and shortfall. Violating days also
include same-state alternative-action and continuous-release power limits.
Counts and diagnostic definitions are retained in each evaluated window's
`xld_dispatch_diagnostics` JSONL summary; actual physical failures retain their
existing failure records and any partial diagnostic summary.

For a violating day, the diagnostic holds today's state, external inflows,
action-date bounds and the other four actors' originally sampled actions fixed.
It checks all XLD action indices and accounts for the actual change in
end-storage and head. A continuous-release check includes stage-storage curve
breakpoints and interior power maxima; maximum discharge need not maximize
power. Selected alternatives are verified by the existing full-cascade physics.
The analysis distinguishes a discrete alternative that meets the guarantee,
a gap caused by the discrete grid, and same-state water, dispatch, capacity or
head limitations. Reference-head and optimistic head calculations are diagnostic
bounds, rather than changes to the physical model.

A feasible same-day alternative establishes that the guarantee could be met
from that state. It does not establish the best five-year strategy: releasing
more water changes later storage, while an unfavorable current state may itself
have resulted from earlier actions. No future inflows or deadline targets are
used. Diagnostics never replace policy actions, alter rewards, update gradients,
or change the executed trajectory. Unexpected counterfactual failures raise a
diagnostic error instead of being counted as failures of the executed policy.

## Physics, actions and reward

Raw data and source notes live in `data/cascade_reservoir_new/`. Hard constraints
cover water balance, global operating storage, daily CSV bounds, outlet capacity
and compatibility with downstream storage and outlet limits in the same transition.
Release intervals use current storage, the current day's external inflows and
the bounds on the resulting state.

`five_reservoir_happo_input_bounds_1965_2024.csv` is the authoritative daily limit
table. A row with `action_date=t` already gives the allowed level and storage
of the state resulting from action t. Both physics and observations use that
row directly. There is no additional shift to the following date. At source-year
joins, the previous executed row constrains current storage and the newly selected
action-date row constrains the next storage.

There is no remaining-horizon reachability calculation, future-inflow lookup,
refill target, or return-to-initial-storage condition. Some supplied daily ranges
have equal lower and upper limits; those rows still require that single level.
The environment raises an error when a current transition is infeasible instead
of silently changing the storage balance. Without an end-storage target, episode
generation can include depletion of initially stored water, which remains reported.

The default `storage_sensitivity` head model uses mean starting/ending forebay
elevation, constant efficiency 0.9, and supplied slope/offset assumptions.
This is a sensitivity scenario, not a calibrated tailwater model. Turbine flow
and installed capacity limit generation. `--head-model fixed` uses typical heads.

For index `k` among `K=100` actions:

```text
u = k / (K - 1)
release = feasible_low + u^2 * (feasible_high - feasible_low)
```

Local and global reward components use a mixing coefficient of 0.5:

```text
local_i = power_i / installed_capacity_i
          - guaranteed_output_weight * relative_power_deficit_i
          - ecological_flow_weight * relative_flow_deficit_i
          - guaranteed_output_violation_weight * power_violation_indicator_i
global = sum(power_i) / sum(installed_capacity_i)
mixed_i = 0.5 * local_i + 0.5 * global
system = 0.5 * sum(local_i) + 0.5 * global
```

Local penalties use each reservoir's own requirement-relative deficit, without
averaging over reservoirs. Existing effective thresholds, weights and penalty
enablement are reused. The frequency cost uses exactly the same tolerance-filtered
shortfall indicator as physical compliance counting; it neither lowers thresholds
nor changes counts. Its per-reservoir amount is logged separately as a component
of the local guarantee penalty. All service costs are disabled when penalties
are disabled. Zero requirements remain inactive. The requested global
term is generation only; no navigation or action-correction penalty is added.
System accounting includes the global term once and is not the sum of mixed
rewards. Episode JSONL records include discounted per-reservoir local/mixed
totals and system/global totals; periodic and final evaluations report them too.

HAPPO uses `system` as its common learning reward. One centralized scalar
value output estimates this team return. Its GAE is normalized once over common
rollout rows and shared by all five actors. The global component enters `system`
only once. HAPPO's command ratios, scalar sequential factors, clipping, update
order and control masks are retained; each actor uses the same team advantage
with its sequential importance factor. These approximate clipped updates do
not establish a monotonic improvement guarantee for measured scheduling results.
MAPPO retains `mixed_i` as each reservoir's learning reward, with five centralized
mixed-return value outputs and a sixth legacy-return output for accounting
diagnostics. These are outputs of one network, not six independent critics.

The preserved daily team reward is normalized generation minus mean relative guaranteed-output
and ecological-flow deficits, each with weight 1. Generation is normalized by
1,624,800 MWh. Requirements and their source are retained in
`reservoir_service_requirements.csv`. `--no-reward-penalties` selects a separate
generation-only objective.

Gamma applies to return estimation and episode reward accounting. Actor loss
rows have uniform temporal weight. Physical violation counts, deficit volumes and deficit
energy remain undiscounted. Episode generation and return are totals over the
whole horizon. Soft costs express a tradeoff; hard feasibility does not imply
zero service shortfalls.

## Daily water-value diagnostics and saved-policy replay

`--trajectory-diagnostics` links all five reservoirs' physical decisions to the
actual learning signals: shared team signals for HAPPO and individual mixed
signals for MAPPO. It implies `--diagnostics` and requires a
new `--episode-log`. A separate `<stem>_seedN_trajectory.csv` contains one row
per training day, lane and reservoir. Existing team metrics and XLD evaluation
diagnostics keep their definitions and output files.

Each row records start/end storage and levels, physical inflow/release/turbine
flow/spill/head/power, feasible release bounds, sampled and effective commands,
local/global/mixed rewards, service deficits, frozen current/next values with explicit reward basis,
TD residual, GAE target, raw and normalized advantage, realized reward-to-go,
action probability, entropy, objective weight and the exact HAPPO factor at
that actor's sequential update entry. Values are frozen before optimization
and before ValueNorm moments change. Forced commands retain their sampled
probability separately from the effective action/log-probability placeholders;
their policy-objective weight is zero. The compatibility column
`episode_time_weight` is always 1.0; it is not an episode-age discount.
Global, system and legacy-team reward fields repeat across the five reservoir
rows for the same day/lane. Count each of those scalars once per day/lane when
reconstructing system accounting; do not sum their repeated CSV cells.

JSONL `trajectory_diagnostics` records summarize all days, the first 14 days,
and calendar months October-May separately for each reservoir. Daily rows are
correlated, realized Monte Carlo returns are not true expected values, and a
positive advantage on a spilling day alone does not prove that spill was
avoidable or caused the advantage. These are observational measurements;
no reward, action, critic target or optimizer decision changes.

Tracing every update at the standard 300-episode budget produces 2,737,500 rows
per seed. Begin with a short diagnostic run, for example two updates:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B examples/train_cascade.py \
  --episodes 12 --episode-days 1825 --parallel-envs 6 \
  --action-bins 200 --seeds 1 --gamma 0.995 --eval-interval 1 \
  --window-sampling coverage_balanced \
  --trajectory-diagnostics --save-checkpoints --checkpoint-interval 1 \
  --xld-diagnostics --skip-final-evaluation --no-plot \
  --episode-log "runs/water_value_diagnostic_$(date +%Y%m%d_%H%M%S).jsonl"
```

`--save-checkpoints` requires `--episode-log` and saves the final update into
`<stem>_seedN_checkpoints/update_000002.pt` in this example. Add
`--checkpoint-interval N` to also save every N completed updates; its default
0 saves only the final update. Files and existing checkpoint directories are
never overwritten. Checkpoints contain all five actors, the single centralized
critic (one team output for HAPPO, six outputs for MAPPO), ValueNorm, effective
settings and source/data hashes. They
support evaluation replay, not training resume.

Replay a saved policy on an ordered training-only or held-out-only source
sequence without changing data or fitting models:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B examples/replay_cascade_checkpoint.py \
  --checkpoint runs/EXPERIMENT_seed1_checkpoints/update_000002.pt \
  --source-years 2007 2008 2009 2010 2011 \
  --output runs/EXPERIMENT_baseline_replay.csv
```

Repeat with a new output path and `--override-action 1 BHT 16` to replace just
the first BHT command. Later commands follow the same saved policy as it reacts
to the changed state. The summary reports energy, service deficits, all five
learning returns, system return, spill and final storage. The five-row-per-day
CSV links physical dispatch to saved values and the corresponding realized
learning reward-to-go, with explicit reward layout and value-basis fields.
It also reports GAE recomputed on the replay trajectory; this does not
reconstruct the original training advantages. Comparing baseline and deviation
under the same forcing measures that particular intervention, not optimality.

Replay requires the saved episode horizon, preserving the policy's remaining
time feature and terminal semantics. It rejects source/config or dataset hash
mismatches. A 16-day smoke-test checkpoint therefore replays 16 days, while a
production checkpoint replays its full 1,825 days. Year-specific initial storage
comes from the fingerprinted dataset; no full-storage reset is substituted.

## Source and output structure

| File or directory | Responsibility |
| --- | --- |
| `examples/train_cascade.py` | CLI, complete configuration and 55/5 split |
| `examples/train_cascade_parallel.py` | Episode checks, fixed scenarios, seed orchestration, logging and final evaluation |
| `examples/plot_cascade.py` | Interactive backend availability check |
| `examples/plot_cascade_parallel.py` | Separate unsmoothed collector and evaluator figures from launch payloads or JSONL |
| `examples/replay_cascade_checkpoint.py` | Frozen-policy replay and isolated command-deviation comparisons |
| `harl/runners/cascade_reservoir_runner.py` | Parallel collection, unchanged HAPPO updates, periodic and final evaluation |
| `harl/envs/cascade_reservoir/periodic_evaluation.py` | Fixed training-only scenarios, state isolation, window metrics and failure accounting |
| `harl/envs/cascade_reservoir/trajectory_diagnostics.py` | Optional per-reservoir physical and frozen learning traces |
| `harl/envs/cascade_reservoir/checkpoint.py` | Evaluation checkpoint persistence, strict provenance checks and deterministic replay |
| `harl/runners/on_policy_ha_runner.py` | Sequential HAPPO updates and likelihood factors |
| `harl/algorithms/` | Actor and critic objectives and optimizer steps |
| `harl/envs/cascade_reservoir/` | Data, episode views, workers, current-transition physics, reward and training diagnostics |
| `harl/common/` and `harl/models/` | Buffers, normalization and feed-forward networks |

`--diagnostics` reports parameter changes, entropy/KL, surrogate changes and
Monte Carlo/GAE fit on collected training trajectories. Collector curves show recorded
stochastic returns and generation, averaged across the six complete episodes
collected before an update. They begin after collection and have no fabricated
episode-zero point. Sampled inflow windows affect their variation.

New JSONL records use `run_configuration`, `training_episode`,
`training_update`, `training_diagnostics`, optional `trajectory_diagnostics` and `policy_checkpoint`,
`periodic_evaluation`, `final_evaluation_sequence` and `run_complete`.
Per-episode records include source years, dates and the actual horizon;
generation units refer to the full episode. Historical files in `runs/` and
`result/` retain the settings and schemas of the runs that produced them.

Dependencies are declared in `setup.py`; install into a suitable Python
environment with `python -m pip install -e .`.

## Explicit target, correction, and execution (Eqs. 3, 4, 16)

The linear action mapping is retained. A sampled categorical index (k_i)
is normalized as (u_i=k_i/(K-1)), and maps to the current day's dynamic
interval as `target_release = low + u * (high - low)`. The interval is
not a fixed outlet range: backward downstream compatibility and forward
upstream-flow propagation enforce outlet capacity and the supplied daily
end-of-day storage/level bounds. Water balance is never repaired by changing
storage after the release is selected.

In `joint_feasible` mode, the optional execution-stage operator uses the same
supported hard constraints as mapping, so ordinary controllable targets need no correction.
There are no new ecological, guaranteed-output, navigation, or reserve
constraints in the operator. The service thresholds and existing weights
are unchanged.

`ReservoirPhysics.correct_releases()` also accepts an explicit target-release
vector for testing or external dispatch integration. It projects the upstream
target onto its downstream-compatible interval, then recomputes downstream
inflows and intervals from the releases actually executed. The nearest release
is chosen at each stage conditional on the executed upstream releases. This is
an upstream-priority lexicographic projection, not a claimed global Euclidean
projection of the five-dimensional vector. A jointly feasible supplied target
is preserved exactly. For external dispatch with separately supplied original
mapping bounds, `low`/`high` retain those mapping bounds and
`execution_low`/`execution_high` report the recomputed physical interval.
Empty feasible intervals raise `InfeasibleCascade`
with state and failure diagnostics; no water or storage is fabricated.

The paper explicitly separates target and executed release (Eq. 4) and defines
the correction deficit (Eq. 16), but does not specify the full projection
algorithm, constraint priorities, empty-set handling, or numerical conventions.
The upstream-priority projection, finite-safe inverse masks, roundoff
classification, and failure accounting here are explicit engineering choices.

### Configurations A, B, and C

The environment YAML contains independent controls:

```yaml
action_correction:
  enabled: false
  penalty_enabled: false
  tau: 0.03
  correction_weight: 1.0
```

- A: both switches off; the original linear mapping and implicit feasibility.
- B: correction on, correction penalty off.
- C: correction on and correction penalty on.

The switches do not alter networks, observations, PPO settings, existing
service weights, or requirement thresholds. The penalty switch is independent
of the existing service-penalty switch. With the new penalty disabled, the
original numerical reward calculation is preserved. The historical legacy-team
reward remains the service/generation accounting metric; the new penalty enters
the local learning reward, MAPPO mixed rewards, and HAPPO shared-system reward.
The shared global term is still counted exactly once in system accounting.

Use `--action-correction` / `--no-action-correction`,
`--action-correction-penalty` / `--no-action-correction-penalty`,
`--correction-tau`, and `--correction-weight` as CLI overrides.
For example, configuration C with both algorithms and unsaved interactive figures:

```bash
cd /Users/hujingxin/Documents/Code_python/HARL-main
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B examples/train_cascade_compare.py \
  --algorithms happo mappo --happo-credit-assignment counterfactual \
  --action-correction --action-correction-penalty --correction-tau 0.03 --correction-weight 1 \
  --episodes 300 --episode-days 1825 --parallel-envs 6 --seeds 1 2 3 --gamma 0.995 \
  --plot-python /Library/Frameworks/Python.framework/Versions/3.9/bin/python3
```

For B, replace the penalty flag with `--no-action-correction-penalty`.
For A, also replace the correction flag with `--no-action-correction`.
Omitting `--episode-log` leaves results and figures unsaved. Existing log,
trajectory, counterfactual trace, and checkpoint options remain available for
controlled experiments. Keep every other setting and seed schedule identical.

### Action audit and PPO accounting

Every normal environment transition distinguishes original discrete
`sampled_actions`, original `normalized_actions`, `target_release`,
`executed_release`, `release_correction_m3s`, `correction_applied`,
`correction_reason`, `low`, `high`, and `controls`.
`release` remains the backward-compatible alias of the executed release.
All generation, downstream inflows, storage, service deficits, and next-state
features use that executed release.

Additional audit fields include `realized_normalized_actions`,
`inverse_action_valid`, `executed_outside_mapping_interval`,
`normalized_action_correction_magnitude`, `correction_deficit`,
`forced_dispatch`, and `numerical_adjustment`.
The inverse is used only on a controllable nondegenerate interval when the
executed release lies inside the original mapping interval. Invalid inverses
have a finite zero sentinel plus an explicit false validity mask internally;
exports render the unavailable normalized values as null/blank. Discharge
corrections remain recorded. No inverse normalized value is silently clipped.

For valid genuine corrections, Eq. (16) is
`correction_deficit = max(0, abs(original_u - realized_u) - tau)`.
The signed local term is `r_corr = -correction_weight * correction_deficit`
when enabled. Forced dispatch and roundoff-sized adjustments incur no correction
cost. A finite flow tolerance distinguishes numerical changes from genuine
physical interventions independently of the normalized tolerance: the flow
threshold is `max(1e-8 m3/s, 32 * eps64 * max(1, abs(target), abs(executed)))`.
Widths at most `1e-8 m3/s` are forced dispatch. A margin of
`8 * eps64 * max(1, abs(action_difference), tau)` suppresses roundoff at tau.

PPO stores the original sampled index and its original old log probability,
including on forced rows. Forced rows retain their zero active mask.
Historical effective-action placeholders remain separate diagnostic fields.
Corrected dispatch must never replace a sampled command in a PPO likelihood
ratio. Counterfactual candidates use the same physics operator and correction
cost, and saved-policy replay uses the same configuration.

### Metric definitions and interpretation

`action_correction_violation_count` counts genuine, valid controllable
reservoir-days where `abs(original_u - realized_u) > tau`.
`action_correction_active_reservoir_days` is the number of controllable rows
with a valid inverse. Their ratio is `action_correction_violation_rate`;
a zero denominator produces null, not a fabricated zero rate. Per-reservoir
rates use the corresponding counts and denominators. The nested
`action_correction_summary` also reports genuine event counts, discharge and
normalized correction magnitudes, forced dispatch, numerical adjustments,
outside/invalid inverses, and no-feasible-execution events. Numerical-only
adjustments are excluded from the numerator; valid controllable rows with
numerical-only adjustments remain in the denominator.

A genuine discharge correction below the normalized tolerance is an event
but not an above-tolerance action-correction violation. Forced dispatch has
its own count and is excluded from the correction-rate denominator.
Failed execution attempts are counted as failures, not as executed
reservoir-days. Partial failed evaluations retain partial diagnostics and
leave completed rates null. Three-window rates pool counts and eligible
reservoir-days rather than averaging per-window percentages.

The ecological and guaranteed-output curves remain separate. New evaluation
histories add an action-correction panel; historical logs lacking the new
counts retain their original two panels rather than inventing zero corrections.

A zero correction rate is the expected valid result of the present model:
the mapping interval and execution checker enforce the same hard constraints.
Nonzero standalone-operator tests deliberately supply physically inadmissible
targets to verify repair; they do not introduce a new constraint or alter the
normal action space. This mechanism does not create water or establish
long-term optimal storage allocation, and no reliability improvement is assumed.

### Verification performed

Run the training/physics/integration suite in the training interpreter:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B -m unittest discover -s tests -q
```

The plotting interpreter covers the Matplotlib checks skipped by that interpreter:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests /Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -B -m unittest test_action_correction_controls.CorrectionFigureTest test_cascade_comparison_plot -q
```

A short process-parallel integration command, saving no results or figures:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B examples/train_cascade_compare.py \
  --algorithms happo mappo --happo-credit-assignment counterfactual \
  --action-correction --action-correction-penalty --episodes 2 --episode-days 3 \
  --parallel-envs 2 --seeds 1 --gamma 0.995 --no-plot --skip-final-evaluation
```

This smoke check completed both algorithms and their round-0/round-1
three-window evaluations. Each evaluation contained 27,375 valid reservoir-days,
zero genuine corrections and zero infeasible executions. Separate 1,825-day
random/boundary-action checks under both fixed and storage-sensitive heads
matched every existing baseline physics/reward field exactly against the
pre-change source, with zero difference between A and C releases. These are
implementation checks, not evidence of a full-budget learning improvement.


## CSV proposal mapping and state-dependent physical execution

This is a specified engineering adaptation of Eqs. (3), (4), and (16), not a
reproduction of unpublished correction code. The default remains
`release_mapping_mode: joint_feasible`. Its numerical physical execution and
reward calculations are preserved. Optional `csv_proposal` separates the target
reference interval from the actual feasible execution interval.

The supplied `five_reservoir_daily_release_bounds_1965_2024.csv` contains exactly
`action_date,reservoir_id,Q_min_m3s,Q_max_m3s`. `CascadeData` loads it once only
when CSV mode is selected, validates every actual date and reservoir, records
its SHA-256, and stores immutable `(date, reservoir)` arrays. `EpisodeData`
slices these arrays by original source indices, and the environment resolves
each row by its actual action date. Leap days and nonconsecutive replay years
are retained without shifting the CSV date to the next day. A missing file,
duplicate/missing row, invalid schema or invalid numerical bound fails explicitly.
Zero-width reference intervals are accepted and diagnosed; above-capacity bounds
are reported, never silently clipped by the loader.

The headers declare m3/s. The user identifies these values as total reservoir
release references derived from historical inflows and admissible storage
envelopes. No generator or independent provenance record was supplied. Values
often exceed turbine capacities, consistently with total release including spill,
but column names and numerical checks cannot authenticate operating rules.
Historical daily inflows may only be known retrospectively or via forecasts;
without the construction algorithm, no claim of causal availability or absence
of future information is justified. This is an offline reference experiment,
not independently verified statutory operating limits.

In CSV mode the normalized Actor command maps only through the supplied reference:

```
u = sampled_index / (action_bins - 1)
Q_target = proposal_low + u * (proposal_high - proposal_low)
```

The existing backward `downstream_intervals()` pass and forward actual executed
upstream flow determine `physical_low` and `physical_high`. The conditional
execution projection is:

```
Q_exec = clip(Q_target, physical_low, physical_high)
S_next = S + dt * (external + executed_upstream - Q_exec)
```

CSV bounds are never intersected into the hard physical constraints. Empty
physical intervals raise the existing structured `InfeasibleCascade`; storage
is never adjusted to conceal conflicts. Already feasible targets are preserved,
subject to the existing numerical and forced-dispatch conventions. Total release,
not the target, drives water balance, generation, service deficits and next states.

`proposal_low/high` and `physical_low/high` are always recorded. Historical
`low/high` remain physical mapping bounds in joint mode; in CSV mode they alias
the proposal interval used for inversion. All new diagnostic consumers use the
explicit interval names. CSV intervals disjoint from physical execution intervals
are separately flagged `proposal_physical_disjoint`.

For a nondegenerate proposal, `u_realized=(Q_exec-proposal_low)/proposal_width`.
A valid inside-domain inverse uses Eq. (16):
`r_corr=-weight*max(0,abs(u-u_realized)-tau)`, with defaults tau=0.03 and weight=1.
An outside-domain inverse is never clipped or called a valid action. Its discharge
change, outside-correction count and `scaled_release_correction=abs(Q_exec-Q_target)/proposal_width`
remain recorded. No outside-interval or scaled-magnitude penalty is introduced;
these cases carry zero Eq. (16) penalty and require separate interpretation.

Physically collapsed execution intervals are `forced_dispatch`. A collapsed
proposal is instead `degenerate_proposal`: it provides no Actor choice, but its
fixed target can still require a physical intervention. Effective `controls`
exclude both; `physical_controls` excludes only physical forcing. This separates
physical intervention frequency from Actor action-violation frequency. Physical
intervention counts include repairs to degenerate proposals, while normalized
rates/penalties exclude them. Numerical-only adjustments remain separate.

The external `correct_releases()` API now requires coherent original bounds when
normalized action metadata is provided; materially reversed bounds and targets
inconsistent with the original mapping are rejected. Physical-only external repair
still works without invented normalized-action metadata.

The PPO buffer continues to retain sampled indices and their original old log
probabilities. Candidate-action credit uses frozen dated proposal intervals;
batched candidate physics, evaluation, XLD diagnostics and checkpoint replay use
the same mapping and execution rules. No network, observation or PPO setting is
changed by selecting the new mode.

### Controlled configurations

Use the same remaining settings and seeds for all three:

| Configuration | CLI selection | Meaning |
| --- | --- | --- |
| A | `--release-mapping-mode joint_feasible --no-action-correction --no-action-correction-penalty` | Existing baseline |
| B | `--release-mapping-mode csv_proposal --action-correction --no-action-correction-penalty` | CSV target representation and physical projection |
| C | `--release-mapping-mode csv_proposal --action-correction --action-correction-penalty` | B plus valid-inverse Eq. (16) penalty |

CSV mode requires physical correction enabled and rejects contradictory settings.
`--release-bounds-file` selects a filename within the reservoir data directory.
A versus B changes the action representation/execution architecture. B versus C
isolates the correction penalty. Improved executed performance alone does not
establish improved learned proposals, especially when projection changes many
commands or outside-CSV executions receive no Eq. (16) penalty.

`action_correction_summary` includes per-reservoir and pooled physical intervention
rates, normalized above-tau rates, effective controllable and physical-nonforced
days, discharge correction means/maxima, scaled magnitudes, outside corrections,
proposal degeneracy/disjointness, physical forcing and infeasible attempts. These
are separate from ecological and guaranteed-output rates and physical generation.
Training trajectories and frozen replay export proposal/execution bounds, storage,
power, sampled actions and log probabilities. `examples/verify_csv_proposal.py`
provides short frozen-policy A/B/C comparisons, optional JSON summaries and optional
five-row-per-day CSV traces; no training or output files are produced by default.


CSV-mode focused verification (from the repository root):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests /opt/anaconda3/envs/harl/bin/python -B -m unittest \
  test_release_bounds test_csv_proposal_physics test_csv_correction_metrics test_csv_proposal_integration -q
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B examples/verify_csv_proposal.py \
  --days 365 --seed 1 --algorithms happo mappo
```

On 2026-11-10, the frozen seed-1 default-policy verification over 365-day dry
2022, median 1991, and wet 2020 windows completed A/B/C for both algorithms.
A had no genuine interventions. B and C matched physical trajectories exactly;
HAPPO had 4,443/5,475 genuine physical interventions (81.15%), MAPPO had
4,850/5,472 (88.63%; three physical forced events). These high rates describe
untrained proposals and the changed mapping architecture, not successful learning.
CSV/physical intervals were disjoint on 357 HAPPO and 238 MAPPO reservoir-days;
three of the latter were physically forced. Zero physical execution failures
occurred. Full summaries, first conflict examples and 32,850 daily reservoir
records are in `runs/csv_proposal_verification_20261110/`.

Configuration C training for both algorithms, displaying unsaved figures:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B examples/train_cascade_compare.py \
  --algorithms happo mappo --happo-credit-assignment counterfactual \
  --release-mapping-mode csv_proposal --action-correction --action-correction-penalty \
  --correction-tau 0.03 --correction-weight 1 \
  --episodes 300 --episode-days 1825 --parallel-envs 6 --seeds 1 2 3 --gamma 0.995 \
  --plot-python /Library/Frameworks/Python.framework/Versions/3.9/bin/python3
```

Replace the penalty flag with `--no-action-correction-penalty` for B. For A,
select `joint_feasible` and both `--no-action-correction` and
`--no-action-correction-penalty`. Keep the same estimator and other settings.
No `--episode-log` means no training result, checkpoint or figure files. Add the
existing logging/trajectory options only when persisted experiment evidence is
wanted. The separate saved verification folder above is not a training run.
