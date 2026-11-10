# Saved-policy dispatch investigation

This analysis evaluates the final saved actors without training or changing the
baseline trajectory. It is kept outside `harl/` and `examples/` so adding an
investigation does not invalidate the implementation fingerprints in checkpoints.

Run from the repository root, using a **new** output directory:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/happo_dispatch_mplconfig \
  /opt/anaconda3/envs/harl/bin/python -B analysis/dispatch_diagnostics/analyze_run.py \
  --run-log runs/happo_seeds123_20261103_154615.jsonl \
  --output-dir runs/NEW_dispatch_analysis \
  --report-python /Library/Frameworks/Python.framework/Versions/3.9/bin/python3
```

The original three training scenarios and the final held-out sequence are
replayed for every saved seed. All configured discrete actions are evaluated for
each violated reservoir/service, alongside exact piecewise continuous power
maxima. Witnesses are checked through the complete cascade physics. Replayed
energy, returns and per-reservoir violation counts must reproduce the original
run log before the report can complete.

The Python interpreter used for plotting can differ from the inference
interpreter. No Matplotlib installation is required in the inference environment
when `--report-python` points to an interpreter that already provides it.

## Original experiment configuration

The investigated run used service multipliers **1.0 / 1.0**. The working YAML was
subsequently changed to **0.4 / 0.4**. The completed investigation therefore uses
the exact source snapshot in
`runs/happo_seeds123_20261103_154615_dispatch_reference_source/`: all 55 original
implementation/configuration fingerprints are checked against the saved
checkpoint. The sole differing source file, the environment YAML, was recovered
from the matching Git HEAD bytes in that isolated copy. The working YAML is
preserved, and no source-integrity check is relaxed.

To reproduce that original experiment's diagnosis while retaining the newer
working YAML, replace the analysis script path above with:

```text
runs/happo_seeds123_20261103_154615_dispatch_reference_source/analysis/dispatch_diagnostics/analyze_run.py
```

The source snapshot uses the checkpoint's original dataset location. The
checkpoint loader checks all dataset fingerprints as usual.

## Outputs

- One `seedN_SCENARIO_dispatch.csv` per replay, with five reservoir records per
  day: physical state, sampled commands, frozen critic values, realized returns,
  service classifications and independently verified alternative actions.
- `manifest.json`: checkpoint, source, dataset and analysis fingerprints, original
  evaluation reconciliation, and scenario accounting.
- `summary.json`: complete seed/scenario/reservoir/service, calendar-month and
  source-episode-year aggregates; separate discounted/undiscounted accounting.
- `report.md` and three PNGs: frequency/severity, classification composition and
  seasonal violation rates. Fixed training scenarios and held-out results are
  kept separate.

## Interpretation

`same_day_policy_alternative` means a different available command can meet that
one service **from the saved starting state**, with other commands held fixed.
It does not establish a better five-year strategy. Additional release, storage
changes and changes to other stations' service counts are recorded explicitly.

`discrete_grid_gap` means only a continuous command can meet the requirement.
Capacity/water/dispatch/head classifications describe restrictions in that same
state and inflow condition. They do not prove that earlier coordinated decisions
could not have produced a better state. Proof flags overlap; classification
labels follow an ordered decision rule and are not independent causal factors.

Alternatives for separate days or reservoirs cannot be combined directly: every
counterfactual starts from the original baseline state. Mean release/storage
costs describe separate witnesses, not an executable cumulative intervention.

Violation frequency, relative deficit severity, generation and service integrals
are undiscounted. Reward accounting also retains the experiment's gamma. The
global, system and legacy-team scalars repeat across five daily CSV records;
the report counts each once per day. Mixed rewards remain reservoir-specific.
Saved critic values are estimated mixed returns, not derivatives of water value.
