# XFL-CloudIDS P1-B01: Baseline and Experiment Harness

## Purpose
P1-B01 provides the independent baseline/evaluation code path that will be merged with the System 6 proposed-method core during P1-02. It is actual implementation work, not a feasibility experiment.

## Implemented baseline primitives
- Local-only: receiver model is evaluated without collaboration.
- FedAvg: sample-count weighted parameter aggregation.
- FedProx: proximal local-objective regularizer around the current global model.
- SCAFFOLD: control-variate gradient correction primitive.
- Ditto: personalized-model proximal regularizer around the global model.
- FedFomo: positive receiver-validation improvement normalization for personalized peer weighting.

The later governed runner will supply publication-faithful local training/evaluation loops and freeze method-specific hyperparameters. P1-B01 intentionally avoids using F0-R2 screening approximations as final publication baselines.

## Common evaluation contract
All methods must export records keyed by method, scenario, seed, round, receiver and split. At minimum, each record contains accuracy, macro-F1, false-alarm rate and worst-class recall. The schema is shared with the proposed method so final tables can be generated without method-specific post-processing.

## Repository boundary
P1-B01 owns only:
- `src/xfl_cloudids/baselines/`
- `src/xfl_cloudids/evaluation/`
- `configs/p1_b01_baseline_harness.json`
- `docs/P1_B01_BASELINE_HARNESS_SPEC.md`
- `scripts/p1_b01_selftest.py`
- `tests/test_p1_b01_baseline_harness.py`

It must not edit `src/xfl_cloudids/method/`, which is owned by System 6 P1-01.
