# XFL-CloudIDS Phase F0 Student Execution Package

Purpose: run the six feasibility tests in order and submit reproducible evidence to the supervisor before the full implementation begins.

This package follows the frozen XFL-CloudIDS manual. F0 uses only CIC-IDS2017 training/development data and seeds 17 and 29. The final test partition created by F0.1 must remain unopened.

## Responsibility allocation

- Student 1: independently checks F0.1 tenant counts, F0.3 utility matrices, file hashes and second-machine reproduction.
- Student 2: executes the main pipeline, model training, F0.2 baselines, F0.4 reliability and F0.5 prototype.
- Both students: preserve commands, terminal output, failures and environment lock.
- Supervisor: verifies F0.6 evidence and makes the final GREEN/CONDITIONAL/RED decision.

Run the main execution on System 6. Reproduce the reduced verification run or supervisor-selected tests on System 5. Both environments must remain Python 3.11.14, PyTorch 2.13.0+cu132 and CUDA 13.2.

## Before starting

Place this folder inside the governed `xfl-cloudids` repository. Do not put the raw dataset inside GitHub.

Activate the verified environment:

```bash
source .venv/bin/activate
```

Move into this package:

```bash
cd XFL_CloudIDS_F0_Tests
```

Confirm the environment and preserve it:

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pip freeze > requirements-f0-lock.txt
```

Your CIC-IDS2017 directory must contain the original CSV files. It may be outside the repository, for example `/data/CICIDS2017/CSVs`.

## Execute the tests in this exact order

### F0.1 — prepare data and tenants

```bash
python F0_1_prepare_data_and_tenants.py \
  --raw-dir /absolute/path/to/CICIDS2017/CSVs \
  2>&1 | tee artifacts_F0_1_terminal.log
```

This reads the raw CSV files, removes direct identifiers, performs train-only scaling, creates five non-IID tenants for both seeds and seals the final-test file.

Check that the last terminal JSON contains `"status": "PASS"` and that `artifacts/f0/f01/F0_1_PASS.json` exists.

### F0.2 — baselines and negative transfer

```bash
python F0_2_baselines_and_negative_transfer.py \
  2>&1 | tee artifacts_F0_2_terminal.log
```

This trains Local-only, Centralized, FedAvg and FedProx with identical settings. It may take several hours depending on dataset size. Do not interrupt merely because the terminal is quiet during training.

Check `artifacts/f0/f02/negative_transfer_summary.json` and preserve `metrics.csv`.

### F0.3 — recipient–contributor utility matrix

```bash
python F0_3_recipient_transfer_matrix.py \
  2>&1 | tee artifacts_F0_3_terminal.log
```

This measures whether each contributor helps or harms each receiver by attack class, temporal window and round.

Check `artifacts/f0/f03/matrix_summary.json` and preserve `utility.csv`.

### F0.4 — forward reliability prediction

```bash
python F0_4_reliability_prediction.py \
  2>&1 | tee artifacts_F0_4_terminal.log
```

This uses only prior-round utility evidence to predict next-round harm. It reports ROC-AUC, Brier score, Spearman correlation and bootstrap intervals.

Check `artifacts/f0/f04/reliability_summary.json`.

### F0.5 — oracle and minimal RUTRA-Gate

```bash
python F0_5_oracle_and_minimal_rutra_gate.py \
  2>&1 | tee artifacts_F0_5_terminal.log
```

This estimates maximum available headroom, solves the risk-constrained weights, applies the held-out gate and measures harmful-update rejection and beneficial-update acceptance.

Check `artifacts/f0/f05/oracle_and_prototype_summary.json`, `weights.csv`, `gate.csv` and `metrics.csv`.

### F0.6 — evidence report

```bash
python F0_6_compile_gate_report.py \
  2>&1 | tee artifacts_F0_6_terminal.log
```

This checks completeness, confirms that the sealed test hash is unchanged, applies the pre-declared decision aids and creates the evidence ZIP.

Submit these to the supervisor:

- `artifacts/f0/f06/F0_RESULTS_FOR_SUPERVISOR.zip`
- all six `artifacts_F0_*_terminal.log` files;
- `requirements-f0-lock.txt`;
- the Git commit hash used for execution;
- any unsuccessful-run logs not already present in the evidence ZIP.

## Reduced code-verification run

The following is only for checking that the scripts execute. Synthetic output is never research evidence and cannot support Gate F0.

```bash
python F0_1_prepare_data_and_tenants.py --synthetic-smoke --artifact-root smoke_artifacts/f0
python F0_2_baselines_and_negative_transfer.py --artifact-root smoke_artifacts/f0 --rounds 2 --local-epochs 1
python F0_3_recipient_transfer_matrix.py --artifact-root smoke_artifacts/f0 --rounds 3 --local-epochs 1
python F0_4_reliability_prediction.py --artifact-root smoke_artifacts/f0 --bootstrap 100
python F0_5_oracle_and_minimal_rutra_gate.py --artifact-root smoke_artifacts/f0 --gate-bootstrap 100
python F0_6_compile_gate_report.py --artifact-root smoke_artifacts/f0
```

## Rules that must not be violated

1. Never open, evaluate or inspect `untouched_final_test.npz` during F0.
2. Never change seeds or thresholds after seeing results.
3. Never delete failed or unfavourable runs.
4. Never use synthetic smoke results for the Gate F0 decision.
5. Do not claim the screening FedFomo/pFedGraph mechanisms as final faithful publication baselines.
6. A script completing is not the same as a favourable F0 result.
7. Only a supervisor-signed GREEN report authorizes Phases 1–10.

## Expected file sequence

| Test | Required completion marker |
|---|---|
| F0.1 | `artifacts/f0/f01/F0_1_PASS.json` |
| F0.2 | `artifacts/f0/f02/F0_2_PASS.json` |
| F0.3 | `artifacts/f0/f03/F0_3_PASS.json` |
| F0.4 | `artifacts/f0/f04/F0_4_PASS.json` |
| F0.5 | `artifacts/f0/f05/F0_5_PASS.json` |
| F0.6 | `artifacts/f0/f06/F0_6_PASS.json` |

If any test stops or raises an error, preserve the terminal log and contact the supervisor before changing code, data or configuration.
