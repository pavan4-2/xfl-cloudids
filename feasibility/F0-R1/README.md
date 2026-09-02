# XFL-CloudIDS F0-R1 Execution Package

**Status:** reconstructed execution package for Amendment **F0-R1-A1**.

This folder supplies the scripts named by the *XFL-CloudIDS F0-R1 Student Execution Manual*:

- `R1_1_prepare_corrected_partitions.py`
- `R1_2_aligned_negative_transfer.py`
- `R1_3_transfer_utility.py`
- `R1_4_reliability_grid.py`
- `R1_5_multiround_oracle_gate.py`
- `R1_6_compile_decision.py`
- `verify_system5_reproduction.py`

The original execution ZIP referenced by the manual was not available. Therefore this is a **reconstructed reference implementation**, not a byte-for-byte recovery of the missing original package. The protocol-facing command line, artifact names, frozen decision thresholds, seeds, scenarios, model shape, gate criteria, and supervisor package flow follow the manual. Values not stated by the manual are explicitly listed in `RECONSTRUCTION_NOTES.md`.

## Required real-run environment

Use the environment frozen by the F0-R1 manual before research execution:

- Python 3.11.14
- PyTorch 2.13.0+cu132
- CUDA available
- NVIDIA GeForce RTX 3060
- real CIC-IDS2017 CSV files outside Git

`requirements-f0-r1.txt` records the direct Python dependencies used by the scripts. For the controlled run, preserve the actual resolved environment with:

```bash
python -m pip freeze > evidence/requirements-f0-r1-lock.txt
```

## Pre-run verification

From this folder:

```bash
python -m py_compile \
  f0r1_core.py f0r1_rutra.py \
  R1_1_prepare_corrected_partitions.py \
  R1_2_aligned_negative_transfer.py \
  R1_3_transfer_utility.py \
  R1_4_reliability_grid.py \
  R1_5_multiround_oracle_gate.py \
  R1_6_compile_decision.py \
  verify_system5_reproduction.py
```

Optional non-research smoke test:

```bash
bash run_synthetic_smoke.sh
```

The smoke path exists only to verify that the code path executes. **Never submit smoke metrics as scientific evidence and never run R1.6 on smoke data.**

## Real execution

Follow the F0-R1 Student Execution Manual exactly. In summary, System 6 executes R1.1 through R1.5 once using the frozen configuration and preserves all logs. System 5 independently repeats R1.1 and runs the reproduction verifier. After the evidence is copied back, System 6 runs R1.6 with declarations that are true.

The scripts refuse selected real-data parameter overrides that would violate the frozen protocol.

## Sealed-test rule

R1.1 creates `untouched_final_test_R1.npz` and a seal JSON. The R1.2-R1.5 feasibility decision path uses development/utility/gate material and **does not load the sealed final test**. Do not inspect, transmit, or use that NPZ during F0-R1.

## Supervisor authority

An automated `GREEN`, `CONDITIONAL`, or `RED` from R1.6 is only a technical recommendation. **Only the supervisor's signed decision authorizes Phase 1.**
