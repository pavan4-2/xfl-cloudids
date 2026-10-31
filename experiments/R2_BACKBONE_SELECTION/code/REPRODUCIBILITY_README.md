# XFL-CloudIDS R2-S5 — Bounded Neural Backbone / Imbalance Selection

This package is a **development-only, single-seed selection stage**. It exists because the centralized tree ceiling passed strongly while the first neural ceiling did not provide adequate minority-class performance.

## Frozen scope
- System: SYSTEM5
- Existing bridge only: `Desktop\XFL_P1_08_A1_SYSTEM5\bridge`
- Canonical manifest: `manifest_alpha_0p3_seed17.json`
- Calibration seed: 11
- Uses pooled TRAIN for fitting and pooled VALIDATION for reporting.
- Primary reporting removes exact validation `X+y` signatures that occur in TRAIN, matching R1.
- Utility, gate, sealed/final-test data are forbidden.
- No parameter search or post-result edits.

## Candidate set
A. MLP + ordinary cross-entropy control.
B. Same MLP + bounded training-only SMOTE + ordinary cross-entropy.
C. Same MLP + one predeclared focal-loss alternative.
D. 1D-CNN + bounded training-only SMOTE + ordinary cross-entropy.

The code selects a winner only if every fixed gate in `R2_SPECIFICATION.json` passes. Otherwise it emits `NO_CANDIDATE_PASSED`. Either outcome is valid evidence.

## Exact System 5 command
```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_r2_system5.ps1
```

After technical completion:
```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\sync_r2_system5_github.ps1
```

## Required return
- `XFL_R2_SYSTEM5_RESULTS.zip`
- `XFL_R2_SYSTEM5.log`
- State whether `GITHUB_SYNC_PASS SYSTEM5` was printed.

Do not submit datasets, NPZ files, model checkpoints, `.pt/.pth/.ckpt` files, or any sealed/final-test artifact.
