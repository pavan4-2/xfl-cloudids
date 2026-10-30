# XFL-CloudIDS R0/R1 SOTA Recovery - Read First

## Scope
R0/R1 is the first SOTA-recovery step. The execution environment and supplied files are treated as trusted; **no checksum/SHA verification is required**.

- **R0:** read-only development benchmark audit. It records manifest structure, class support, client/role sizes, exact duplicate overlap, and leakage-relevant structure.
- **R1 System 5:** centralized HistGradientBoosting performance ceiling.
- **R1 System 6:** centralized modern 256-128 BatchNorm/dropout MLP ceiling on GPU.
- The sealed/final test remains unopened.
- Calibration seed: 11 only.
- Do not change parameters, labels, partitions, model code, or paths.

## Execute
System 5 runs only:
```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_r0_r1_system5.ps1
```

System 6 runs only:
```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_r0_r1_system6.ps1
```

## GitHub reproducibility
After a successful experiment, run the assigned GitHub sync script. It copies only code/config/results/logs into the existing `xfl-cloudids` repository and pushes them. It never copies datasets or checkpoints.

System 5:
```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\sync_r0_r1_system5_github.ps1
```
System 6:
```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\sync_r0_r1_system6_github.ps1
```

If GitHub push fails because authentication/network access is unavailable, **do not rerun the experiment**. Preserve the experiment and report the Git error separately.

## Return
Return exactly two files from the submission folder:
- `XFL_R0_R1_SYSTEM5_RESULTS.zip` and `XFL_R0_R1_SYSTEM5.log`, or
- `XFL_R0_R1_SYSTEM6_RESULTS.zip` and `XFL_R0_R1_SYSTEM6.log`.

Then stop and wait for written authorization.
