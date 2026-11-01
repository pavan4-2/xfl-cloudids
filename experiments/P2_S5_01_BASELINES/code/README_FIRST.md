# XFL-CloudIDS | P2-S5-01 | SYSTEM 5 ONLY

This package authorizes the first REAL development-data baseline campaign:
Local-only, FedAvg, FedProx using the *same* MLP 256-128 BatchNorm/ReLU/dropout
architecture and client-local bounded training-only SMOTE policy designated
for P2-S6-01 M1.1. CPU-only System 5. No additional R2 candidate search.

**Existing frozen R2 result:** `NO_CANDIDATE_PASSED` on both systems.
**Supervisor's separate implementation backbone:** B_MLP_SMOTE_CE.
This is NOT an R2 PASS, and the results of P2-S5-01 are NOT confirmatory.

## One-command execution on System 5 (Windows PowerShell)
From the extracted execution-package folder:

powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_p2_s5_01.ps1

After `P2_S5_01_TECHNICAL_PASS`:

powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\sync_p2_s5_01_github.ps1

Outputs from `Desktop\XFL_P2_S5_01\submission`:
- `P2_S5_01_RESULTS.zip`
- `P2_S5_01.log`

Report `GITHUB_SYNC_PASS SYSTEM5 P2-S5-01` and the printed Git commit ID.

Frozen pilot: alpha=0.3, partition-seed 17, training seed 11, five rounds,
two epochs/client/round, batch 512, AdamW lr .0005, WD .0001, dropout .3;
FedProx mu .01 (predeclared pilot engineering default, not retrospectively tuned).

Data guard: read TRAIN and VALIDATION arrays only. Withhold same train-only
probe rows as P2-S6-01 for fair training budgets. Use only original pre-SMOTE
client counts for weighted averaging; do not use VAL to choose weights.

Client-local SMOTE only; no cross-tenant synthesis; no singleton duplication.
No UTILITY/GATE/SEALED final test, no data uploads, no model checkpoints.
Output is diagnostic development evidence (not Q1 performance claim).

If path/dependency/check fails: preserve log, stop, do not alter code or rerun.
