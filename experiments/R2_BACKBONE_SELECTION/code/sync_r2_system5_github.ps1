$ErrorActionPreference="Stop"
Set-StrictMode -Version Latest
$ROOT=$PSScriptRoot
$REPO="C:\Users\SRMAOCSELAB2022-248\xfl-cloudids"
$WORK="C:\Users\SRMAOCSELAB2022-248\Desktop\XFL_R2_SYSTEM5"
$DEST=Join-Path $REPO "experiments\R2_BACKBONE_SELECTION"
$SYSDEST=Join-Path $DEST "system5"
if(!(Test-Path -LiteralPath (Join-Path $REPO ".git"))){throw "STOP: xfl-cloudids Git repository not found at $REPO"}
if(!(Test-Path -LiteralPath (Join-Path $WORK "results\r2\R2_SYSTEM5_SUMMARY.json"))){throw "STOP: completed R2 results not found. Do not run Git sync before the experiment."}
$remote=(git -C $REPO remote get-url origin).Trim()
Write-Host "Git remote: $remote"
if($remote -notmatch "pavan4-2/xfl-cloudids(?:\.git)?$"){throw "STOP: unexpected Git remote. Expected pavan4-2/xfl-cloudids. Do not push to an unrelated repository."}
New-Item -ItemType Directory -Path (Join-Path $DEST "code") -Force | Out-Null
if(Test-Path -LiteralPath $SYSDEST){Get-ChildItem -LiteralPath $SYSDEST -Force | Remove-Item -Recurse -Force}
New-Item -ItemType Directory -Path $SYSDEST -Force | Out-Null
$codeFiles=@("CLASS_LABELS.json","R2_SPECIFICATION.json","REFERENCE_R1_SYSTEM5.json","REPRODUCIBILITY_README.md","FIRST_READ_GUIDE.md","common.py","r2_backbone_selection.py","package_results.py","selftest.py","run_r2_system5.ps1","sync_r2_system5_github.ps1")
foreach($f in $codeFiles){if(Test-Path -LiteralPath (Join-Path $ROOT $f)){Copy-Item -LiteralPath (Join-Path $ROOT $f) -Destination (Join-Path $DEST "code\$f") -Force}}
Copy-Item -LiteralPath (Join-Path $WORK "results\r2") -Destination (Join-Path $SYSDEST "results") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $WORK "XFL_R2_SYSTEM5.log") -Destination (Join-Path $SYSDEST "XFL_R2_SYSTEM5.log") -Force
$runRecord=@"
# XFL-CloudIDS R2-S5 run record

- Stage: bounded neural backbone/imbalance selection
- System: SYSTEM5
- Scope: development TRAIN/VALIDATION only; utility/gate/sealed final test not accessed
- Calibration seed: 11
- Canonical manifest: manifest_alpha_0p3_seed17.json
- Candidate set and thresholds: frozen in R2_SPECIFICATION.json
- Student tuning/manual selection: forbidden
- Terminal log: XFL_R2_SYSTEM5.log
- Trusted environment: no checksum governance
"@
Set-Content -LiteralPath (Join-Path $SYSDEST "RUN_RECORD.md") -Value $runRecord -Encoding UTF8
git -C $REPO add -- "experiments/R2_BACKBONE_SELECTION"
$staged=(git -C $REPO diff --cached --name-only)
if([string]::IsNullOrWhiteSpace(($staged -join ""))){Write-Host "GitHub reproducibility export already committed; no new changes."; Write-Host "GITHUB_SYNC_PASS SYSTEM5"; exit 0}
git -C $REPO commit -m "Add R2 backbone selection evidence from SYSTEM5"
if($LASTEXITCODE -ne 0){throw "STOP: Git commit failed. Experiment remains valid; report Git error without rerunning."}
git -C $REPO push origin HEAD
if($LASTEXITCODE -ne 0){throw "STOP: Git push failed. Experiment remains valid; report Git error without rerunning."}
Write-Host "GITHUB_SYNC_PASS SYSTEM5: R2 code/config/results/log pushed to pavan4-2/xfl-cloudids"
