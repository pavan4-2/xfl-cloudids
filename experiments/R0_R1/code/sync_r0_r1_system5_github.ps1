$ErrorActionPreference="Stop"
Set-StrictMode -Version Latest
$ROOT=$PSScriptRoot
$REPO="C:\Users\SRMAOCSELAB2022-248\xfl-cloudids"
$WORK="C:\Users\SRMAOCSELAB2022-248\Desktop\XFL_R0_R1_SYSTEM5"
$DEST=Join-Path $REPO "experiments\R0_R1"
$SYSDEST=Join-Path $DEST "system5"
if(!(Test-Path -LiteralPath (Join-Path $REPO ".git"))){throw "STOP: xfl-cloudids Git repository not found at $REPO"}
if(!(Test-Path -LiteralPath (Join-Path $WORK "results\r0\R0_SUMMARY.json"))){throw "STOP: completed R0/R1 results not found. Run the experiment first."}
$remote=(git -C $REPO remote get-url origin).Trim()
Write-Host "Git remote: $remote"
if($remote -notmatch "pavan4-2/xfl-cloudids(?:\.git)?$"){throw "STOP: unexpected Git remote. Expected pavan4-2/xfl-cloudids. Do not push to an unrelated repository."}
New-Item -ItemType Directory -Path (Join-Path $DEST "code") -Force | Out-Null
New-Item -ItemType Directory -Path $SYSDEST -Force | Out-Null
$codeFiles=@("CLASS_LABELS.json","RECOVERY_SPECIFICATION.json","REPRODUCIBILITY_README.md","FIRST_READ_GUIDE.md","common.py","r0_audit.py","r1_common.py","r1_tree_ceiling.py","r1_mlp_ceiling.py","package_results.py","selftest.py","run_r0_r1_system5.ps1","sync_r0_r1_system5_github.ps1")
foreach($f in $codeFiles){if(Test-Path -LiteralPath (Join-Path $ROOT $f)){Copy-Item -LiteralPath (Join-Path $ROOT $f) -Destination (Join-Path $DEST "code\$f") -Force}}
if(Test-Path -LiteralPath $SYSDEST){Get-ChildItem -LiteralPath $SYSDEST -Force | Remove-Item -Recurse -Force}
New-Item -ItemType Directory -Path $SYSDEST -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $WORK "results") -Destination (Join-Path $SYSDEST "results") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $WORK "XFL_R0_R1_SYSTEM5.log") -Destination (Join-Path $SYSDEST "XFL_R0_R1_SYSTEM5.log") -Force
$runRecord=@"
# XFL-CloudIDS R0/R1 System5 run record

- Experiment: R0/R1 SOTA recovery
- System: SYSTEM5
- Scope: development only; sealed final test not accessed
- Calibration seed: 11
- Code/config: supervisor-supplied package v1.1
- Result files: results/r0 and results/r1
- Terminal log: XFL_R0_R1_SYSTEM5.log
- Environment rule: trusted; no checksum governance
"@
Set-Content -LiteralPath (Join-Path $SYSDEST "RUN_RECORD.md") -Value $runRecord -Encoding UTF8
git -C $REPO add -- "experiments/R0_R1"
$staged=(git -C $REPO diff --cached --name-only)
if([string]::IsNullOrWhiteSpace(($staged -join ""))){Write-Host "GitHub reproducibility export already committed; no new changes."; exit 0}
git -C $REPO commit -m "Add R0/R1 SOTA recovery evidence from SYSTEM5"
if($LASTEXITCODE -ne 0){throw "STOP: Git commit failed. Experiment remains valid; report the Git error."}
git -C $REPO push origin HEAD
if($LASTEXITCODE -ne 0){throw "STOP: Git push failed. Experiment remains valid; report the Git error without rerunning."}
Write-Host "GITHUB_SYNC_PASS SYSTEM5: code, configuration, result tables/JSON and log pushed to pavan4-2/xfl-cloudids"
