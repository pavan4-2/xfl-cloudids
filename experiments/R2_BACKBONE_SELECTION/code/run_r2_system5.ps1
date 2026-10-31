$ErrorActionPreference="Stop"
Set-StrictMode -Version Latest
$ROOT=$PSScriptRoot
$PYTHON="C:\Users\SRMAOCSELAB2022-248\xfl-cloudids\.venv\Scripts\python.exe"
$BRIDGE=Join-Path ([Environment]::GetFolderPath('Desktop')) "XFL_P1_08_A1_SYSTEM5\bridge"
$WORK=Join-Path ([Environment]::GetFolderPath('Desktop')) "XFL_R2_SYSTEM5"
if(!(Test-Path -LiteralPath $PYTHON)){throw "STOP: approved System 5 Python environment not found at $PYTHON"}
if(!(Test-Path -LiteralPath $BRIDGE)){throw "STOP: existing governed System 5 development bridge not found at $BRIDGE. Do not rebuild it."}
if(Test-Path -LiteralPath $WORK){throw "STOP: R2 System 5 workspace already exists. Preserve/report it; do not overwrite or rerun."}
New-Item -ItemType Directory -Path "$WORK\results\r2" -Force | Out-Null
New-Item -ItemType Directory -Path "$WORK\submission" -Force | Out-Null
$LOG=Join-Path $WORK "XFL_R2_SYSTEM5.log"
$env:PYTHONPATH="";$env:OMP_NUM_THREADS="1";$env:MKL_NUM_THREADS="1"
Start-Transcript -Path $LOG
try {
  Write-Host "===== XFL-CLOUDIDS R2-S5 BOUNDED BACKBONE/IMBALANCE SELECTION ====="
  Write-Host "Development TRAIN/VALIDATION only. Utility, gate and sealed final test are not authorized."
  & $PYTHON -u "$ROOT\selftest.py"
  if($LASTEXITCODE -ne 0){throw "STOP: package self-test failed."}
  & $PYTHON -u "$ROOT\r2_backbone_selection.py" --bridge $BRIDGE --out "$WORK\results\r2" --system SYSTEM5
  if($LASTEXITCODE -ne 0){throw "STOP: R2 execution failed. Preserve workspace/log and report; do not edit or rerun."}
  & $PYTHON -u "$ROOT\package_results.py" --work $WORK --system SYSTEM5
  if($LASTEXITCODE -ne 0){throw "STOP: result packaging failed."}
  Write-Host "===== R2 SYSTEM 5 TECHNICAL PASS ====="
  Write-Host "Read R2_SELECTION.json only to identify the machine-generated scientific status. Do not alter or rerun."
} finally {
  Stop-Transcript | Out-Null
  if(Test-Path -LiteralPath $LOG){Copy-Item -LiteralPath $LOG -Destination "$WORK\submission\XFL_R2_SYSTEM5.log" -Force}
}
Write-Host "Return exactly XFL_R2_SYSTEM5_RESULTS.zip and XFL_R2_SYSTEM5.log from $WORK\submission; then run the authorized GitHub sync and stop."
