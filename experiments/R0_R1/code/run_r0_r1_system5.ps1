$ErrorActionPreference="Stop"
Set-StrictMode -Version Latest
$ROOT=$PSScriptRoot
$PYTHON="C:\Users\SRMAOCSELAB2022-248\xfl-cloudids\.venv\Scripts\python.exe"
$BRIDGE=Join-Path ([Environment]::GetFolderPath('Desktop')) "XFL_P1_08_A1_SYSTEM5\bridge"
$WORK=Join-Path ([Environment]::GetFolderPath('Desktop')) "XFL_R0_R1_SYSTEM5"
if(!(Test-Path -LiteralPath $PYTHON)){throw "STOP: approved System 5 Python environment not found at $PYTHON"}
if(!(Test-Path -LiteralPath $BRIDGE)){throw "STOP: existing governed System 5 development bridge not found at $BRIDGE. Do not rebuild it."}
if(Test-Path -LiteralPath $WORK){throw "STOP: R0/R1 System 5 workspace already exists. Preserve/report it; do not overwrite or rerun."}
New-Item -ItemType Directory -Path $WORK | Out-Null
New-Item -ItemType Directory -Path "$WORK\results\r0" -Force | Out-Null
New-Item -ItemType Directory -Path "$WORK\results\r1" -Force | Out-Null
$LOG=Join-Path $WORK "XFL_R0_R1_SYSTEM5.log"
$env:PYTHONPATH="";$env:OMP_NUM_THREADS="1";$env:MKL_NUM_THREADS="1"
Start-Transcript -Path $LOG
try {
  Write-Host "===== XFL-CLOUDIDS SOTA RECOVERY R0/R1 SYSTEM 5 ====="
  Write-Host "R0 is read-only; R1 uses development TRAIN and VALIDATION only; sealed final test is not authorized."
  & $PYTHON -u "$ROOT\selftest.py"
  if($LASTEXITCODE -ne 0){throw "STOP: package self-test failed."}
  & $PYTHON -u "$ROOT\r0_audit.py" --bridge $BRIDGE --out "$WORK\results\r0" --system SYSTEM5
  if($LASTEXITCODE -ne 0){throw "STOP: R0 audit failed. Preserve workspace/log and report; do not modify data."}
  & $PYTHON -u "$ROOT\r1_tree_ceiling.py" --bridge $BRIDGE --out "$WORK\results\r1" --system SYSTEM5
  if($LASTEXITCODE -ne 0){throw "STOP: R1 System 5 tree ceiling failed. Preserve workspace/log and report; do not tune/retry."}
  & $PYTHON -u "$ROOT\package_results.py" --work $WORK --system SYSTEM5
  if($LASTEXITCODE -ne 0){throw "STOP: result packaging failed."}
  Write-Host "===== R0/R1 SYSTEM 5 TECHNICAL PASS ====="
} finally {
  Stop-Transcript | Out-Null
  New-Item -ItemType Directory -Path "$WORK\submission" -Force | Out-Null
  Copy-Item -LiteralPath $LOG -Destination "$WORK\submission\XFL_R0_R1_SYSTEM5.log" -Force
}
Write-Host "Return exactly XFL_R0_R1_SYSTEM5_RESULTS.zip and XFL_R0_R1_SYSTEM5.log from $WORK\submission; then stop."
