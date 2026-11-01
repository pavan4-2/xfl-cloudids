$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$ROOT = $PSScriptRoot
$REPO = 'C:\Users\SRMAOCSELAB2022-248\xfl-cloudids'
$PYTHON = Join-Path $REPO '.venv\Scripts\python.exe'
$DESKTOP = [Environment]::GetFolderPath('Desktop')
$BRIDGE = Join-Path $DESKTOP 'XFL_P1_08_A1_SYSTEM5\bridge'
$WORK = Join-Path $DESKTOP 'XFL_P2_S5_01'
if (!(Test-Path -LiteralPath $PYTHON)) { throw 'STOP: approved System 5 Python environment missing; do not install/switch environments' }
if (!(Test-Path -LiteralPath $BRIDGE)) { throw 'STOP: approved System 5 development bridge missing; do not rebuild it' }
if (Test-Path -LiteralPath $WORK) { throw 'STOP: work directory already exists; preserve it; do not overwrite or rerun' }
New-Item -ItemType Directory -Path $WORK -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $WORK 'submission') -Force | Out-Null
$LOG = Join-Path $WORK 'P2_S5_01.log'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:PYTHONPATH=''
Start-Transcript -LiteralPath $LOG | Out-Null
$SUCCEEDED = $false
try {
 Write-Host '===== XFL-CLOUDIDS P2-S5-01: MATCHED FEDERATED BASELINES ====='
 Write-Host 'TRAIN and VALIDATION ONLY; utility/gate/sealed final test are not authorized.'
 & $PYTHON -u (Join-Path $ROOT 'selftest.py')
 if ($LASTEXITCODE -ne 0) { throw 'STOP: package self-test failed. Preserve log and report.' }
 & $PYTHON -u (Join-Path $ROOT 'p2_s5_01_baselines.py') --bridge $BRIDGE --out $WORK --protocol (Join-Path $ROOT 'protocol.json')
 if ($LASTEXITCODE -ne 0) { throw 'STOP: baseline campaign failed. Preserve log and report. No rerun.' }
 if (!(Test-Path -LiteralPath (Join-Path $WORK 'P2_S5_01_RESULTS.zip'))) { throw 'STOP: results archive missing' }
 $SUCCEEDED = $true
 Write-Host '===== P2_S5_01_TECHNICAL_PASS ====='
} finally {
 Stop-Transcript | Out-Null
 if (Test-Path -LiteralPath $LOG) {
  Copy-Item -LiteralPath $LOG -Destination (Join-Path $WORK 'submission\P2_S5_01.log') -Force
 }
 if ($SUCCEEDED) {
  Copy-Item -LiteralPath (Join-Path $WORK 'P2_S5_01_RESULTS.zip') -Destination (Join-Path $WORK 'submission\P2_S5_01_RESULTS.zip') -Force
 }
}
Write-Host "Submit P2_S5_01_RESULTS.zip and P2_S5_01.log from $WORK\submission. After technical pass, run the supplied GitHub sync once."
