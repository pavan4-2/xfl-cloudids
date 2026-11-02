$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$ROOT=$PSScriptRoot
$PYTHON='C:\Users\SRMAOCSELAB2022-248\xfl-cloudids\.venv\Scripts\python.exe'
$BRIDGE='C:\Users\SRMAOCSELAB2022-248\Desktop\XFL_P1_08_A1_SYSTEM5\bridge'
$WORK='C:\Users\SRMAOCSELAB2022-248\Desktop\XFL_P2_S5_02'
if (!(Test-Path -LiteralPath $PYTHON)) { throw 'STOP: approved Python environment missing' }
if (!(Test-Path -LiteralPath $BRIDGE)) { throw 'STOP: approved development bridge missing' }
if (Test-Path -LiteralPath $WORK) { throw 'STOP: workspace exists. Preserve original; no rerun' }
New-Item -ItemType Directory -Path $WORK | Out-Null
$LOG=Join-Path $WORK 'P2_S5_02.log'
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:PYTHONPATH=''
Start-Transcript -LiteralPath $LOG | Out-Null
$OK=$false
try {
 Write-Host '===== XFL-CLOUDIDS P2-S5-02: BATCHNORM CONSISTENCY DEVELOPMENT ====='
 Write-Host 'Only TRAIN and VALIDATION; P2-S5-01 preserved; no gate/utility/test.'
 & $PYTHON -u (Join-Path $ROOT 'selftest.py')
 if ($LASTEXITCODE -ne 0) { throw 'STOP: new package selftest failed' }
 & $PYTHON -u (Join-Path $ROOT 'p2_s5_02_localbn.py') --bridge $BRIDGE --out $WORK --protocol (Join-Path $ROOT 'protocol.json')
 if ($LASTEXITCODE -ne 0) { throw 'STOP: experiment failed; preserve all evidence' }
 if (!(Test-Path -LiteralPath (Join-Path $WORK 'P2_S5_02_RESULTS.zip'))) { throw 'STOP: results ZIP missing' }
 $OK=$true
 Write-Host 'P2_S5_02_TECHNICAL_PASS'
} finally {
 Stop-Transcript | Out-Null
 $OUT=Join-Path $WORK 'submission'
 if (!(Test-Path -LiteralPath $OUT)) { New-Item -ItemType Directory -Path $OUT | Out-Null }
 if (Test-Path -LiteralPath $LOG) { Copy-Item -LiteralPath $LOG -Destination (Join-Path $OUT 'P2_S5_02.log') -Force }
 if ($OK) { Copy-Item -LiteralPath (Join-Path $WORK 'P2_S5_02_RESULTS.zip') -Destination (Join-Path $OUT 'P2_S5_02_RESULTS.zip') -Force }
}
Write-Host "Submit the two files in $WORK\submission, then stop."
