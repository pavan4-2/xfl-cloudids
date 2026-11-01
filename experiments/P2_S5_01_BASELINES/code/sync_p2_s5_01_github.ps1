$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$ROOT=$PSScriptRoot
$REPO='C:\Users\SRMAOCSELAB2022-248\xfl-cloudids'
$WORK=Join-Path ([Environment]::GetFolderPath('Desktop')) 'XFL_P2_S5_01'
$REL='experiments/P2_S5_01_BASELINES'
$DEST=Join-Path $REPO $REL
if (!(Test-Path -LiteralPath (Join-Path $REPO '.git'))) {throw 'STOP: governed System5 git repository missing'}
if (!(Test-Path -LiteralPath (Join-Path $WORK 'results\summary.json'))) {throw 'STOP: complete scientific results not found; no Git sync'}
if (Test-Path -LiteralPath $DEST) {throw 'STOP: GitHub destination already exists; do not overwrite'}
$REMOTE=(git -C $REPO remote get-url origin).Trim()
if ($REMOTE -notmatch 'pavan4-2/xfl-cloudids(?:\.git)?$') {throw 'STOP: unexpected git remote'}
New-Item -ItemType Directory -Path (Join-Path $DEST 'code') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $DEST 'system5\results') -Force | Out-Null
$files=@('common.py','r2_backbone_selection.py','p2_s5_01_baselines.py','selftest.py','protocol.json','README_FIRST.md','run_p2_s5_01.ps1','sync_p2_s5_01_github.ps1')
foreach($f in $files){Copy-Item -LiteralPath (Join-Path $ROOT $f) -Destination (Join-Path $DEST 'code')}
Copy-Item -Path (Join-Path $WORK 'results\*') -Destination (Join-Path $DEST 'system5\results')
Copy-Item -LiteralPath (Join-Path $WORK 'P2_S5_01.log') -Destination (Join-Path $DEST 'system5\P2_S5_01.log')
git -C $REPO add -- $REL
if ($LASTEXITCODE -ne 0) {throw 'STOP: git add failed'}
git -C $REPO commit -m 'Add System5 P2-S5-01 architecture-matched FL development baselines'
if ($LASTEXITCODE -ne 0) {throw 'STOP: git commit failed; preserve workspace'}
git -C $REPO push origin HEAD
if ($LASTEXITCODE -ne 0) {throw 'STOP: git push failed; preserve workspace'}
Write-Host 'GITHUB_SYNC_PASS SYSTEM5 P2-S5-01'
git -C $REPO rev-parse --short HEAD
