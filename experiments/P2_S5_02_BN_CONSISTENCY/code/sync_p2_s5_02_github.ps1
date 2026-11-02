$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$ROOT=$PSScriptRoot
$REPO='C:\Users\SRMAOCSELAB2022-248\xfl-cloudids'
$WORK='C:\Users\SRMAOCSELAB2022-248\Desktop\XFL_P2_S5_02'
$REL='experiments/P2_S5_02_BN_CONSISTENCY'
$DEST=Join-Path $REPO $REL
if (!(Test-Path -LiteralPath (Join-Path $REPO '.git'))) { throw 'STOP: Git repository missing' }
if (!(Test-Path -LiteralPath (Join-Path $WORK 'results\summary.json'))) { throw 'STOP: no results; no Git sync' }
if (Test-Path -LiteralPath $DEST) { throw 'STOP: destination exists; do not overwrite or recommit' }
$REMOTE=(git -C $REPO remote get-url origin).Trim()
if ($REMOTE -notmatch 'pavan4-2/xfl-cloudids(?:\.git)?$') { throw 'STOP: unexpected remote' }
$BRANCH=(git -C $REPO branch --show-current).Trim()
if ($BRANCH -ne 'p1-b01-baseline-harness') { throw 'STOP: unexpected current branch' }
New-Item -ItemType Directory -Path (Join-Path $DEST 'code') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $DEST 'system5\results') -Force | Out-Null
$names=@('common.py','r2_backbone_selection.py','p2_s5_02_localbn.py','selftest.py','protocol.json','README_FIRST.md','run_p2_s5_02.ps1','sync_p2_s5_02_github.ps1')
foreach($n in $names) { Copy-Item -LiteralPath (Join-Path $ROOT $n) -Destination (Join-Path $DEST 'code') }
Copy-Item -Path (Join-Path $WORK 'results\*') -Destination (Join-Path $DEST 'system5\results')
Copy-Item -LiteralPath (Join-Path $WORK 'P2_S5_02.log') -Destination (Join-Path $DEST 'system5\P2_S5_02.log')
git -C $REPO add -- $REL
if ($LASTEXITCODE -ne 0) { throw 'STOP: git add failed' }
git -C $REPO commit -m 'Add System5 P2-S5-02 local BatchNorm development control'
if ($LASTEXITCODE -ne 0) { throw 'STOP: git commit failed; preserve worktree' }
git -C $REPO push origin HEAD
if ($LASTEXITCODE -ne 0) { throw 'STOP: push failed; preserve local commit, do not recommit' }
Write-Host 'GITHUB_SYNC_PASS SYSTEM5 P2-S5-02'
git -C $REPO rev-parse --short HEAD
