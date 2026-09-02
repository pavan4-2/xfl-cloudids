#!/usr/bin/env bash
set -euo pipefail
python -m py_compile \
  f0r1_core.py f0r1_rutra.py \
  R1_1_prepare_corrected_partitions.py \
  R1_2_aligned_negative_transfer.py \
  R1_3_transfer_utility.py \
  R1_4_reliability_grid.py \
  R1_5_multiround_oracle_gate.py \
  R1_6_compile_decision.py \
  verify_system5_reproduction.py
python - <<'PY'
import json
from pathlib import Path
cfg=json.loads(Path('f0_config.json').read_text())
assert cfg['amendment']=='F0-R1-A1'
assert cfg['seeds']==[17,29]
assert [x['alpha'] for x in cfg['scenarios']]==[0.3,0.1]
assert cfg['training']['rounds']==10
assert cfg['decision_rounds']==[5,6,7,8,9]
assert cfg['bootstrap']==1000
print('PACKAGE_STATIC_CHECK: PASS')
PY
