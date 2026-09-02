#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-smoke/f0_r1}"
rm -rf "$ROOT"
python R1_1_prepare_corrected_partitions.py --synthetic-smoke --artifact-root "$ROOT"
python R1_2_aligned_negative_transfer.py --artifact-root "$ROOT" --rounds 3 --local-epochs 1
python R1_3_transfer_utility.py --artifact-root "$ROOT" --rounds 3 --windows 2
python R1_4_reliability_grid.py --artifact-root "$ROOT" --bootstrap 100
python R1_5_multiround_oracle_gate.py --artifact-root "$ROOT" --rounds 3 --gate-bootstrap 100
printf '\nSynthetic smoke completed. These outputs are NOT research evidence. Do not run R1.6 on them.\n'
