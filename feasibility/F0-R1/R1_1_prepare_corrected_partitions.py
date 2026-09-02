#!/usr/bin/env python3
import argparse
from f0r1_core import prepare_r1_1

p=argparse.ArgumentParser(description="F0-R1 R1.1 corrected sampling and partitions")
p.add_argument("--raw-dir")
p.add_argument("--artifact-root",default="artifacts/f0_r1")
p.add_argument("--synthetic-smoke",action="store_true")
a=p.parse_args()
prepare_r1_1(a.raw_dir,a.artifact_root,a.synthetic_smoke)
