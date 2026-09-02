#!/usr/bin/env python3
import argparse
from f0r1_rutra import run_r1_4

p=argparse.ArgumentParser(description="F0-R1 R1.4 prospective reliability test")
p.add_argument("--artifact-root",default="artifacts/f0_r1")
p.add_argument("--bootstrap",type=int)
a=p.parse_args()
run_r1_4(a.artifact_root,a.bootstrap)
