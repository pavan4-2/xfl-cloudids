#!/usr/bin/env python3
import argparse
from f0r1_rutra import run_r1_5

p=argparse.ArgumentParser(description="F0-R1 R1.5 multi-round oracle and minimal RUTRA-Gate")
p.add_argument("--artifact-root",default="artifacts/f0_r1")
p.add_argument("--rounds",type=int)
p.add_argument("--gate-bootstrap",type=int)
a=p.parse_args()
run_r1_5(a.artifact_root,a.rounds,a.gate_bootstrap)
