#!/usr/bin/env python3
import argparse
from f0r1_core import run_r1_2

p=argparse.ArgumentParser(description="F0-R1 R1.2 aligned negative-transfer test")
p.add_argument("--artifact-root",default="artifacts/f0_r1")
p.add_argument("--rounds",type=int)
p.add_argument("--local-epochs",type=int)
a=p.parse_args()
run_r1_2(a.artifact_root,a.rounds,a.local_epochs)
