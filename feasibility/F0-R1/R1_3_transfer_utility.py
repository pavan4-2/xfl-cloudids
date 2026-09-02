#!/usr/bin/env python3
import argparse
from f0r1_rutra import run_r1_3

p=argparse.ArgumentParser(description="F0-R1 R1.3 recipient/class/window transfer utility")
p.add_argument("--artifact-root",default="artifacts/f0_r1")
p.add_argument("--rounds",type=int)
p.add_argument("--windows",type=int)
a=p.parse_args()
run_r1_3(a.artifact_root,a.rounds,a.windows)
