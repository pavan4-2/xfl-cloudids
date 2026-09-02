#!/usr/bin/env python3
import argparse,sys
from f0r1_core import verify_system5

p=argparse.ArgumentParser(description="Compare independent System 5 R1.1 evidence against System 6")
p.add_argument("--main-r1",required=True)
p.add_argument("--system5-r1",required=True)
p.add_argument("--output",required=True)
a=p.parse_args()
sys.exit(0 if verify_system5(a.main_r1,a.system5_r1,a.output) else 1)
