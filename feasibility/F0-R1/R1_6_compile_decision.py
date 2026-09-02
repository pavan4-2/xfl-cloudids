#!/usr/bin/env python3
import argparse
from f0r1_core import run_r1_6

p=argparse.ArgumentParser(description="F0-R1 R1.6 compile supervisor decision package")
p.add_argument("--environment-lock",required=True)
p.add_argument("--terminal-log-dir",required=True)
p.add_argument("--system5-reproduction",required=True)
p.add_argument("--git-commit",required=True)
p.add_argument("--confirm-test-unopened",action="store_true")
p.add_argument("--confirm-thresholds-unchanged",action="store_true")
p.add_argument("--confirm-failures-preserved",action="store_true")
a=p.parse_args()
run_r1_6(a.environment_lock,a.terminal_log_dir,a.system5_reproduction,a.git_commit,
         a.confirm_test_unopened,a.confirm_thresholds_unchanged,a.confirm_failures_preserved)
