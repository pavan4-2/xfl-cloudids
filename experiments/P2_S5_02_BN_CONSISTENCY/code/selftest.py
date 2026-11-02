"""Deterministic selftest only; no development data."""
import torch
from r2_backbone_selection import ModernTabularMLP
from p2_s5_02_localbn import clone_state,merge_states

def check():
 torch.manual_seed(11)
 m1=ModernTabularMLP(78,15,.3);m2=ModernTabularMLP(78,15,.3)
 m1.train();m2.train()
 with torch.no_grad():
  m1(torch.randn(8,78)+3);m2(torch.randn(8,78)-3)
 states=[clone_state(m1),clone_state(m2)]
 merged=merge_states(states,[.4,.6])
 local_state=dict(merged)
 for key in merged:
  if key.endswith(('running_mean','running_var','num_batches_tracked')):
   local_state[key]=states[0][key].clone()
   assert torch.equal(local_state[key],states[0][key])
 assert any(key.endswith('running_mean') and not torch.equal(merged[key],local_state[key]) for key in merged)
 print('P2_S5_02_LOCALBN_SELFTEST_PASS')
if __name__=='__main__':check()
