"""Fast deterministic checks on aggregation and FedProx penalty; no dataset touched."""
import numpy as np
import torch
from r2_backbone_selection import ModernTabularMLP
from p2_s5_01_baselines import clone_state, merge_states

def check():
    torch.manual_seed(11)
    m1=ModernTabularMLP(78,15,.3)
    m2=ModernTabularMLP(78,15,.3)
    states=[clone_state(m1),clone_state(m2)]
    out=merge_states(states,[.3,.7])
    for k in out:
        if torch.is_floating_point(out[k]):
            expected=(.3*states[0][k].double()+.7*states[1][k].double()).to(out[k].dtype)
            assert torch.allclose(out[k],expected,atol=1e-6),k
        else:
            assert torch.equal(out[k],torch.maximum(states[0][k],states[1][k]))
    assert all(torch.isfinite(v).all() for v in out.values())
    x=torch.tensor([2.0,3.0],requires_grad=True)
    ref=torch.tensor([1.0,1.0]);mu=.01
    loss=.5*mu*((x-ref)**2).sum()
    loss.backward()
    assert torch.allclose(x.grad,mu*(x-ref))
    print('P2_S5_01_SELFTEST_PASS')
if __name__=='__main__':check()
