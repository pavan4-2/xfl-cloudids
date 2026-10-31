import numpy as np
import torch
from r2_backbone_selection import ModernTabularMLP, CNN1DTabular, FocalLoss, bounded_smote, inverse_sqrt_alpha

X=np.array([[0.,0.],[1.,1.],[2.,2.],[3.,3.],[10.,10.],[11.,11.]],dtype=np.float32)
y=np.array([0,0,0,0,1,1],dtype=np.int64)
Xs,ys,a=bounded_smote(X,y,2,target=4,kmax=2,seed=11)
assert Xs.shape==(8,2) and ys.shape==(8,)
assert np.bincount(ys,minlength=2).tolist()==[4,4]
alpha,_=inverse_sqrt_alpha(np.array([0,0,0,1,1,2],dtype=np.int64),3)
loss=FocalLoss(torch.from_numpy(alpha),2.0)(torch.randn(6,3),torch.tensor([0,0,0,1,1,2]))
assert torch.isfinite(loss)
for m in [ModernTabularMLP(78,15,0.3),CNN1DTabular(78,15,0.3)]:
    out=m(torch.randn(4,78)); assert out.shape==(4,15)
print('SELFTEST_PASS')
