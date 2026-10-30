from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from common import classification_metrics, load_manifest, pool_role, validation_duplicate_mask


def build_bridge(root: Path):
    rng=np.random.default_rng(7); n_classes=3; dim=5
    clients=[]
    for cid in range(2):
        splits={}
        for role,n in [("train",60),("utility",20),("gate",20),("validation",15)]:
            y=np.arange(n,dtype=np.int64)%n_classes
            X=(rng.normal(size=(n,dim)).astype(np.float32)+y[:,None]*0.75+cid*0.1)
            fn=f"c{cid}_{role}.npz"; np.savez(root/fn,X=X,y=y); splits[role]={"path":fn}
        clients.append({"cid":cid,"splits":splits})
    for scenario in ["alpha_0p1","alpha_0p3"]:
        for seed in [17,29]:
            m={"scope":"development_only","feature_dim":dim,"n_classes":n_classes,"benign_label":0,"clients":clients,
               "partition_lock":"SYNTHETIC_LOCK_1234567890","data_origin":"SYNTHETIC_SELFTEST","scenario":scenario,"partition_seed":seed}
            (root/f"manifest_{scenario}_seed{seed}.json").write_text(json.dumps(m),encoding="utf-8")


def main():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); build_bridge(root)
        ds=load_manifest(root/"manifest_alpha_0p3_seed17.json",root)
        train=pool_role(ds,"train"); val=pool_role(ds,"validation")
        mask=validation_duplicate_mask(train,val)
        assert len(train.y)==120 and len(val.y)==30 and mask.dtype==bool
        pred=val.y.copy(); m=classification_metrics(val.y,pred,ds.n_classes,ds.benign_label)
        assert abs(m["accuracy"]-1.0)<1e-12 and abs(m["macro_f1"]-1.0)<1e-12 and abs(m["far"]-0.0)<1e-12
    print("R0_R1_SELFTEST=PASS")

if __name__=="__main__": main()
