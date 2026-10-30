from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

from common import dump_json, seed_all, system_environment, write_csv
from r1_common import emit_metrics, load_r1_data


class ModernTabularMLP(nn.Module):
    def __init__(self, dim: int, classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, classes),
        )
    def forward(self, x):
        return self.net(x)


def internal_split(y: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    train_idx=[]; val_idx=[]
    for c in range(int(y.max())+1):
        idx=np.where(y==c)[0].copy(); rng.shuffle(idx)
        # Keep ultra-rare classes entirely in training. For sufficiently supported
        # classes reserve 10% for early stopping only.
        if len(idx) >= 20:
            n_val=max(1,int(round(0.10*len(idx))))
            val_idx.extend(idx[:n_val].tolist()); train_idx.extend(idx[n_val:].tolist())
        else:
            train_idx.extend(idx.tolist())
    train_idx=np.asarray(train_idx,dtype=np.int64); val_idx=np.asarray(val_idx,dtype=np.int64)
    rng.shuffle(train_idx); rng.shuffle(val_idx)
    return train_idx,val_idx


def predict(model, X: np.ndarray, device: str, batch_size: int=4096) -> np.ndarray:
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(X),batch_size):
            xb=torch.from_numpy(X[i:i+batch_size]).to(device,non_blocking=True)
            out.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--bridge",required=True); ap.add_argument("--out",required=True); ap.add_argument("--system",choices=["SYSTEM6"],default="SYSTEM6")
    args=ap.parse_args(); root=Path(__file__).resolve().parent
    spec=json.loads((root/"RECOVERY_SPECIFICATION.json").read_text()); cfg=spec["r1_system6_model"]
    out=Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    seed=int(spec["calibration_seed"]); seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("STOP: R1 System 6 neural ceiling requires CUDA; do not silently change to CPU")
    device="cuda"
    env=system_environment(); env.update({"step":"R1","system":"SYSTEM6","role":"centralized_modern_mlp_ceiling"}); dump_json(out/"R1_SYSTEM6_ENVIRONMENT.json",env)

    ds,train,validation,duplicate_mask,clean_mask=load_r1_data(Path(args.bridge).resolve(),spec)
    tr_idx,es_idx=internal_split(train.y,seed)
    if len(es_idx)==0: raise AssertionError("no internal early-stop rows")

    # Train-only scaling: the reporting validation split never contributes to mean/std.
    mean=train.X[tr_idx].mean(axis=0,dtype=np.float64).astype(np.float32)
    std=train.X[tr_idx].std(axis=0,dtype=np.float64).astype(np.float32); std[std<1e-6]=1.0
    Xtr=((train.X[tr_idx]-mean)/std).astype(np.float32,copy=False)
    ytr=train.y[tr_idx]
    Xes=((train.X[es_idx]-mean)/std).astype(np.float32,copy=False)
    yes=train.y[es_idx]
    Xreport=((validation.X-mean)/std).astype(np.float32,copy=False)
    dump_json(out/"R1_SYSTEM6_TRAIN_ONLY_SCALER.json",{"mean":mean.tolist(),"std":std.tolist(),"fit_rows":int(len(tr_idx)),"reporting_validation_used_for_fit":False})

    dataset=TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(ytr))
    g=torch.Generator().manual_seed(seed)
    loader=DataLoader(dataset,batch_size=int(cfg["batch_size"]),shuffle=True,generator=g,num_workers=0,pin_memory=True,drop_last=False)
    model=ModernTabularMLP(ds.feature_dim,ds.n_classes,float(cfg["dropout"])).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=float(cfg["learning_rate"]),weight_decay=float(cfg["weight_decay"]))
    criterion=nn.CrossEntropyLoss()
    history=[]; best_score=-1.0; best_epoch=0; best_state=None; stale=0
    t0=time.perf_counter()
    for epoch in range(1,int(cfg["max_epochs"])+1):
        model.train(); total_loss=0.0; total_rows=0
        for xb,yb in loader:
            xb=xb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True); logits=model(xb); loss=criterion(logits,yb); loss.backward(); opt.step()
            total_loss += float(loss.item())*len(yb); total_rows += len(yb)
        es_pred=predict(model,Xes,device)
        es_weighted=float(f1_score(yes,es_pred,average="weighted",zero_division=0))
        es_macro=float(f1_score(yes,es_pred,average="macro",labels=np.arange(ds.n_classes),zero_division=0))
        row={"epoch":epoch,"train_ce":total_loss/max(total_rows,1),"early_stop_weighted_f1":es_weighted,"early_stop_macro_f1":es_macro}
        history.append(row); print(json.dumps(row),flush=True)
        if es_weighted > best_score + 1e-6:
            best_score=es_weighted; best_epoch=epoch; best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()}); stale=0
        else:
            stale += 1
        if epoch >= int(cfg["min_epochs"]) and stale >= int(cfg["early_stopping_patience"]):
            break
    fit_seconds=time.perf_counter()-t0
    if best_state is None: raise AssertionError("no best neural state")
    model.load_state_dict(best_state)
    pred=predict(model,Xreport,device)
    write_csv(out/"R1_SYSTEM6_TRAINING_HISTORY.csv",history)
    result=emit_metrics(out,"MODERN_TABULAR_MLP",ds,validation,pred,duplicate_mask,spec["r1_ceiling_gates"]["system6_mlp"])
    summary={
        "step":"R1","system":"SYSTEM6","status":"TECHNICAL_PASS","experiment":"centralized_performance_ceiling",
        "calibration_seed":seed,"canonical_manifest":spec["canonical_manifest"],"train_rows_total":int(len(train.y)),
        "train_rows_optimizer":int(len(tr_idx)),"internal_early_stop_rows":int(len(es_idx)),"raw_validation_rows":int(len(validation.y)),
        "deduplicated_validation_rows":int(clean_mask.sum()),"model":"ModernTabularMLP_256_128_BN_Dropout",
        "optimizer":"AdamW","learning_rate":float(cfg["learning_rate"]),"weight_decay":float(cfg["weight_decay"]),
        "training_loss":"ordinary_unweighted_cross_entropy","best_epoch":int(best_epoch),"best_internal_weighted_f1":float(best_score),
        "fit_seconds":fit_seconds,"primary_metrics":result["primary_deduplicated_validation"],
        "ceiling_gate_pass":result["ceiling_gate_pass"],"sealed_final_test_access":False,"automatic_r2_authorization":False,
        "scientific_interpretation":"SUPERVISOR_ONLY"
    }
    dump_json(out/"R1_SYSTEM6_SUMMARY.json",summary); print(json.dumps({"R1_SYSTEM6":summary},sort_keys=True))

if __name__=="__main__": main()
