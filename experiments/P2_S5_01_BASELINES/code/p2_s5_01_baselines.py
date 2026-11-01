"""P2-S5-01 | CPU architecture-matched Local-only/FedAvg/FedProx pilot.

Input is the existing governed CICIDS2017 development manifest only. All evaluation
and model-selection in this stage are descriptive, not prospective confirmation.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, platform, sys, zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from common import load_manifest, pool_role, row_key, classification_metrics, write_csv, dump_json, seed_all
from r2_backbone_selection import ModernTabularMLP, bounded_smote


METHODS = ('LOCAL_ONLY', 'FEDAVG', 'FEDPROX')


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def merge_states(states, fractions):
    assert abs(sum(fractions) - 1.0) < 1e-8
    result = {}
    for key in states[0]:
        values = [state[key] for state in states]
        if torch.is_floating_point(values[0]):
            result[key] = sum(float(w) * v.double() for w, v in zip(fractions, values)).to(dtype=values[0].dtype)
        else:
            # Matches System 6's frozen BatchNorm integer-buffer aggregation.
            result[key] = torch.stack(values).max(dim=0).values
    return result


def predict(model, X, batch=4096):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(X), batch):
            chunks.append(model(torch.from_numpy(X[start:start+batch])).argmax(1).numpy())
    return np.concatenate(chunks) if chunks else np.empty((0,), dtype=np.int64)


def train_local(model, X, y, cfg, round_id, cid, prox_reference=None):
    # Same local optimizer, dropout seed and two-epoch budget as P2-S6-01.
    torch.manual_seed(int(cfg['calibration_seed']) + 100 * round_id + cid)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg['learning_rate']),
                            weight_decay=float(cfg['weight_decay']))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=int(cfg['batch_size']), shuffle=True, drop_last=False)
    refs = None
    if prox_reference is not None:
        refs = [p.detach().clone() for p in prox_reference.parameters()]
    model.train()
    for _ in range(int(cfg['local_epochs'])):
        for xb, yb in loader:
            if len(xb) < 2:  # BatchNorm has no training statistic for a singleton batch.
                continue
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(xb), yb)
            if refs is not None:
                penalty = sum((p-r).square().sum() for p, r in zip(model.parameters(), refs))
                loss = loss + 0.5 * float(cfg['fedprox_mu']) * penalty
            loss.backward()
            opt.step()
    return model


def client_metrics(model, Xv, yv, ds, stable_classes):
    pred = predict(model, Xv)
    result = classification_metrics(yv, pred, ds.n_classes, ds.benign_label)
    supported = np.unique(yv).tolist()
    result['supported_macro_f1'] = float(f1_score(yv, pred, labels=supported,
                                                  average='macro', zero_division=0))
    result['stable_macro_f1'] = float(f1_score(yv, pred, labels=stable_classes,
                                               average='macro', zero_division=0))
    return result


def prepare_data(bridge, cfg, smoke=False):
    root = Path(bridge).resolve()
    if not smoke and 'SYSTEM5' not in str(root).upper():
        raise RuntimeError('STOP: expected System 5 governed bridge; report path mismatch')
    ds = load_manifest(root / cfg['manifest'], root, allowed_roles=('train', 'validation'))
    if ds.feature_dim != 78 or ds.n_classes != 15:
        raise ValueError('STOP: data dimensions/classes differ from frozen configuration')
    cids = sorted(ds.clients)
    if len(cids) < 2:
        raise ValueError('STOP: at least two clients required')
    if not smoke and ds.partition_lock != cfg['partition_lock']:
        raise RuntimeError('STOP: partition lock differs; never regenerate bridge')
    if not smoke and (ds.scenario != 'alpha_0p3' or ds.partition_seed != 17):
        raise RuntimeError('STOP: wrong scenario/partition seed')
    split_training, split_validation, n_original = [], [], []
    for cid in cids:
        sp = ds.clients[cid].splits['train']
        rng = np.random.default_rng(int(cfg['calibration_seed'])+cid)
        probe = np.zeros(len(sp.y), dtype=bool)
        for classid in range(ds.n_classes):
            indices = np.flatnonzero(sp.y == classid).copy()
            rng.shuffle(indices)
            take = 0 if len(indices) < 10 else max(1, int(round(len(indices)*float(cfg['probe_fraction_of_train']))))
            take = min(take, max(0, len(indices)-2))
            probe[indices[:take]] = True
        if not smoke and not probe.any():
            raise RuntimeError(f'STOP: no held-out development probe samples client {cid}')
        # As in S6, the same probe subset is withheld from optimizer TRAIN,
        # even though these architecture-matched baselines never use its labels.
        split_training.append((sp.X[~probe], sp.y[~probe]))
        split_validation.append((ds.clients[cid].splits['validation'].X,
                                 ds.clients[cid].splits['validation'].y))
        n_original.append(int((~probe).sum()))

    # Match the shared pooled TRAIN-only feature normalization used in P2-S6-01.
    scaler = StandardScaler().fit(np.concatenate([X for X, _ in split_training]))
    scaled_train = [(np.asarray(scaler.transform(X), dtype=np.float32),
                     y.astype(np.int64)) for X, y in split_training]
    scaled_val = [(np.asarray(scaler.transform(X), dtype=np.float32),
                   y.astype(np.int64)) for X, y in split_validation]

    global_train = pool_role(ds, 'train')
    keys = {row_key(X, int(y)) for X, y in zip(global_train.X, global_train.y)}
    kept_val = []
    duplicates = 0
    for (rawX, rawY), (Xs, Ys) in zip(split_validation, scaled_val):
        keep = np.asarray([row_key(x, int(y)) not in keys for x, y in zip(rawX, rawY)], dtype=bool)
        duplicates += int((~keep).sum())
        kept_val.append((Xs[keep], Ys[keep]))
    del keys
    if any(len(y) == 0 for _, y in kept_val):
        raise RuntimeError('STOP: client has zero evaluation rows after duplicate filter')
    n_train = sum(len(ds.clients[cid].splits['train'].y) for cid in cids)
    n_val = sum(len(ds.clients[cid].splits['validation'].y) for cid in cids)
    if not smoke and (n_train, n_val, duplicates, sum(len(y) for _, y in kept_val)) != (450000, 37501, 3097, 34404):
        raise RuntimeError('STOP: data counts/overlap differ from frozen shared bridge')
    all_val_y = np.concatenate([y for X, y in kept_val])
    stable = np.flatnonzero(np.bincount(all_val_y, minlength=ds.n_classes) >= 20).tolist()
    if not smoke and stable != [0,1,2,3,4,5,6,7,10,11,12,14]:
        raise RuntimeError('STOP: stable class set mismatch')
    scaler_digest = hashlib.sha256(scaler.mean_.tobytes()+scaler.scale_.tobytes()).hexdigest()
    audit = {'manifest': cfg['manifest'], 'scenario': ds.scenario, 'partition_seed':ds.partition_seed,
             'partition_lock': ds.partition_lock, 'n_clients':len(cids), 'client_ids':cids,
             'train_rows':n_train, 'validation_rows_raw':n_val,
             'validation_removed_train_overlaps':duplicates,
             'validation_rows_primary':sum(len(y) for X,y in kept_val),
             'stable_classes':stable, 'train_only_scaler_digest':scaler_digest,
             'probe_withheld_from_optimizer':int(n_train-sum(n_original)),
             'only_arrays_loaded':['train','validation']}
    return ds, cids, scaled_train, kept_val, n_original, stable, audit


def client_local_smote(X, y, cfg, seed):
    Xs, ys, audit = bounded_smote(X, y, int(cfg['n_classes']), int(cfg['client_smote_target']),
                                   int(cfg['client_smote_max_k']), seed)
    # S6's frozen R2 utility initially duplicates singleton rows; the P2 S6
    # adaptation discards those repeats. Apply exactly the same per-client rule.
    singleton = [int(a['class_id']) for a in audit
                 if int(a['original_rows']) == 1 and int(a['synthetic_rows']) > 0]
    if singleton:
        keep = np.ones(len(ys), dtype=bool)
        for cl in singleton:
            pos = np.flatnonzero(ys == cl)
            keep[pos[1:]] = False
        Xs, ys = Xs[keep], ys[keep]
    for item in audit:
        if int(item['class_id']) in singleton:
            item['removed_singleton_copies'] = int(item['synthetic_rows'])
            item['effective_synthetic_rows'] = 0
        else:
            item['removed_singleton_copies'] = 0
            item['effective_synthetic_rows'] = int(item['synthetic_rows'])
    return Xs, ys, audit


def run(bridge: Path, out: Path, protocol: Path, smoke=False):
    cfg = json.loads(Path(protocol).read_text(encoding='utf-8'))
    if cfg['backbone'] != 'B_MLP_SMOTE_CE' or cfg['manifest'] != 'manifest_alpha_0p3_seed17.json':
        raise RuntimeError('STOP: frozen backbone/manifest mismatch')
    if cfg['calibration_seed'] != 11 or cfg['n_classes'] != 15:
        raise RuntimeError('STOP: wrong seed/class count')
    device = 'cpu'
    torch.set_num_threads(1)
    torch.manual_seed(int(cfg['calibration_seed']))
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True, warn_only=True)
    seed_all(cfg['calibration_seed'])
    ds, cids, trains, vals, counts, stable, data_audit = prepare_data(bridge, cfg, smoke)
    result_dir = out / 'results'
    result_dir.mkdir(parents=True, exist_ok=False)
    cfg = dict(cfg)
    if smoke:
        cfg.update(rounds=2, local_epochs=1, batch_size=16, client_smote_target=6)

    torch.manual_seed(cfg['calibration_seed'])
    init_model = ModernTabularMLP(ds.feature_dim, ds.n_classes, cfg['dropout'])
    initial = clone_state(init_model)
    models = {}
    for method in METHODS:
        models[method] = []
        for cid in cids:
            m = ModernTabularMLP(ds.feature_dim, ds.n_classes, cfg['dropout'])
            m.load_state_dict(initial)
            models[method].append(m)
    globals_ = {method:copy.deepcopy(initial) for method in ('FEDAVG','FEDPROX')}

    metrics = []
    class_rows = []
    round_summaries = []
    aggregation_rows = []
    smote_rows = []
    for round_id in range(1, int(cfg['rounds'])+1):
        cached_training = []
        for i,cid in enumerate(cids):
            X,y = trains[i]
            xs,ys,audit = client_local_smote(X,y,cfg,int(cfg['calibration_seed'])+round_id*100+cid)
            cached_training.append((xs,ys))
            for entry in audit:
                smote_rows.append({'round':round_id,'client':cid,**entry})
        for method in METHODS:
            local_states=[]; per_client_pre=[]
            for i,cid in enumerate(cids):
                model = models[method][i]
                prox_ref = None
                if method in globals_:
                    model.load_state_dict(globals_[method])
                    if method == 'FEDPROX':
                        prox_ref = ModernTabularMLP(ds.feature_dim,ds.n_classes,cfg['dropout'])
                        prox_ref.load_state_dict(globals_[method]);prox_ref.eval()
                train_local(model,*cached_training[i],cfg,round_id,cid,prox_reference=prox_ref)
                pre = client_metrics(model,*vals[i],ds,stable)
                per_client_pre.append(pre)
                local_states.append(clone_state(model))
            if method != 'LOCAL_ONLY':
                fractions=np.asarray(counts,dtype=np.float64)/sum(counts)
                global_state = merge_states(local_states, fractions)
                globals_[method] = global_state
                for i,cid in enumerate(cids):
                    models[method][i].load_state_dict(global_state)
                    aggregation_rows.append({'round':round_id,'method':method,'client':cid,
                                             'client_weight':float(fractions[i]),
                                             'weight_denominator':'original_train_rows_after_probe_exclusion_not_SMOTE'})
            for i,cid in enumerate(cids):
                result = client_metrics(models[method][i],*vals[i],ds,stable)
                pre = per_client_pre[i]
                line={'round':round_id,'method':method,'cid':cid,'n_train_original_after_probe':counts[i],
                      'n_val':len(vals[i][1]),'accuracy':result['accuracy'],'weighted_f1':result['weighted_f1'],
                      'macro_f1_all_15':result['macro_f1'], 'balanced_accuracy':result['balanced_accuracy'],
                      'far':result['far'],'supported_macro_f1':result['supported_macro_f1'],
                      'stable_macro_f1':result['stable_macro_f1'],
                      'preaggregation_supported_macro_f1':pre['supported_macro_f1'],
                      'negative_transfer_vs_preaggregation':int(method!='LOCAL_ONLY' and
                          result['supported_macro_f1'] < pre['supported_macro_f1']-0.005)}
                metrics.append(line)
                for cls in result['per_class']:
                    class_rows.append({'round':round_id,'method':method,'cid':cid,
                                       'class_id':cls['class_id'],'support':cls['support'],
                                       'precision':cls['precision'],'recall':cls['recall'],'f1':cls['f1']})
        # Describe paired comparison using local-only client after this same round.
        round_records=[r for r in metrics if r['round']==round_id]
        locals_by_cid = {r['cid']:r for r in round_records if r['method']=='LOCAL_ONLY'}
        for r in round_records:
            r['delta_supported_macro_vs_local_only'] = float(r['supported_macro_f1'] - locals_by_cid[r['cid']]['supported_macro_f1'])
            r['negative_transfer_vs_local_only'] = int(r['delta_supported_macro_vs_local_only'] < -0.005)
        for method in METHODS:
            rr=[r for r in round_records if r['method']==method]
            round_summaries.append({'round':round_id, 'method':method,
                                    'mean_client_supported_macro_f1':float(np.mean([r['supported_macro_f1'] for r in rr])),
                                    'worst_client_supported_macro_f1':float(min(r['supported_macro_f1'] for r in rr)),
                                    'mean_client_accuracy':float(np.mean([r['accuracy'] for r in rr])),
                                    'mean_client_far':float(np.mean([r['far'] for r in rr])),
                                    'negative_transfer_vs_local_only_count':sum(r['negative_transfer_vs_local_only'] for r in rr),
                                    'n_clients':len(rr)})
        print(f'P2_S5_01_ROUND_DONE={round_id}/{cfg["rounds"]}',flush=True)
    write_csv(result_dir/'client_round_metrics.csv',metrics)
    write_csv(result_dir/'client_round_class_metrics.csv',class_rows)
    write_csv(result_dir/'round_summary.csv',round_summaries)
    write_csv(result_dir/'aggregation_weights.csv',aggregation_rows)
    write_csv(result_dir/'local_smote_audit.csv',smote_rows)
    dump_json(result_dir/'data_audit.json',data_audit)
    dump_json(result_dir/'protocol_snapshot.json',cfg)
    dump_json(result_dir/'environment.json',{'python':platform.python_version(),
              'torch':torch.__version__,'numpy':np.__version__,'device':'cpu',
              'deterministic_warning_only':True})
    last = {r['method']:r for r in round_summaries if r['round']==cfg['rounds']}
    dump_json(result_dir/'summary.json',{
        'step':'P2-S5-01','technical_status':'TECHNICAL_PASS',
        'scientific_status':'DEVELOPMENT_ONLY_NOT_CONFIRMATORY',
        'backbone':'B_MLP_SMOTE_CE', 'R2_original_gate':'NO_CANDIDATE_PASSED',
        'methods':list(METHODS),'rounds':cfg['rounds'], 'local_epochs':cfg['local_epochs'],
        'mu_fedprox':cfg['fedprox_mu'],'partition_lock':ds.partition_lock,
        'scenario':ds.scenario, 'calibration_seed':cfg['calibration_seed'],
        'last_round_summary':last,
        'privacy_note':'Centralized simulation shares TRAIN-only feature moments and FL states; no secure aggregation claim.',
        'data_access':['train','validation'],
        'caveats':['Single seed and partition, NOT independent generalization',
                    'FedProx proximal coefficient is a frozen engineering default for this pilot',
                    'SMOTE per-client training only; singleton copies removed',
                    'Primary evaluation excludes exact TRAIN-overlap validation rows',
                    'Stable-class macro-F1 applies global support threshold, not client-level support']})
    archive = out/'P2_S5_01_RESULTS.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(result_dir.glob('*')):
            z.write(f,'results/'+f.name)
    if zipfile.ZipFile(archive).testzip():
        raise IOError('result archive verification failed')
    print(f'P2_S5_01_RESULTS_WRITTEN={archive}',flush=True)
    return archive


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bridge',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--protocol',required=True)
    ap.add_argument('--smoke',action='store_true',help='Synthetic-only internal package test; NOT an authorized student run')
    args=ap.parse_args()
    run(Path(args.bridge),Path(args.out),Path(args.protocol),smoke=args.smoke)

if __name__=='__main__': main()
