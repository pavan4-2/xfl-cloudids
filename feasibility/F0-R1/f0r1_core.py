from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "f0_config.json"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def canonical_json(obj) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(obj), encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def deterministic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.save(buf, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            zf.writestr(info, buf.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def stable_seed(*parts: object) -> int:
    s = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(s).hexdigest()[:8], 16) & 0x7FFFFFFF


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def artifact_dir(artifact_root: str | Path, stage: str) -> Path:
    p = Path(artifact_root) / stage
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_label(x: object) -> str:
    s = str(x).strip().upper()
    s = s.replace("–", "-").replace("—", "-")
    s = "_".join(s.split())
    return s


def normalize_col(x: object) -> str:
    return " ".join(str(x).strip().split())


def raw_dataset_hashes(csv_paths: Sequence[Path]) -> List[dict]:
    return [{"file": p.name, "sha256": sha256_file(p), "bytes": p.stat().st_size} for p in csv_paths]


def find_label_column(csv_paths: Sequence[Path]) -> str:
    for p in csv_paths:
        cols = list(pd.read_csv(p, nrows=0).columns)
        for c in cols:
            if normalize_col(c).lower() == "label":
                return c
    raise RuntimeError("Could not find a column named 'Label' in CIC-IDS2017 CSV files.")


def class_counts_first_pass(csv_paths: Sequence[Path], label_raw: str, chunksize: int = 250000) -> Counter:
    counts: Counter = Counter()
    for p in csv_paths:
        header = list(pd.read_csv(p, nrows=0).columns)
        local_label = next((c for c in header if normalize_col(c).lower() == "label"), None)
        if local_label is None:
            raise RuntimeError(f"No Label column in {p.name}")
        for chunk in pd.read_csv(p, usecols=[local_label], chunksize=chunksize, low_memory=False):
            vals = chunk[local_label].map(normalize_label)
            counts.update(vals.tolist())
    return counts


def quotas_from_counts(counts: Mapping[str, int], cap: int, rare_max: int) -> Tuple[Dict[str, int], Dict[str, bool]]:
    total = int(sum(counts.values()))
    if total <= cap:
        return {k: int(v) for k, v in counts.items()}, {k: True for k in counts}
    rare = {k: int(v) for k, v in counts.items() if int(v) <= rare_max}
    rare_total = sum(rare.values())
    if rare_total >= cap:
        raise RuntimeError("Rare-preservation reserve exceeds row cap; protocol cannot proceed without amendment.")
    common = {k: int(v) for k, v in counts.items() if k not in rare}
    remaining = cap - rare_total
    common_total = sum(common.values())
    raw = {k: remaining * v / common_total for k, v in common.items()}
    q = {k: int(math.floor(v)) for k, v in raw.items()}
    remainder = remaining - sum(q.values())
    order = sorted(common, key=lambda k: (raw[k] - q[k], common[k], k), reverse=True)
    for k in order[:remainder]:
        q[k] += 1
    quotas = {**rare, **q}
    preserved = {k: k in rare for k in counts}
    return quotas, preserved


def selected_occurrence_positions(counts: Mapping[str, int], quotas: Mapping[str, int], seed: int) -> Dict[str, np.ndarray]:
    result = {}
    for label in sorted(counts):
        n, q = int(counts[label]), int(quotas[label])
        if q >= n:
            result[label] = np.arange(n, dtype=np.int64)
        else:
            rng = np.random.default_rng(stable_seed(seed, "sampling", label))
            result[label] = np.sort(rng.choice(n, size=q, replace=False).astype(np.int64))
    return result


def read_sampled_real_data(csv_paths: Sequence[Path], positions: Mapping[str, np.ndarray], chunksize: int = 100000) -> pd.DataFrame:
    offsets = defaultdict(int)
    pieces = []
    source_order = 0
    for p in csv_paths:
        for chunk in pd.read_csv(p, chunksize=chunksize, low_memory=False):
            chunk.columns = [normalize_col(c) for c in chunk.columns]
            label_col = next((c for c in chunk.columns if c.lower() == "label"), None)
            if label_col is None:
                raise RuntimeError(f"No normalized Label column in {p.name}")
            labels = chunk[label_col].map(normalize_label).to_numpy()
            take = np.zeros(len(chunk), dtype=bool)
            for lab in np.unique(labels):
                locs = np.flatnonzero(labels == lab)
                start = offsets[lab]
                end = start + len(locs)
                selected = positions.get(lab, np.empty(0, dtype=np.int64))
                left = np.searchsorted(selected, start, side="left")
                right = np.searchsorted(selected, end, side="left")
                if right > left:
                    rel = selected[left:right] - start
                    take[locs[rel]] = True
                offsets[lab] = end
            if take.any():
                chosen = chunk.loc[take].copy()
                chosen[label_col] = chosen[label_col].map(normalize_label)
                chosen["__source_order"] = np.arange(source_order, source_order + len(chunk), dtype=np.int64)[take]
                pieces.append(chosen)
            source_order += len(chunk)
    if not pieces:
        raise RuntimeError("Sampling selected zero rows.")
    return pd.concat(pieces, ignore_index=True)


def generate_synthetic_data(seed: int = 17) -> Tuple[pd.DataFrame, Counter, Dict[str, int], Dict[str, bool], List[dict]]:
    rng = np.random.default_rng(seed)
    specs = [
        ("BENIGN", 3000),
        ("DOS_HULK", 1500),
        ("PORTSCAN", 900),
        ("BOT", 400),
        ("HEARTBLEED", 100),
        ("INFILTRATION", 100),
    ]
    n_features = 14
    rows = []
    order = 0
    for ci, (label, n) in enumerate(specs):
        mean = (ci - 1.0) * 0.55
        X = rng.normal(mean, 1.0 + 0.05 * ci, size=(n, n_features))
        X[:, 0] += ci * 0.8
        X[:, 1] += np.sin(np.arange(n) / (20 + ci))
        d = pd.DataFrame(X, columns=[f"Feature_{j:02d}" for j in range(n_features)])
        d["Label"] = label
        d["__source_order"] = np.arange(order, order + n)
        rows.append(d)
        order += n
    df = pd.concat(rows, ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).sort_values("__source_order").reset_index(drop=True)
    counts = Counter(df["Label"].tolist())
    quotas = dict(counts)
    preserved = {k: True for k in counts}
    hashes = [{"file": "SYNTHETIC_SMOKE_ONLY", "sha256": "NOT_RESEARCH_EVIDENCE", "bytes": 0}]
    return df, counts, quotas, preserved, hashes


def stratified_split_indices(y: np.ndarray, proportions: Sequence[float], seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in proportions]
    for cls in sorted(np.unique(y).tolist()):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n = len(idx)
        exact = np.array(proportions, dtype=float) * n
        alloc = np.floor(exact).astype(int)
        remainder = n - int(alloc.sum())
        order = np.argsort(-(exact - alloc))
        for j in order[:remainder]:
            alloc[j] += 1
        if n >= len(proportions):
            zero = np.flatnonzero(alloc == 0)
            for z in zero:
                donor = int(np.argmax(alloc))
                if alloc[donor] > 1:
                    alloc[donor] -= 1
                    alloc[z] += 1
        start = 0
        for j, a in enumerate(alloc):
            if a:
                buckets[j].extend(idx[start:start + a].tolist())
            start += a
    out = []
    for b in buckets:
        arr = np.asarray(b, dtype=np.int64)
        rng.shuffle(arr)
        out.append(arr)
    return out


def extract_features(df: pd.DataFrame, train_idx: np.ndarray) -> Tuple[np.ndarray, List[str], dict]:
    label_col = next(c for c in df.columns if c.lower() == "label")
    exclude_exact = {"flow id", "source ip", "destination ip", "timestamp", label_col.lower(), "__source_order"}
    candidate = [c for c in df.columns if c.lower() not in exclude_exact]
    numeric_cols = []
    numeric_data = {}
    dropped = []
    for c in candidate:
        s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        train_non_na = int(s.iloc[train_idx].notna().sum())
        if train_non_na == 0:
            dropped.append(c)
            continue
        numeric_cols.append(c)
        numeric_data[c] = s.astype("float64")
    if not numeric_cols:
        raise RuntimeError("No numeric CIC-IDS2017 features remain after preprocessing.")
    medians, means, stds = {}, {}, {}
    X = np.empty((len(df), len(numeric_cols)), dtype=np.float32)
    for j, c in enumerate(numeric_cols):
        s = numeric_data[c]
        med = float(s.iloc[train_idx].median())
        filled = s.fillna(med).to_numpy(dtype=np.float64)
        mean = float(np.mean(filled[train_idx]))
        std = float(np.std(filled[train_idx]))
        if not np.isfinite(std) or std < 1e-12:
            std = 1.0
        X[:, j] = ((filled - mean) / std).astype(np.float32)
        medians[c], means[c], stds[c] = med, mean, std
    prep = {"medians": medians, "means": means, "stds": stds, "dropped_non_numeric_or_empty": dropped}
    return X, numeric_cols, prep


def dirichlet_partition_indices(y: np.ndarray, n_tenants: int, alpha: float, seed: int, min_size: int = 20) -> List[np.ndarray]:
    for attempt in range(100):
        rng = np.random.default_rng(stable_seed(seed, alpha, attempt))
        parts = [[] for _ in range(n_tenants)]
        for cls in sorted(np.unique(y).tolist()):
            idx = np.flatnonzero(y == cls)
            rng.shuffle(idx)
            props = rng.dirichlet(np.repeat(alpha, n_tenants))
            counts = rng.multinomial(len(idx), props)
            start = 0
            for t, c in enumerate(counts):
                if c:
                    parts[t].extend(idx[start:start + c].tolist())
                start += c
        arrs = [np.asarray(x, dtype=np.int64) for x in parts]
        if min(map(len, arrs)) >= min_size or len(y) < n_tenants * min_size:
            for a in arrs:
                rng.shuffle(a)
            return arrs
    raise RuntimeError(f"Unable to obtain non-empty Dirichlet partition for alpha={alpha}, seed={seed}.")


def prepare_r1_1(raw_dir: str | None, artifact_root: str, synthetic_smoke: bool) -> dict:
    cfg = load_config()
    out = artifact_dir(artifact_root, "r1_1")
    (out / "partitions").mkdir(parents=True, exist_ok=True)
    if synthetic_smoke:
        df, counts, quotas, preserved, raw_hashes = generate_synthetic_data(cfg["seeds"][0])
        mode = "SYNTHETIC_SMOKE_ONLY"
    else:
        if not raw_dir:
            raise SystemExit("--raw-dir is required for the real F0-R1 run.")
        csv_paths = sorted(Path(raw_dir).glob("*.csv"))
        if not csv_paths:
            raise SystemExit(f"No CSV files found under {raw_dir}")
        label_raw = find_label_column(csv_paths)
        counts = class_counts_first_pass(csv_paths, label_raw)
        quotas, preserved = quotas_from_counts(counts, cfg["rows_cap"], cfg["rare_preserve_max_count"])
        positions = selected_occurrence_positions(counts, quotas, cfg["seeds"][0])
        df = read_sampled_real_data(csv_paths, positions)
        raw_hashes = raw_dataset_hashes(csv_paths)
        mode = "REAL_CIC_IDS2017"
    label_col = next(c for c in df.columns if c.lower() == "label")
    labels = sorted(df[label_col].map(normalize_label).unique().tolist())
    if "BENIGN" in labels:
        labels = ["BENIGN"] + [x for x in labels if x != "BENIGN"]
    label_to_id = {x: i for i, x in enumerate(labels)}
    y_all = df[label_col].map(normalize_label).map(label_to_id).to_numpy(dtype=np.int64)
    proportions = [cfg["splits"]["train"], cfg["splits"]["utility"], cfg["splits"]["gate"], cfg["splits"]["sealed_test"]]
    split_idx = stratified_split_indices(y_all, proportions, stable_seed(cfg["amendment"], "split"))
    train_idx, utility_idx, gate_idx, test_idx = split_idx
    X_all, feature_names, prep = extract_features(df, train_idx)

    deterministic_npz(out / "development_data.npz",
                      X_train=X_all[train_idx], y_train=y_all[train_idx],
                      X_utility=X_all[utility_idx], y_utility=y_all[utility_idx],
                      X_gate=X_all[gate_idx], y_gate=y_all[gate_idx],
                      order_train=df["__source_order"].to_numpy(dtype=np.int64)[train_idx],
                      order_utility=df["__source_order"].to_numpy(dtype=np.int64)[utility_idx],
                      order_gate=df["__source_order"].to_numpy(dtype=np.int64)[gate_idx])
    test_path = out / "untouched_final_test_R1.npz"
    deterministic_npz(test_path,
                      X_test=X_all[test_idx], y_test=y_all[test_idx],
                      order_test=df["__source_order"].to_numpy(dtype=np.int64)[test_idx],
                      feature_names=np.asarray(feature_names, dtype="U"),
                      label_names=np.asarray(labels, dtype="U"))
    config_hash = sha256_file(CONFIG_PATH)
    seal = {
        "protocol": cfg["protocol"], "amendment": cfg["amendment"], "file": test_path.name,
        "sha256": sha256_file(test_path), "bytes": test_path.stat().st_size,
        "config_sha256": config_hash, "instruction": "DO NOT OPEN OR EVALUATE THIS NPZ DURING F0-R1"
    }
    write_json(out / "untouched_final_test_R1.SEAL.json", seal)

    audit_rows = []
    for lab in sorted(counts):
        audit_rows.append({
            "label": lab,
            "original_rows": int(counts[lab]),
            "selected_rows": int(quotas.get(lab, 0)),
            "rare_preserved": bool(preserved.get(lab, False)),
            "selection_fraction": float(quotas.get(lab, 0) / counts[lab]) if counts[lab] else 0.0,
        })
    pd.DataFrame(audit_rows).to_csv(out / "sampling_audit.csv", index=False)

    with np.load(out / "development_data.npz", allow_pickle=False) as dev:
        split_arrays = {k: (dev[f"X_{k}"], dev[f"y_{k}"]) for k in ("train", "utility", "gate")}
        tenant_rows = []
        for sc in cfg["scenarios"]:
            for seed in cfg["seeds"]:
                save_arrays = {}
                for split_name, (_, y) in split_arrays.items():
                    parts = dirichlet_partition_indices(y, cfg["tenants"], sc["alpha"], seed, min_size=10 if synthetic_smoke else 50)
                    for t, inds in enumerate(parts):
                        save_arrays[f"{split_name}_t{t}"] = inds
                        row = {"scenario": sc["name"], "alpha": sc["alpha"], "seed": seed, "split": split_name, "tenant": t, "rows": len(inds)}
                        cls_counts = Counter(y[inds].tolist())
                        for lid, lname in enumerate(labels):
                            row[f"class::{lname}"] = int(cls_counts.get(lid, 0))
                        tenant_rows.append(row)
                deterministic_npz(out / "partitions" / f"{sc['name']}_seed{seed}.npz", **save_arrays)
        pd.DataFrame(tenant_rows).sort_values(["scenario", "seed", "split", "tenant"]).to_csv(out / "tenant_counts.csv", index=False)

    eligible_rare = [x for x in labels if x in {"HEARTBLEED", "INFILTRATION"}]
    meta = {
        "protocol": cfg["protocol"], "version": cfg["version"], "amendment": cfg["amendment"], "mode": mode,
        "synthetic_smoke": bool(synthetic_smoke), "config_sha256": config_hash,
        "raw_dataset_files": raw_hashes, "raw_rows": int(sum(counts.values())), "selected_rows": int(len(df)),
        "features": feature_names, "n_features": len(feature_names), "labels": labels, "label_to_id": label_to_id,
        "split_counts": {"train": len(train_idx), "utility": len(utility_idx), "gate": len(gate_idx), "sealed_test": len(test_idx)},
        "preprocessing_fit_scope": "TRAIN_ONLY", "preprocessing": prep,
        "rare_preserve_max_count": cfg["rare_preserve_max_count"], "eligible_rare_classes_present": eligible_rare,
        "sealed_test_sha256": seal["sha256"]
    }
    write_json(out / "data_metadata.json", meta)
    structural_checks = {
        "amendment_recorded": meta["amendment"] == "F0-R1-A1",
        "both_scenarios": len(cfg["scenarios"]) == 2,
        "both_seeds": cfg["seeds"] == [17, 29],
        "sealed_test_created": test_path.exists(),
        "eligible_rare_classes_present": len(eligible_rare) > 0 if synthetic_smoke else all(x in labels for x in ["HEARTBLEED", "INFILTRATION"]),
    }
    structural = {
        "status": "PASS" if all(structural_checks.values()) else "FAIL",
        "mode": mode,
        "checks": structural_checks,
    }
    write_json(out / "R1_1_PASS.json", structural)
    print(f"R1.1 {structural['status']} [{mode}] -> {out}")
    print(f"Selected rows: {len(df):,}; features: {len(feature_names)}; labels: {len(labels)}")
    print(f"Sealed test SHA-256: {seal['sha256']}")
    return meta


class LayerNormMLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: Sequence[int], dropout: float):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(n_features, h1), nn.LayerNorm(h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.LayerNorm(h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def new_model(n_features: int, n_classes: int, cfg: dict, device: torch.device) -> nn.Module:
    return LayerNormMLP(n_features, n_classes, cfg["model"]["hidden"], cfg["model"]["dropout"]).to(device)


def clone_state(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def train_from_state(base_state: Mapping[str, torch.Tensor], X: np.ndarray, y: np.ndarray, cfg: dict,
                     epochs: int, seed: int, device: torch.device, prox_mu: float = 0.0) -> Dict[str, torch.Tensor]:
    set_seed(seed)
    model = new_model(X.shape[1], int(np.max(y)) + 1 if len(y) else 2, cfg, device)
    # n_classes may exceed local labels; derive from state output layer instead.
    out_w = next(v for k, v in base_state.items() if k.endswith("net.8.weight") or k.endswith("8.weight")) if False else None
    # Rebuild using state dimensions when local class set is incomplete.
    n_classes = int(base_state["net.8.weight"].shape[0])
    model = new_model(X.shape[1], n_classes, cfg, device)
    model.load_state_dict({k: v.to(device) for k, v in base_state.items()})
    model.train()
    ds = TensorDataset(torch.from_numpy(np.asarray(X, dtype=np.float32)), torch.from_numpy(np.asarray(y, dtype=np.int64)))
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(ds, batch_size=min(cfg["training"]["batch_size"], max(1, len(ds))), shuffle=True, num_workers=0, generator=gen)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=n_classes).astype(np.float64)
    present = counts > 0
    class_w = np.zeros(n_classes, dtype=np.float32)
    if present.any():
        class_w[present] = float(len(y)) / (float(present.sum()) * counts[present])
        class_w[present] /= max(float(class_w[present].mean()), 1e-12)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float32, device=device))
    base_params = {k: v.detach().to(device) for k, v in base_state.items()} if prox_mu > 0 else None
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if prox_mu > 0:
                prox = torch.zeros((), device=device)
                for name, p in model.named_parameters():
                    prox = prox + torch.sum((p - base_params[name]) ** 2)
                loss = loss + 0.5 * prox_mu * prox
            loss.backward()
            opt.step()
    return clone_state(model.state_dict())


def aggregate_states(states: Sequence[Mapping[str, torch.Tensor]], weights: Sequence[float]) -> Dict[str, torch.Tensor]:
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(states), dtype=float)
    w = w / w.sum()
    out = {}
    for k in states[0]:
        acc = None
        for wi, st in zip(w, states):
            term = st[k].detach().cpu().float() * float(wi)
            acc = term if acc is None else acc + term
        out[k] = acc.to(dtype=states[0][k].dtype)
    return out


def personalized_state(receiver_state: Mapping[str, torch.Tensor], contributor_states: Sequence[Mapping[str, torch.Tensor]],
                       contributor_weights: Sequence[float], external_mass: float) -> Dict[str, torch.Tensor]:
    if external_mass <= 0 or not contributor_states:
        return clone_state(receiver_state)
    cw = np.asarray(contributor_weights, dtype=float)
    if cw.sum() <= 0:
        return clone_state(receiver_state)
    cw = cw / cw.sum() * external_mass
    states = [receiver_state] + list(contributor_states)
    weights = [1.0 - external_mass] + cw.tolist()
    return aggregate_states(states, weights)


def evaluate_state(state: Mapping[str, torch.Tensor], X: np.ndarray, y: np.ndarray, cfg: dict,
                   n_classes: int, benign_id: int, device: torch.device) -> dict:
    if len(y) == 0:
        return {"macro_f1": float("nan"), "accuracy": float("nan"), "far": float("nan"), "worst_class_recall": float("nan"), "per_class_recall": {}, "predictions": np.empty(0, dtype=np.int64)}
    model = new_model(X.shape[1], n_classes, cfg, device)
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    model.eval()
    preds = []
    bs = min(4096, max(1, cfg["training"]["batch_size"] * 2))
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(np.asarray(X[i:i + bs], dtype=np.float32)).to(device)
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
    pred = np.concatenate(preds)
    labels_present = sorted(np.unique(y).tolist())
    macro = float(f1_score(y, pred, labels=labels_present, average="macro", zero_division=0))
    acc = float(accuracy_score(y, pred))
    recalls = recall_score(y, pred, labels=list(range(n_classes)), average=None, zero_division=0)
    present_recalls = [float(recalls[c]) for c in labels_present]
    benign_mask = y == benign_id
    far = float(np.mean(pred[benign_mask] != benign_id)) if benign_mask.any() else 0.0
    return {
        "macro_f1": macro,
        "accuracy": acc,
        "far": far,
        "worst_class_recall": float(min(present_recalls)) if present_recalls else 0.0,
        "per_class_recall": {str(i): float(recalls[i]) for i in range(n_classes)},
        "predictions": pred,
    }


def load_r1_1(artifact_root: str) -> Tuple[dict, dict]:
    r1 = Path(artifact_root) / "r1_1"
    meta = json.loads((r1 / "data_metadata.json").read_text(encoding="utf-8"))
    dev = np.load(r1 / "development_data.npz", allow_pickle=False)
    data = {k: dev[k] for k in dev.files}
    dev.close()
    return meta, data


def partition_file(artifact_root: str, scenario: str, seed: int) -> Path:
    return Path(artifact_root) / "r1_1" / "partitions" / f"{scenario}_seed{seed}.npz"


def load_partitions(artifact_root: str, scenario: str, seed: int) -> Dict[str, np.ndarray]:
    with np.load(partition_file(artifact_root, scenario, seed), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def ensure_override_allowed(meta: dict, name: str, requested: int, frozen: int) -> None:
    if requested != frozen and not meta.get("synthetic_smoke", False):
        raise SystemExit(f"Refusing --{name}={requested} on real F0-R1 data; frozen value is {frozen}.")


def prepare_model_base(meta: dict, cfg: dict, scenario: str, seed: int, device: torch.device) -> Dict[str, torch.Tensor]:
    set_seed(stable_seed("model-init", scenario, seed))
    m = new_model(meta["n_features"], len(meta["labels"]), cfg, device)
    return clone_state(m.state_dict())


def run_r1_2(artifact_root: str, rounds: int | None, local_epochs: int | None) -> dict:
    cfg = load_config(); meta, data = load_r1_1(artifact_root)
    rounds = cfg["training"]["rounds"] if rounds is None else rounds
    local_epochs = cfg["training"]["local_epochs"] if local_epochs is None else local_epochs
    ensure_override_allowed(meta, "rounds", rounds, cfg["training"]["rounds"])
    ensure_override_allowed(meta, "local-epochs", local_epochs, cfg["training"]["local_epochs"])
    out = artifact_dir(artifact_root, "r1_2"); ckpt_dir = out / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    device = get_device(); n_classes = len(meta["labels"]); benign_id = meta["label_to_id"].get("BENIGN", 0)
    rows = []
    for sc in cfg["scenarios"]:
        for seed in cfg["seeds"]:
            parts = load_partitions(artifact_root, sc["name"], seed)
            base = prepare_model_base(meta, cfg, sc["name"], seed, device)
            for rnd in range(rounds):
                local_states, prox_states, train_sizes = [], [], []
                for t in range(cfg["tenants"]):
                    ti = parts[f"train_t{t}"]
                    train_sizes.append(len(ti))
                    ls = train_from_state(base, data["X_train"][ti], data["y_train"][ti], cfg, local_epochs,
                                          stable_seed(seed, sc["name"], rnd, t, "local"), device, 0.0)
                    ps = train_from_state(base, data["X_train"][ti], data["y_train"][ti], cfg, local_epochs,
                                          stable_seed(seed, sc["name"], rnd, t, "prox"), device, cfg["training"]["fedprox_mu"])
                    local_states.append(ls); prox_states.append(ps)
                fedavg = aggregate_states(local_states, train_sizes)
                fedprox = aggregate_states(prox_states, train_sizes)
                for receiver in range(cfg["tenants"]):
                    ui = parts[f"utility_t{receiver}"]
                    for method, st in [("Local-only", local_states[receiver]), ("FedAvg", fedavg), ("FedProx", fedprox)]:
                        ev = evaluate_state(st, data["X_utility"][ui], data["y_utility"][ui], cfg, n_classes, benign_id, device)
                        rows.append({"scenario": sc["name"], "alpha": sc["alpha"], "seed": seed, "round": rnd,
                                     "receiver": receiver, "method": method, "macro_f1": ev["macro_f1"],
                                     "accuracy": ev["accuracy"], "far": ev["far"], "worst_class_recall": ev["worst_class_recall"]})
                torch.save({"scenario": sc["name"], "alpha": sc["alpha"], "seed": seed, "round": rnd,
                            "base_state": base, "local_states": local_states, "fedavg_state": fedavg,
                            "fedprox_state": fedprox, "train_sizes": train_sizes},
                           ckpt_dir / f"{sc['name']}_seed{seed}_round{rnd}.pt")
                base = clone_state(fedavg)
                print(f"R1.2 {sc['name']} seed={seed} round={rnd} complete")
    df = pd.DataFrame(rows)
    # Add aligned deltas per receiver-round.
    pivot = df.pivot_table(index=["scenario", "alpha", "seed", "round", "receiver"], columns="method", values="macro_f1").reset_index()
    pivot["fedavg_loss_vs_local"] = pivot["Local-only"] - pivot["FedAvg"]
    pivot["fedprox_loss_vs_local"] = pivot["Local-only"] - pivot["FedProx"]
    pivot["fedavg_negative_transfer"] = pivot["fedavg_loss_vs_local"] >= cfg["negative_transfer_margin"]
    pivot["fedavg_material_loss"] = pivot["fedavg_loss_vs_local"] >= cfg["material_loss_margin"]
    pivot.to_csv(out / "aligned_comparisons.csv", index=False)
    df.to_csv(out / "metrics.csv", index=False)
    decision_rounds = [r for r in cfg["decision_rounds"] if r < rounds]
    if meta.get("synthetic_smoke") and not decision_rounds:
        decision_rounds = list(range(rounds))
    dec = pivot[pivot["round"].isin(decision_rounds)]
    groups = []
    for (scenario, seed), g in dec.groupby(["scenario", "seed"]):
        groups.append({"scenario": scenario, "seed": int(seed), "cases": len(g),
                       "fedavg_nt_rate": float(g["fedavg_negative_transfer"].mean()) if len(g) else 0.0,
                       "material_cases": int(g["fedavg_material_loss"].sum())})
    summary = {"mode": meta["mode"], "rounds": rounds, "local_epochs": local_epochs, "decision_rounds": decision_rounds,
               "groups": groups, "total_decision_cases": int(len(dec)),
               "material_loss_cases": int(dec["fedavg_material_loss"].sum()) if len(dec) else 0,
               "max_nt_rate": float(max([g["fedavg_nt_rate"] for g in groups], default=0.0)),
               "worst_fedavg_loss": float(dec["fedavg_loss_vs_local"].max()) if len(dec) else 0.0}
    write_json(out / "negative_transfer_summary.json", summary)
    print(f"R1.2 complete -> {out}")
    return summary


def checkpoint_path(artifact_root: str, scenario: str, seed: int, rnd: int) -> Path:
    return Path(artifact_root) / "r1_2" / "checkpoints" / f"{scenario}_seed{seed}_round{rnd}.pt"


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _legacy_run_r1_3_unused(artifact_root: str, rounds: int | None, windows: int | None) -> dict:
    cfg = load_config(); meta, data = load_r1_1(artifact_root)
    r2sum = json.loads((Path(artifact_root) / "r1_2" / "negative_transfer_summary.json").read_text())
    rounds = r2sum["rounds"] if rounds is None else rounds
    windows = cfg["utility_windows"] if windows is None else windows
    if not meta.get("synthetic_smoke"):
        if rounds != cfg["training"]["rounds"]: raise SystemExit("Refusing R1.3 round override on real data.")
        if windows != cfg["utility_windows"]: raise SystemExit("Refusing R1.3 window override on real data.")
    out = artifact_dir(artifact_root, "r1_3"); device = get_device(); n_classes = len(meta["labels"]); benign_id = meta["label_to_id"].get("BENIGN", 0)
    rows = []
    for sc in cfg["scenarios"]:
        for seed in cfg["seeds"]:
            parts = load_partitions(artifact_root, sc["name"], seed)
            for rnd in range(rounds):
                ckpt = load_checkpoint(checkpoint_path(artifact_root, sc["name"], seed, rnd))
                local_states = ckpt["local_states"]
                for receiver in range(cfg["tenants"]):
                    ui_all = np.sort(parts[f"utility_t{receiver}"])
                    win_arrays = {"overall": ui_all}
                    for wi, arr in enumerate(np.array_split(ui_all, windows)):
                        win_arrays[str(wi)] = arr
                    for contributor in range(cfg["tenants"]):
                        if contributor == receiver: continue
                        pair = personalized_state(local_states[receiver], [local_states[contributor]], [1.0], 0.5)
                        for win, ui in win_arrays.items():
                            if len(ui) == 0: continue
                            ev_l = evaluate_state(local_states[receiver], data["X_utility"][ui], data["y_utility"][ui], cfg, n_classes, benign_id, device)
                            ev_p = evaluate_state(pair, data["X_utility"][ui], data["y_utility"][ui], cfg, n_classes, benign_id, device)
                            rows.append({"scenario": sc["name"], "alpha": sc["alpha"], "seed": seed, "round": rnd,
                                         "receiver": receiver, "contributor": contributor, "window": win, "class": "__macro__",
                                         "baseline_metric": ev_l["macro_f1"], "pair_metric": ev_p["macro_f1"], "utility_delta": ev_p["macro_f1"] - ev_l["macro_f1"]})
                            ywin = data["y_utility"][ui]
                            for cid in sorted(np.unique(ywin).tolist()):
                                rows.append({"scenario": sc["name"], "alpha": sc["alpha"], "seed": seed, "round": rnd,
                                             "receiver": receiver, "contributor": contributor, "window": win, "class": meta["labels"][cid],
                                             "baseline_metric": ev_l["per_class_recall"][str(cid)], "pair_metric": ev_p["per_class_recall"][str(cid)],
                                             "utility_delta": ev_p["per_class_recall"][str(cid)] - ev_l["per_class_recall"][str(cid)]})
                print(f"R1.3 {sc['name']} seed={seed} round={rnd} complete")
    df = pd.DataFrame(rows); df.to_csv(out / "utility.csv", index=False)
    macro = df[(df["window"] == "overall") & (df["class"] == "__macro__")]
    scenario_summaries = []
    different = 0; total_receiver_groups = 0
    for sc in cfg["scenarios"]:
        gsc = macro[macro["scenario"] == sc["name"]]
        tops = []
        for (seed, receiver), g in gsc.groupby(["seed", "receiver"]):
            means = g.groupby("contributor")["utility_delta"].mean()
            if len(means):
                top = int(means.idxmax()); tops.append(top); total_receiver_groups += 1
        if tops:
            modal = Counter(tops).most_common(1)[0][0]
            diff_here = sum(t != modal for t in tops); different += diff_here
            top_fraction = diff_here / len(tops)
        else:
            top_fraction = 0.0
        receiver_means = gsc.groupby(["receiver", "contributor"])["utility_delta"].mean().unstack()
        dispersion = float(receiver_means.std(axis=0, ddof=0).mean()) if not receiver_means.empty else 0.0
        recipient_signal = bool((len(set(tops)) >= 2) or dispersion > 1e-4)
        scenario_summaries.append({"scenario": sc["name"], "recipient_signal": recipient_signal,
                                   "top_contributor_difference_fraction": float(top_fraction), "receiver_dispersion": dispersion})
    rare_rows = {lab: int((df["class"] == lab).sum()) for lab in ["HEARTBLEED", "INFILTRATION"] if lab in meta["labels"]}
    overall_diff_fraction = different / total_receiver_groups if total_receiver_groups else 0.0
    summary = {"mode": meta["mode"], "windows": windows, "scenario_summaries": scenario_summaries,
               "receiver_groups": total_receiver_groups, "different_top_contributor_groups": different,
               "different_top_contributor_fraction": float(overall_diff_fraction), "eligible_rare_class_rows": rare_rows}
    write_json(out / "matrix_summary.json", summary)
    write_json(out / "R1_3_PASS.json", {"status": "PASS", "structural_checks": {"utility_rows": len(df) > 0, "both_scenarios": len(scenario_summaries) == 2, "rare_rows_recorded": bool(rare_rows)}})
    print(f"R1.3 complete -> {out}")
    return summary


def safe_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int); s = np.asarray(scores, dtype=float)
    if len(y) < 2 or len(np.unique(y)) < 2: return float("nan")
    return float(roc_auc_score(y, s))


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12: return float("nan")
    return float(spearmanr(x, y).statistic)


def prospective_predictions(macro: pd.DataFrame, scenario: str, seed: int, gamma: float, nt_margin: float) -> pd.DataFrame:
    g = macro[(macro["scenario"] == scenario) & (macro["seed"] == seed)].copy()
    recs = []
    for _, row in g.iterrows():
        hist = g[(g["contributor"] == row["contributor"]) & (g["receiver"] != row["receiver"]) & (g["round"] < row["round"])]
        if hist.empty: continue
        ages = (int(row["round"]) - hist["round"].astype(int)).to_numpy()
        weights = np.power(gamma, np.maximum(ages - 1, 0))
        pred = float(np.average(hist["utility_delta"].to_numpy(dtype=float), weights=weights))
        actual = float(row["utility_delta"])
        recs.append({"scenario": scenario, "seed": int(seed), "round": int(row["round"]), "receiver": int(row["receiver"]),
                     "contributor": int(row["contributor"]), "gamma": gamma, "predicted_utility": pred,
                     "actual_utility": actual, "actual_harm": int(actual <= -nt_margin), "history_rows": int(len(hist))})
    return pd.DataFrame(recs)


def bootstrap_metrics(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    if df.empty: return {"auc_ci95": [None, None], "spearman_ci95": [None, None]}
    rng = np.random.default_rng(seed); aucs=[]; sps=[]; n=len(df)
    for _ in range(n_boot):
        ii = rng.integers(0, n, n); b=df.iloc[ii]
        a=safe_auc(b["actual_harm"], -b["predicted_utility"]); s=safe_spearman(b["predicted_utility"], b["actual_utility"])
        if np.isfinite(a): aucs.append(a)
        if np.isfinite(s): sps.append(s)
    def ci(v):
        if not v: return [None, None]
        return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
    return {"auc_ci95": ci(aucs), "spearman_ci95": ci(sps)}


def metric_pack(df: pd.DataFrame) -> dict:
    auc=safe_auc(df["actual_harm"], -df["predicted_utility"]) if not df.empty else float("nan")
    sp=safe_spearman(df["predicted_utility"], df["actual_utility"]) if not df.empty else float("nan")
    return {"n": int(len(df)), "harm_cases": int(df["actual_harm"].sum()) if len(df) else 0,
            "roc_auc": None if not np.isfinite(auc) else float(auc), "spearman": None if not np.isfinite(sp) else float(sp)}


def _legacy_run_r1_4_unused(artifact_root: str, bootstrap: int | None) -> dict:
    cfg=load_config(); meta,_=load_r1_1(artifact_root)
    bootstrap=cfg["bootstrap"] if bootstrap is None else bootstrap
    ensure_override_allowed(meta, "bootstrap", bootstrap, cfg["bootstrap"])
    out=artifact_dir(artifact_root,"r1_4")
    util=pd.read_csv(Path(artifact_root)/"r1_3"/"utility.csv")
    macro=util[(util["window"].astype(str)=="overall") & (util["class"]=="__macro__")].copy()
    allpred=[]; selection_rows=[]; summaries=[]
    for sc in cfg["scenarios"]:
        candidates=[]
        for gamma in cfg["reliability_gamma"]:
            p=prospective_predictions(macro,sc["name"],17,gamma,cfg["negative_transfer_margin"])
            m=metric_pack(p); candidates.append((gamma,p,m))
            selection_rows.append({"scenario":sc["name"],"gamma":gamma,"n":m["n"],"harm_cases":m["harm_cases"],"roc_auc":m["roc_auc"],"spearman":m["spearman"]})
        def key(item):
            m=item[2]; auc=m["roc_auc"] if m["roc_auc"] is not None else -1; sp=abs(m["spearman"]) if m["spearman"] is not None else -1
            return (auc,sp,-item[0])
        gamma,p_sel,m_sel=max(candidates,key=key)
        p_sel=p_sel.copy(); p_sel["phase"]="selection_seed17"; allpred.append(p_sel)
        p_val=prospective_predictions(macro,sc["name"],29,gamma,cfg["negative_transfer_margin"]); p_val["phase"]="heldout_seed29"; allpred.append(p_val)
        m_val=metric_pack(p_val); ci=bootstrap_metrics(p_val,bootstrap,stable_seed("bootstrap",sc["name"],gamma))
        summaries.append({"scenario":sc["name"],"selected_gamma":gamma,"selection_seed17":m_sel,
                          "heldout_seed29":{**m_val,**ci}})
    pd.DataFrame(selection_rows).to_csv(out/"gamma_selection.csv",index=False)
    pred=pd.concat(allpred,ignore_index=True) if allpred else pd.DataFrame(); pred.to_csv(out/"reliability_predictions.csv",index=False)
    summary={"mode":meta["mode"],"bootstrap":bootstrap,"gamma_candidates":cfg["reliability_gamma"],"scenarios":summaries,
             "history_rule":"same contributor; prior rounds only; current receiving tenant excluded"}
    write_json(out/"reliability_summary.json",summary)
    print(f"R1.4 complete -> {out}")
    return summary


def state_update_vector(local: Mapping[str, torch.Tensor], base: Mapping[str, torch.Tensor]) -> np.ndarray:
    arr=[]
    for k in sorted(local):
        if torch.is_floating_point(local[k]): arr.append((local[k].float()-base[k].float()).reshape(-1).numpy())
    return np.concatenate(arr) if arr else np.zeros(1,dtype=np.float32)


def cosine(a: np.ndarray,b: np.ndarray) -> float:
    d=float(np.linalg.norm(a)*np.linalg.norm(b))
    return 0.0 if d<1e-12 else float(np.dot(a,b)/d)


def weighted_cvar(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    v=np.asarray(values,dtype=float); w=np.asarray(weights,dtype=float)
    if len(v)==0 or w.sum()<=0: return 0.0
    w=w/w.sum(); order=np.argsort(v); v=v[order]; w=w[order]; c=np.cumsum(w)
    mask=c>=q
    if not mask.any(): return float(v[-1])
    tail_w=w[mask]; return float(np.average(v[mask],weights=tail_w)) if tail_w.sum()>0 else float(v[-1])


def selected_gamma_map(summary: dict) -> Dict[str,float]:
    return {x["scenario"]:float(x["selected_gamma"]) for x in summary["scenarios"]}


def prediction_map(pred: pd.DataFrame) -> Dict[Tuple[str,int,int,int,int],float]:
    d={}
    for _,r in pred.iterrows(): d[(str(r.scenario),int(r.seed),int(r["round"]),int(r.receiver),int(r.contributor))]=float(r.predicted_utility)
    return d


def candidate_from_scores(receiver:int, local_states, base_state, pred_map, scenario, seed, rnd, cfg):
    rv=state_update_vector(local_states[receiver],base_state); raw=[]
    for c in range(cfg["tenants"]):
        if c==receiver: continue
        sim=cosine(rv,state_update_vector(local_states[c],base_state)); pu=float(pred_map.get((scenario,seed,rnd,receiver,c),0.0))
        reliability=max(0.0,pu+cfg["negative_transfer_margin"])
        score=max(0.0,sim)*reliability
        raw.append({"contributor":c,"similarity":sim,"predicted_utility":pu,"score":score})
    positive=[x for x in raw if x["score"]>0]
    if not positive: return clone_state(local_states[receiver]),raw,0.0,0.0
    scores=np.array([x["score"] for x in positive],dtype=float); scores=scores/scores.sum()
    ext=cfg["risk"]["budget"]
    risks=[max(0.0,-x["predicted_utility"]) for x in positive]
    mean_risk=float(np.average(risks,weights=scores)); cvar=weighted_cvar(risks,scores,cfg["risk"]["q"])
    risk_score=mean_risk+cfg["risk"]["cvar_lambda"]*cvar
    state=personalized_state(local_states[receiver],[local_states[x["contributor"]] for x in positive],scores,ext)
    for x,w in zip(positive,scores): x["normalized_external_weight"]=float(w*ext)
    return state,raw,ext,risk_score


def guardrail_accept(local_ev:dict,cand_ev:dict,risk_score:float,cfg:dict)->bool:
    tol=cfg["gate_tolerances"]
    return bool(risk_score<=cfg["risk"]["budget"] and
                cand_ev["macro_f1"]>=local_ev["macro_f1"]+tol["f1_margin"] and
                cand_ev["far"]<=local_ev["far"]+tol["far"] and
                cand_ev["worst_class_recall"]>=local_ev["worst_class_recall"]-tol["worst_class_recall"])


def _legacy_run_r1_5_unused(artifact_root: str, rounds: int | None, gate_bootstrap: int | None) -> dict:
    cfg=load_config(); meta,data=load_r1_1(artifact_root)
    frozen_rounds=len(cfg["decision_rounds"]); rounds=frozen_rounds if rounds is None else rounds
    gate_bootstrap=cfg["bootstrap"] if gate_bootstrap is None else gate_bootstrap
    if not meta.get("synthetic_smoke"):
        if rounds!=frozen_rounds: raise SystemExit("Refusing R1.5 --rounds override on real data.")
        if gate_bootstrap!=cfg["bootstrap"]: raise SystemExit("Refusing R1.5 --gate-bootstrap override on real data.")
    r4sum=json.loads((Path(artifact_root)/"r1_4"/"reliability_summary.json").read_text()); pred=pd.read_csv(Path(artifact_root)/"r1_4"/"reliability_predictions.csv")
    pmap=prediction_map(pred); out=artifact_dir(artifact_root,"r1_5"); device=get_device(); n_classes=len(meta["labels"]); benign_id=meta["label_to_id"].get("BENIGN",0)
    r2sum=json.loads((Path(artifact_root)/"r1_2"/"negative_transfer_summary.json").read_text())
    available_rounds=r2sum["rounds"]
    decision=[r for r in cfg["decision_rounds"] if r<available_rounds]
    if meta.get("synthetic_smoke"):
        decision=list(range(max(0,available_rounds-rounds),available_rounds))
    else:
        decision=decision[:rounds]
    util=pd.read_csv(Path(artifact_root)/"r1_3"/"utility.csv")
    macro_util=util[(util["window"].astype(str)=="overall") & (util["class"]=="__macro__")]
    gate_rows=[]; metric_rows=[]; weight_rows=[]
    for sc in cfg["scenarios"]:
        for seed in cfg["seeds"]:
            parts=load_partitions(artifact_root,sc["name"],seed)
            for rnd in decision:
                ck=load_checkpoint(checkpoint_path(artifact_root,sc["name"],seed,rnd)); local_states=ck["local_states"]; base=ck["base_state"]
                for receiver in range(cfg["tenants"]):
                    ui=parts[f"utility_t{receiver}"]; gi=parts[f"gate_t{receiver}"]
                    local_u=evaluate_state(local_states[receiver],data["X_utility"][ui],data["y_utility"][ui],cfg,n_classes,benign_id,device)
                    local_g=evaluate_state(local_states[receiver],data["X_gate"][gi],data["y_gate"][gi],cfg,n_classes,benign_id,device)
                    cand,raw,ext,risk_score=candidate_from_scores(receiver,local_states,base,pmap,sc["name"],seed,rnd,cfg)
                    cand_u=evaluate_state(cand,data["X_utility"][ui],data["y_utility"][ui],cfg,n_classes,benign_id,device)
                    accept=guardrail_accept(local_u,cand_u,risk_score,cfg) and ext>=cfg["nontrivial"]["external_weight_mass"]
                    cand_g=evaluate_state(cand,data["X_gate"][gi],data["y_gate"][gi],cfg,n_classes,benign_id,device)
                    dep_g=cand_g if accept else local_g
                    fed_g=evaluate_state(ck["fedavg_state"],data["X_gate"][gi],data["y_gate"][gi],cfg,n_classes,benign_id,device)
                    pred_change=float(np.mean(cand_g["predictions"]!=local_g["predictions"])) if len(gi) else 0.0
                    # Diagnostic FedFomo-screening: current pair utility on utility split.
                    ug=macro_util[(macro_util.scenario==sc["name"])&(macro_util.seed==seed)&(macro_util["round"]==rnd)&(macro_util.receiver==receiver)]
                    fomo_scores=[]; fomo_states=[]
                    for c in range(cfg["tenants"]):
                        if c==receiver: continue
                        rr=ug[ug.contributor==c]; s=max(0.0,float(rr.utility_delta.iloc[0])) if len(rr) else 0.0
                        if s>0: fomo_scores.append(s); fomo_states.append(local_states[c])
                    fomo=personalized_state(local_states[receiver],fomo_states,fomo_scores,cfg["risk"]["budget"]) if fomo_scores else clone_state(local_states[receiver])
                    fomo_g=evaluate_state(fomo,data["X_gate"][gi],data["y_gate"][gi],cfg,n_classes,benign_id,device)
                    # Diagnostic pFedGraph-core: similarity-only external weighting.
                    rv=state_update_vector(local_states[receiver],base); sims=[]; simstates=[]
                    for c in range(cfg["tenants"]):
                        if c==receiver: continue
                        s=max(0.0,cosine(rv,state_update_vector(local_states[c],base)))
                        if s>0: sims.append(s); simstates.append(local_states[c])
                    pgraph=personalized_state(local_states[receiver],simstates,sims,cfg["risk"]["budget"]) if sims else clone_state(local_states[receiver])
                    pgraph_g=evaluate_state(pgraph,data["X_gate"][gi],data["y_gate"][gi],cfg,n_classes,benign_id,device)
                    # Oracle headroom: best truth-known 20% single-contributor candidate or local on held-out gate.
                    oracle_ev=local_g; oracle_name="Local-only"
                    for c in range(cfg["tenants"]):
                        if c==receiver: continue
                        st=personalized_state(local_states[receiver],[local_states[c]],[1.0],cfg["risk"]["budget"])
                        ev=evaluate_state(st,data["X_gate"][gi],data["y_gate"][gi],cfg,n_classes,benign_id,device)
                        if ev["macro_f1"]>oracle_ev["macro_f1"]:
                            oracle_ev=ev; oracle_name=f"pair_t{c}"
                    candidate_delta=cand_g["macro_f1"]-local_g["macro_f1"]
                    harmful=candidate_delta<=-cfg["negative_transfer_margin"]
                    beneficial=candidate_delta>0.0
                    nontrivial=bool(pred_change>=cfg["nontrivial"]["prediction_change"] and ext>=cfg["nontrivial"]["external_weight_mass"])
                    case={"scenario":sc["name"],"alpha":sc["alpha"],"seed":seed,"round":rnd,"receiver":receiver,
                          "local_f1":local_g["macro_f1"],"candidate_f1":cand_g["macro_f1"],"deployed_f1":dep_g["macro_f1"],"fedavg_f1":fed_g["macro_f1"],
                          "oracle_f1":oracle_ev["macro_f1"],"oracle_choice":oracle_name,"candidate_delta":candidate_delta,
                          "accepted":bool(accept),"harmful":bool(harmful),"beneficial":bool(beneficial),"risk_score":risk_score,
                          "external_weight_mass":ext,"prediction_change_fraction":pred_change,"nontrivial":nontrivial,
                          "utility_local_f1":local_u["macro_f1"],"utility_candidate_f1":cand_u["macro_f1"],
                          "fedavg_negative_transfer":bool(fed_g["macro_f1"]<=local_g["macro_f1"]-cfg["negative_transfer_margin"])}
                    gate_rows.append(case)
                    for method,ev in [("Local-only",local_g),("FedAvg",fed_g),("RUTRA-candidate",cand_g),("RUTRA-deployed",dep_g),("Oracle",oracle_ev),("FedFomo-screening",fomo_g),("pFedGraph-core-screening",pgraph_g)]:
                        metric_rows.append({"scenario":sc["name"],"seed":seed,"round":rnd,"receiver":receiver,"method":method,"macro_f1":ev["macro_f1"],"accuracy":ev["accuracy"],"far":ev["far"],"worst_class_recall":ev["worst_class_recall"]})
                    for rr in raw:
                        weight_rows.append({"scenario":sc["name"],"seed":seed,"round":rnd,"receiver":receiver,**rr,
                                            "external_weight_mass":ext,"risk_score":risk_score})
                print(f"R1.5 {sc['name']} seed={seed} round={rnd} complete")
    gate=pd.DataFrame(gate_rows); metrics=pd.DataFrame(metric_rows); weights=pd.DataFrame(weight_rows)
    gate.to_csv(out/"gate.csv",index=False); metrics.to_csv(out/"metrics.csv",index=False); weights.to_csv(out/"weights.csv",index=False)
    harmful=int(gate.harmful.sum()); rejected_harm=int(((gate.harmful)&(~gate.accepted)).sum()); beneficial=int(gate.beneficial.sum()); accepted_ben=int(((gate.beneficial)&(gate.accepted)).sum())
    deployed_harm=int(((gate.harmful)&(gate.accepted)).sum())
    rutra_nt_red=float(1-deployed_harm/harmful) if harmful else 0.0
    fed_harm=int(gate.fedavg_negative_transfer.sum())
    oracle_harm=int(((gate.oracle_f1<=gate.local_f1-cfg["negative_transfer_margin"])) .sum())
    oracle_nt_red=float(1-oracle_harm/fed_harm) if fed_harm else 0.0
    head_by_tenant={int(r):float(np.mean(g["oracle_f1"]-g["local_f1"])) for r,g in gate.groupby("receiver")}
    worst_head=float(min(head_by_tenant.values())) if head_by_tenant else 0.0
    # Case-level bootstrap intervals for the key gate rates/headroom.
    rng=np.random.default_rng(stable_seed("r1_5_bootstrap",gate_bootstrap)); boot_hr=[]; boot_ba=[]; boot_head=[]
    if len(gate):
        for _ in range(gate_bootstrap):
            b=gate.iloc[rng.integers(0,len(gate),len(gate))]
            bh=int(b.harmful.sum()); bb=int(b.beneficial.sum())
            if bh: boot_hr.append(float(((b.harmful)&(~b.accepted)).sum()/bh))
            if bb: boot_ba.append(float(((b.beneficial)&(b.accepted)).sum()/bb))
            hb=[float(np.mean(g["oracle_f1"]-g["local_f1"])) for _,g in b.groupby("receiver")]
            if hb: boot_head.append(float(min(hb)))
    def _ci(vals):
        return [float(np.quantile(vals,0.025)),float(np.quantile(vals,0.975))] if vals else [None,None]
    summary={"mode":meta["mode"],"decision_rounds":decision,"receiver_round_cases":int(len(gate)),"gate_bootstrap":gate_bootstrap,
             "harmful_cases":harmful,"harmful_rejected":rejected_harm,"harmful_rejection_rate":float(rejected_harm/harmful) if harmful else 0.0,
             "harmful_rejection_ci95":_ci(boot_hr),"rutra_nt_reduction":rutra_nt_red,"beneficial_cases":beneficial,"beneficial_accepted":accepted_ben,
             "beneficial_acceptance_rate":float(accepted_ben/beneficial) if beneficial else 0.0,"beneficial_acceptance_ci95":_ci(boot_ba),
             "accepted_nontrivial_cases":int(((gate.accepted)&(gate.nontrivial)).sum()),
             "fedavg_harmful_cases":fed_harm,"oracle_nt_reduction":oracle_nt_red,"worst_tenant_headroom":worst_head,"worst_tenant_headroom_ci95":_ci(boot_head),
             "headroom_by_tenant":head_by_tenant,"screening_note":"FedFomo-screening and pFedGraph-core-screening are diagnostics only."}
    write_json(out/"oracle_gate_summary.json",summary)
    print(f"R1.5 complete -> {out}; cases={len(gate)}")
    return summary


def scientific_decision_checks(artifact_root: str, cfg: dict) -> Tuple[dict,dict]:
    r1=json.loads((Path(artifact_root)/"r1_1"/"R1_1_PASS.json").read_text()); r2=json.loads((Path(artifact_root)/"r1_2"/"negative_transfer_summary.json").read_text())
    r3=json.loads((Path(artifact_root)/"r1_3"/"matrix_summary.json").read_text()); r4=json.loads((Path(artifact_root)/"r1_4"/"reliability_summary.json").read_text()); r5=json.loads((Path(artifact_root)/"r1_5"/"oracle_gate_summary.json").read_text())
    th=cfg["decision_thresholds"]
    r2ok=(r2["max_nt_rate"]>=th["r1_2_nt_rate"] or r2["material_loss_cases"]>=th["r1_2_material_cases"])
    scenario_signal=all(x["recipient_signal"] for x in r3["scenario_summaries"])
    rare_ok=all(r3["eligible_rare_class_rows"].get(x,0)>0 for x in ["HEARTBLEED","INFILTRATION"])
    r3ok=scenario_signal and r3["different_top_contributor_fraction"]>=th["r1_3_receiver_top_contributor_fraction"] and rare_ok
    r4ok=False
    for s in r4["scenarios"]:
        h=s["heldout_seed29"]; auc=h.get("roc_auc"); sp=h.get("spearman")
        if (auc is not None and auc>=th["r1_4_auc"]) or (sp is not None and abs(sp)>=th["r1_4_spearman_abs"]): r4ok=True
    oracle_ok=(r5["worst_tenant_headroom"]>=th["r1_5_worst_tenant_headroom"] or r5["oracle_nt_reduction"]>=th["r1_5_oracle_nt_reduction"])
    safety_ok=(r5["rutra_nt_reduction"]>=th["r1_5_rutra_nt_reduction"] and r5["harmful_cases"]>=th["r1_5_min_harmful"] and r5["harmful_rejection_rate"]>=th["r1_5_harmful_rejection"])
    usefulness_ok=(r5["beneficial_cases"]>=th["r1_5_min_beneficial"] and r5["beneficial_acceptance_rate"]>=th["r1_5_beneficial_acceptance"] and r5["accepted_nontrivial_cases"]>=th["r1_5_min_nontrivial_accepted"])
    power_ok=r5["receiver_round_cases"]>=th["r1_5_min_cases"]
    checks={"R1.1":bool(r1.get("status")=="PASS"),"R1.2":bool(r2ok),"R1.3":bool(r3ok),"R1.4":bool(r4ok),
            "R1.5_oracle":bool(oracle_ok),"R1.5_safety":bool(safety_ok),"R1.5_usefulness":bool(usefulness_ok),"R1.5_power":bool(power_ok)}
    values={"r1_2":r2,"r1_3":r3,"r1_4":r4,"r1_5":r5}
    return checks,values


def verify_system5(main_r1: str, system5_r1: str, output: str) -> bool:
    names=["data_metadata.json","sampling_audit.csv","tenant_counts.csv","untouched_final_test_R1.SEAL.json"]
    lines=[]; ok=True
    for n in names:
        a=Path(main_r1)/n; b=Path(system5_r1)/n
        if not a.exists() or not b.exists():
            lines.append(f"{n}: MISSING"); ok=False; continue
        ha,hb=sha256_file(a),sha256_file(b); match=ha==hb; ok &= match
        lines.append(f"{n}: {'MATCH' if match else 'MISMATCH'}  main={ha}  system5={hb}")
    lines.append(f"status: {'PASS' if ok else 'FAIL'}")
    Path(output).parent.mkdir(parents=True,exist_ok=True); Path(output).write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines)); return ok


def audit_checks(environment_lock: str, terminal_log_dir: str, system5_reproduction: str, git_commit: str,
                 confirm_test_unopened: bool, confirm_thresholds_unchanged: bool, confirm_failures_preserved: bool) -> dict:
    logdir=Path(terminal_log_dir)
    required_logs=[f"R1_{i}_system6.log" for i in range(1,6)]
    logs_ok=all((logdir/x).exists() and (logdir/x).stat().st_size>0 for x in required_logs)
    s5=Path(system5_reproduction); s5ok=s5.exists() and "status: PASS" in s5.read_text(encoding="utf-8",errors="ignore")
    return {"environment_lock_present":Path(environment_lock).exists(),"r1_1_to_r1_5_logs_present":logs_ok,
            "system5_reproduction_pass":s5ok,"git_commit_40_hex":bool(len(git_commit)==40 and all(c in "0123456789abcdefABCDEF" for c in git_commit)),
            "test_unopened_declaration":confirm_test_unopened,"thresholds_unchanged_declaration":confirm_thresholds_unchanged,
            "failures_preserved_declaration":confirm_failures_preserved}


def copy_if_exists(src: Path, dst_root: Path, arcname: str, manifest: List[str]) -> None:
    if src.exists() and src.is_file():
        dest=dst_root/arcname; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dest); manifest.append(arcname)


def run_r1_6(environment_lock: str, terminal_log_dir: str, system5_reproduction: str, git_commit: str,
             confirm_test_unopened: bool, confirm_thresholds_unchanged: bool, confirm_failures_preserved: bool) -> dict:
    cfg=load_config(); artifact_root="artifacts/f0_r1"; out=artifact_dir(artifact_root,"r1_6")
    meta=json.loads((Path(artifact_root)/"r1_1"/"data_metadata.json").read_text())
    if meta.get("synthetic_smoke"): raise SystemExit("R1.6 is prohibited on synthetic smoke data.")
    sci,values=scientific_decision_checks(artifact_root,cfg)
    audit=audit_checks(environment_lock,terminal_log_dir,system5_reproduction,git_commit,confirm_test_unopened,confirm_thresholds_unchanged,confirm_failures_preserved)
    essential=["R1.2","R1.3","R1.4","R1.5_oracle","R1.5_safety","R1.5_usefulness","R1.5_power"]
    scientific_fail=sum(not sci[x] for x in essential)
    all_audit=all(audit.values()); all_sci=all(sci.values())
    if all_sci and all_audit: recommendation="GREEN"
    elif scientific_fail>=2: recommendation="RED"
    else: recommendation="CONDITIONAL"
    result={"protocol":cfg["protocol"],"amendment":cfg["amendment"],"recommendation":recommendation,
            "scientific_checks":sci,"audit_checks":audit,"essential_scientific_failures":scientific_fail,
            "supervisor_authority":"Automated recommendation does not authorize Phase 1; supervisor signature is required."}
    write_json(out/"recommendation.json",result)
    report=["# XFL-CloudIDS F0-R1 Gate Report","",f"Automated recommendation: **{recommendation}**","",
            "This report is an automated technical recommendation only. Phase 1 requires the supervisor's signed decision.","","## Scientific checks"]
    report += [f"- {k}: {'PASS' if v else 'FAIL'}" for k,v in sci.items()]
    report += ["","## Audit checks"]+[f"- {k}: {'PASS' if v else 'FAIL'}" for k,v in audit.items()]
    report += ["","## Supervisor decision","GREEN / CONDITIONAL / RED","","Reason: ______________________________","","Signature: ____________________________   Date: ____________",""]
    (out/"gate_report.md").write_text("\n".join(report),encoding="utf-8")
    staging=out/"supervisor_zip_staging"; shutil.rmtree(staging,ignore_errors=True); staging.mkdir(parents=True)
    manifest=[]
    # Frozen config and audit files.
    copy_if_exists(CONFIG_PATH,staging,"f0_config.json",manifest)
    copy_if_exists(Path(environment_lock),staging,"evidence/requirements-f0-r1-lock.txt",manifest)
    copy_if_exists(Path(system5_reproduction),staging,"evidence/system5_reproduction.txt",manifest)
    for n in ["git_commit.txt","f0_config.sha256","raw_cicids2017.sha256"]: copy_if_exists(ROOT/"evidence"/n,staging,f"evidence/{n}",manifest)
    for n in ["R1_1_PASS.json","data_metadata.json","sampling_audit.csv","tenant_counts.csv","untouched_final_test_R1.SEAL.json"]: copy_if_exists(Path(artifact_root)/"r1_1"/n,staging,f"r1_1/{n}",manifest)
    for n in ["negative_transfer_summary.json","metrics.csv","aligned_comparisons.csv"]: copy_if_exists(Path(artifact_root)/"r1_2"/n,staging,f"r1_2/{n}",manifest)
    for n in ["matrix_summary.json","utility.csv","R1_3_PASS.json"]: copy_if_exists(Path(artifact_root)/"r1_3"/n,staging,f"r1_3/{n}",manifest)
    for n in ["reliability_summary.json","reliability_predictions.csv","gamma_selection.csv"]: copy_if_exists(Path(artifact_root)/"r1_4"/n,staging,f"r1_4/{n}",manifest)
    for n in ["oracle_gate_summary.json","metrics.csv","gate.csv","weights.csv"]: copy_if_exists(Path(artifact_root)/"r1_5"/n,staging,f"r1_5/{n}",manifest)
    copy_if_exists(out/"recommendation.json",staging,"r1_6/recommendation.json",manifest); copy_if_exists(out/"gate_report.md",staging,"r1_6/gate_report.md",manifest)
    logdir=Path(terminal_log_dir)
    for p in sorted(logdir.glob("*.log")):
        if p.name=="R1_6_system6.log": continue
        copy_if_exists(p,staging,f"evidence/terminal_logs/{p.name}",manifest)
    # Safety: refuse restricted artifacts.
    forbidden=[]
    for p in staging.rglob("*"):
        if p.is_file() and (p.suffix.lower() in {".npz",".pt",".pth"} or "checkpoint" in p.name.lower()): forbidden.append(str(p))
    if forbidden: raise RuntimeError(f"Forbidden artifacts entered supervisor staging: {forbidden}")
    zip_path=out/"F0_R1_RESULTS_FOR_SUPERVISOR.zip"
    with zipfile.ZipFile(zip_path,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
        for p in sorted(staging.rglob("*")):
            if p.is_file(): zf.write(p,p.relative_to(staging).as_posix())
    sha=sha256_file(zip_path); sha_path=out/"F0_R1_RESULTS_FOR_SUPERVISOR.sha256"; sha_path.write_text(f"{sha}  {zip_path.name}\n",encoding="utf-8")
    write_json(out/"package_manifest.json",{"files":manifest,"zip_sha256":sha})
    shutil.rmtree(staging,ignore_errors=True)
    print(f"R1.6 recommendation: {recommendation}")
    print(f"Supervisor ZIP: {zip_path}")
    print(f"SHA-256: {sha}")
    return result
