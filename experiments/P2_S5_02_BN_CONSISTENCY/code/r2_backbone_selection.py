from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors

from common import (
    classification_metrics,
    dump_json,
    load_manifest,
    pool_role,
    seed_all,
    system_environment,
    validation_duplicate_mask,
    write_csv,
)


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


class CNN1DTabular(nn.Module):
    def __init__(self, dim: int, classes: int, dropout: float):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.head(self.features(x))


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = float(gamma)

    def forward(self, logits, target):
        logp = torch.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        idx = torch.arange(target.shape[0], device=target.device)
        logpt = logp[idx, target]
        pt = p[idx, target]
        at = self.alpha[target]
        return (-at * (1.0 - pt).pow(self.gamma) * logpt).mean()


def stratified_internal_split(y: np.ndarray, seed: int, frac: float) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for c in range(int(y.max()) + 1):
        idx = np.where(y == c)[0].copy()
        rng.shuffle(idx)
        if len(idx) >= 20:
            n_val = max(1, int(round(float(frac) * len(idx))))
            val_idx.extend(idx[:n_val].tolist())
            train_idx.extend(idx[n_val:].tolist())
        else:
            train_idx.extend(idx.tolist())
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def stable_class_ids(y: np.ndarray, n_classes: int, min_support: int) -> List[int]:
    counts = np.bincount(y.astype(np.int64), minlength=n_classes)
    return [int(i) for i, n in enumerate(counts) if int(n) >= int(min_support)]


def stable_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: List[int]) -> dict:
    if not classes:
        raise ValueError("No stable classes available for metric calculation")
    labels = np.asarray(classes, dtype=np.int64)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "stable_classes": [int(x) for x in classes],
        "stable_macro_f1": float(np.mean(f)),
        "worst_stable_recall": float(np.min(r)),
        "stable_per_class": [
            {"class_id": int(c), "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(classes)
        ],
    }


def bounded_smote(X: np.ndarray, y: np.ndarray, n_classes: int, target: int, kmax: int, seed: int):
    rng = np.random.default_rng(seed)
    xs = [X]
    ys = [y]
    audit = []
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        n = int(len(idx))
        need = max(0, int(target) - n)
        row = {"class_id": c, "original_rows": n, "synthetic_rows": 0, "target": int(target)}
        if need == 0:
            audit.append(row)
            continue
        Xc = X[idx]
        if n == 0:
            row["note"] = "class absent; no synthetic generation"
            audit.append(row)
            continue
        if n == 1:
            synth = np.repeat(Xc, need, axis=0)
            row["note"] = "single-row deterministic duplication fallback"
        else:
            k = min(int(kmax), n - 1)
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(Xc)
            neigh = nbrs.kneighbors(Xc, return_distance=False)[:, 1:]
            base_pos = rng.integers(0, n, size=need)
            neigh_col = rng.integers(0, k, size=need)
            other_pos = neigh[base_pos, neigh_col]
            lam = rng.random(need, dtype=np.float32).reshape(-1, 1)
            synth = Xc[base_pos] + lam * (Xc[other_pos] - Xc[base_pos])
            row["k_neighbors"] = int(k)
        xs.append(np.asarray(synth, dtype=np.float32))
        ys.append(np.full(need, c, dtype=np.int64))
        row["synthetic_rows"] = int(need)
        audit.append(row)
    Xo = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
    yo = np.concatenate(ys, axis=0).astype(np.int64, copy=False)
    order = rng.permutation(len(yo))
    return Xo[order], yo[order], audit


def inverse_sqrt_alpha(y: np.ndarray, n_classes: int) -> Tuple[np.ndarray, List[dict]]:
    counts = np.bincount(y.astype(np.int64), minlength=n_classes).astype(np.float64)
    if np.any(counts <= 0):
        raise ValueError("Focal-loss calibration requires every class in optimizer training rows")
    raw = 1.0 / np.sqrt(counts)
    raw = raw / raw.mean()
    weights = np.clip(raw, 0.25, 8.0).astype(np.float32)
    audit = [{"class_id": int(i), "rows": int(counts[i]), "alpha": float(weights[i])} for i in range(n_classes)]
    return weights, audit


def predict(model, X: np.ndarray, device: str, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device, non_blocking=True)
            out.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(out)


def make_model(candidate: dict, dim: int, classes: int, dropout: float):
    if candidate["architecture"].startswith("MLP_"):
        return ModernTabularMLP(dim, classes, dropout)
    if candidate["architecture"].startswith("CNN1D_"):
        return CNN1DTabular(dim, classes, dropout)
    raise ValueError(f"Unknown architecture: {candidate['architecture']}")


def train_candidate(candidate: dict, spec: dict, ds, Xopt, yopt, Xes, yes, Xreport, clean_mask, yreport, out_dir: Path, device: str):
    seed = int(spec["calibration_seed"])
    cfg = spec["shared_training"]
    seed_all(seed)

    train_X, train_y = Xopt, yopt
    smote_audit = []
    if candidate["imbalance"] == "bounded_smote":
        sm = spec["smote"]
        train_X, train_y, smote_audit = bounded_smote(
            Xopt, yopt, ds.n_classes, int(sm["target_per_class"]), int(sm["k_neighbors_max"]), seed + 101
        )
        write_csv(out_dir / f"{candidate['id']}_SMOTE_AUDIT.csv", smote_audit)

    alpha_audit = []
    alpha_tensor = None
    if candidate["loss"] == "focal_gamma_2":
        alpha, alpha_audit = inverse_sqrt_alpha(yopt, ds.n_classes)
        write_csv(out_dir / f"{candidate['id']}_FOCAL_ALPHA.csv", alpha_audit)
        alpha_tensor = torch.from_numpy(alpha).to(device)

    dataset = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=gen,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    model = make_model(candidate, ds.feature_dim, ds.n_classes, float(cfg["dropout"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    if candidate["loss"] == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif candidate["loss"] == "focal_gamma_2":
        criterion = FocalLoss(alpha_tensor, gamma=float(spec["focal"]["gamma"]))
    else:
        raise ValueError(candidate["loss"])

    es_stable = stable_class_ids(yes, ds.n_classes, int(spec["reporting"]["stable_class_support_min"]))
    history = []
    best_score = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    t0 = time.perf_counter()

    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in {candidate['id']} epoch {epoch}")
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(yb)
            total_rows += len(yb)

        es_pred = predict(model, Xes, device)
        es_macro = float(f1_score(yes, es_pred, labels=np.asarray(es_stable), average="macro", zero_division=0))
        es_weighted = float(f1_score(yes, es_pred, average="weighted", zero_division=0))
        row = {
            "candidate": candidate["id"],
            "epoch": epoch,
            "train_loss": total_loss / max(total_rows, 1),
            "early_stop_stable_macro_f1": es_macro,
            "early_stop_weighted_f1": es_weighted,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        if es_macro > best_score + 1e-6:
            best_score = es_macro
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if epoch >= int(cfg["min_epochs"]) and stale >= int(cfg["early_stopping_patience"]):
            break

    fit_seconds = time.perf_counter() - t0
    if best_state is None:
        raise AssertionError("No best state captured")
    model.load_state_dict(best_state)
    ypred = predict(model, Xreport, device)

    primary = classification_metrics(yreport[clean_mask], ypred[clean_mask], ds.n_classes, ds.benign_label)
    secondary = classification_metrics(yreport, ypred, ds.n_classes, ds.benign_label)
    stable_ids = stable_class_ids(yreport[clean_mask], ds.n_classes, int(spec["reporting"]["stable_class_support_min"]))
    sm = stable_metrics(yreport[clean_mask], ypred[clean_mask], stable_ids)
    gate = spec["selection_gate"]
    checks = {
        "accuracy": primary["accuracy"] >= float(gate["accuracy_min"]),
        "weighted_f1": primary["weighted_f1"] >= float(gate["weighted_f1_min"]),
        "stable_macro_f1": sm["stable_macro_f1"] >= float(gate["stable_macro_f1_min"]),
        "far": primary["far"] <= float(gate["far_max"]),
        "worst_stable_recall": sm["worst_stable_recall"] >= float(gate["worst_stable_recall_min"]),
    }
    result = {
        "candidate": candidate,
        "optimizer_training_rows_original": int(len(yopt)),
        "optimizer_training_rows_after_imbalance_recipe": int(len(train_y)),
        "best_epoch": int(best_epoch),
        "best_internal_stable_macro_f1": float(best_score),
        "fit_seconds": float(fit_seconds),
        "primary_deduplicated_validation": primary,
        "secondary_raw_validation": secondary,
        "stable_metrics": sm,
        "gate_thresholds": gate,
        "gate_checks": checks,
        "candidate_pass": bool(all(checks.values())),
        "sealed_final_test_access": False,
    }
    dump_json(out_dir / f"{candidate['id']}_METRICS.json", result)
    write_csv(out_dir / f"{candidate['id']}_TRAINING_HISTORY.csv", history)
    per_class_rows = []
    for scope, metrics in [("deduplicated_validation", primary), ("raw_validation", secondary)]:
        for r in metrics["per_class"]:
            per_class_rows.append({"candidate": candidate["id"], "scope": scope, **r})
    write_csv(out_dir / f"{candidate['id']}_PER_CLASS.csv", per_class_rows)

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--system", choices=["SYSTEM6"], default="SYSTEM6")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    spec = json.loads((root / "R2_SPECIFICATION.json").read_text(encoding="utf-8"))
    reference = json.loads((root / "REFERENCE_R1_SYSTEM6.json").read_text(encoding="utf-8"))
    bridge = Path(args.bridge).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    seed = int(spec["calibration_seed"])
    seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("STOP: R2 System 6 neural selection requires CUDA. Do not silently switch to CPU.")
    device = "cuda"

    env = system_environment()
    env.update({"step": "R2-S6", "system": "SYSTEM6", "role": "bounded_neural_backbone_imbalance_selection"})
    dump_json(out / "R2_SYSTEM6_ENVIRONMENT.json", env)

    ds = load_manifest(bridge / spec["canonical_manifest"], bridge, allowed_roles=("train", "validation"))
    train = pool_role(ds, "train")
    validation = pool_role(ds, "validation")
    if ds.feature_dim != int(spec["expected_feature_dim"]) or ds.n_classes != int(spec["expected_n_classes"]):
        raise RuntimeError("STOP: canonical development bridge dimensions do not match frozen R2 specification")
    if len(train.y) != int(spec["expected_bridge_rows"]["train"]) or len(validation.y) != int(spec["expected_bridge_rows"]["validation"]):
        raise RuntimeError("STOP: canonical development bridge row counts do not match frozen R2 specification")

    dup = validation_duplicate_mask(train, validation)
    clean = ~dup
    if int(clean.sum()) == 0:
        raise RuntimeError("STOP: no deduplicated validation rows remain")

    tr_idx, es_idx = stratified_internal_split(train.y, seed, float(spec["shared_training"]["internal_early_stop_fraction"]))
    if len(es_idx) == 0:
        raise RuntimeError("STOP: internal early-stop split is empty")

    mean = train.X[tr_idx].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train.X[tr_idx].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    Xopt = ((train.X[tr_idx] - mean) / std).astype(np.float32, copy=False)
    yopt = train.y[tr_idx]
    Xes = ((train.X[es_idx] - mean) / std).astype(np.float32, copy=False)
    yes = train.y[es_idx]
    Xreport = ((validation.X - mean) / std).astype(np.float32, copy=False)

    stable_ids = stable_class_ids(validation.y[clean], ds.n_classes, int(spec["reporting"]["stable_class_support_min"]))
    audit = {
        "canonical_manifest": spec["canonical_manifest"],
        "partition_lock": ds.partition_lock,
        "data_origin": ds.data_origin,
        "scenario": ds.scenario,
        "partition_seed": ds.partition_seed,
        "train_rows": int(len(train.y)),
        "optimizer_rows": int(len(tr_idx)),
        "internal_early_stop_rows": int(len(es_idx)),
        "raw_validation_rows": int(len(validation.y)),
        "train_duplicate_validation_rows_removed": int(dup.sum()),
        "deduplicated_validation_rows": int(clean.sum()),
        "stable_support_min": int(spec["reporting"]["stable_class_support_min"]),
        "stable_classes": stable_ids,
        "unstable_classes": [i for i in range(ds.n_classes) if i not in stable_ids],
        "sealed_final_test_access": False,
    }
    dump_json(out / "R2_DATA_AUDIT.json", audit)
    dump_json(out / "R2_TRAIN_ONLY_SCALER.json", {"mean": mean.tolist(), "std": std.tolist(), "fit_rows": int(len(tr_idx)), "reporting_validation_used_for_fit": False})

    results = []
    for candidate in spec["candidates"]:
        print(f"===== R2 CANDIDATE {candidate['id']} =====", flush=True)
        r = train_candidate(candidate, spec, ds, Xopt, yopt, Xes, yes, Xreport, clean, validation.y, out, device)
        results.append(r)

    summary_rows = []
    for order, r in enumerate(results):
        p = r["primary_deduplicated_validation"]
        s = r["stable_metrics"]
        summary_rows.append({
            "candidate_order": order,
            "candidate": r["candidate"]["id"],
            "architecture": r["candidate"]["architecture"],
            "imbalance": r["candidate"]["imbalance"],
            "loss": r["candidate"]["loss"],
            "accuracy": p["accuracy"],
            "weighted_f1": p["weighted_f1"],
            "macro_f1_all_15": p["macro_f1"],
            "balanced_accuracy": p["balanced_accuracy"],
            "far": p["far"],
            "stable_macro_f1": s["stable_macro_f1"],
            "worst_stable_recall": s["worst_stable_recall"],
            "candidate_pass": r["candidate_pass"],
            "best_epoch": r["best_epoch"],
            "fit_seconds": r["fit_seconds"],
        })
    write_csv(out / "R2_CANDIDATE_SUMMARY.csv", summary_rows)

    passed = [x for x in summary_rows if x["candidate_pass"]]
    if passed:
        passed.sort(key=lambda x: (-x["stable_macro_f1"], x["far"], -x["weighted_f1"], x["candidate_order"]))
        selected = passed[0]
        scientific_status = "CANDIDATE_SELECTED"
    else:
        selected = None
        scientific_status = "NO_CANDIDATE_PASSED"

    selection = {
        "step": "R2-S6",
        "system": "SYSTEM6",
        "technical_status": "TECHNICAL_PASS",
        "scientific_status": scientific_status,
        "selected_candidate": None if selected is None else selected["candidate"],
        "selected_metrics": selected,
        "selection_gate": spec["selection_gate"],
        "selection_rule": spec["selection_rule"],
        "tree_reference": reference,
        "automatic_r3_authorization": False,
        "student_manual_override_allowed": False,
        "sealed_final_test_access": False,
    }
    dump_json(out / "R2_SELECTION.json", selection)
    dump_json(out / "R2_SYSTEM6_SUMMARY.json", {
        "step": "R2-S6",
        "system": "SYSTEM6",
        "status": "TECHNICAL_PASS",
        "scientific_status": scientific_status,
        "calibration_seed": seed,
        "canonical_manifest": spec["canonical_manifest"],
        "candidates_run": [x["candidate"] for x in summary_rows],
        "selected_candidate": selection["selected_candidate"],
        "deduplicated_validation_rows": int(clean.sum()),
        "train_duplicate_validation_rows_removed": int(dup.sum()),
        "stable_classes": stable_ids,
        "sealed_final_test_access": False,
        "automatic_r3_authorization": False,
        "scientific_interpretation": "SUPERVISOR_ONLY",
    })
    print(json.dumps({"R2_SYSTEM6": selection}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
