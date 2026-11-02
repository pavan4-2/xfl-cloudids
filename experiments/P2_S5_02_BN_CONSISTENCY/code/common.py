from __future__ import annotations

import csv
import json
import os
import random
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

FORBIDDEN_PATH_TOKENS = ("sealed", "untouched", "final_test", "final-test")


@dataclass(frozen=True)
class Split:
    X: np.ndarray
    y: np.ndarray
    source_path: Path


@dataclass(frozen=True)
class Client:
    cid: int
    splits: Dict[str, Split]


@dataclass(frozen=True)
class Dataset:
    clients: Dict[int, Client]
    feature_dim: int
    n_classes: int
    benign_label: int
    manifest_path: Path
    partition_lock: str
    data_origin: str
    scenario: str
    partition_seed: int


def dump_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)


def _json_default(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Path):
        return str(v)
    raise TypeError(type(v).__name__)


def write_csv(path: str | Path, rows: Sequence[Mapping]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _is_forbidden_relpath(rel: Path) -> bool:
    s = str(rel).lower().replace("\\", "/")
    return any(tok in s for tok in FORBIDDEN_PATH_TOKENS)


def _read_split(root: Path, desc: Mapping, feature_dim: int, n_classes: int) -> Split:
    rel = Path(str(desc["path"]))
    if rel.is_absolute() or rel.suffix.lower() != ".npz" or _is_forbidden_relpath(rel):
        raise ValueError(f"R0/R1 permits only non-sealed development NPZ paths: {rel}")
    p = (root / rel).resolve()
    if root not in p.parents:
        raise ValueError("split path escapes development bridge")
    with np.load(p, allow_pickle=False) as z:
        X = np.asarray(z["X"], dtype=np.float32)
        y = np.asarray(z["y"], dtype=np.int64)
    if X.ndim != 2 or X.shape[1] != feature_dim:
        raise ValueError(f"invalid feature shape {X.shape} in {p.name}")
    if y.ndim != 1 or len(y) != len(X) or len(y) == 0:
        raise ValueError(f"invalid label shape in {p.name}")
    if not np.isfinite(X).all():
        raise ValueError(f"non-finite feature value in {p.name}")
    if np.any(y < 0) or np.any(y >= n_classes):
        raise ValueError(f"label outside 0..{n_classes-1} in {p.name}")
    return Split(X=X, y=y, source_path=p)


def load_manifest(manifest_path: str | Path, bridge_root: str | Path, allowed_roles: tuple[str, ...] | None = None) -> Dataset:
    root = Path(bridge_root).resolve()
    mp = Path(manifest_path).resolve()
    d = json.loads(mp.read_text(encoding="utf-8-sig"))
    if d.get("scope") != "development_only":
        raise ValueError("STOP: manifest is not development_only")
    if "sealed" in json.dumps(d).lower() or "untouched_final" in json.dumps(d).lower():
        # The manifest may legitimately contain a boolean declaration, but it must never reference a sealed path.
        for c in d.get("clients", []):
            for desc in c.get("splits", {}).values():
                if _is_forbidden_relpath(Path(str(desc.get("path", "")))):
                    raise ValueError("STOP: forbidden sealed/final path present in manifest")
    feature_dim = int(d["feature_dim"])
    n_classes = int(d["n_classes"])
    benign_label = int(d["benign_label"])
    clients: Dict[int, Client] = {}
    expected_roles = {"train", "utility", "gate", "validation"}
    roles_to_read = expected_roles if allowed_roles is None else set(allowed_roles)
    if not roles_to_read or not roles_to_read.issubset(expected_roles):
        raise ValueError("STOP: unsupported data roles requested")
    for c in d["clients"]:
        cid = int(c["cid"])
        if set(c["splits"]) != expected_roles:
            raise ValueError(f"client {cid}: expected exactly {sorted(expected_roles)}")
        # Explicitly load only authorized data roles; other roles remain metadata-only.
        splits = {role: _read_split(root, c["splits"][role], feature_dim, n_classes)
                  for role in sorted(roles_to_read)}
        clients[cid] = Client(cid=cid, splits=splits)
    if len(clients) < 2:
        raise ValueError("expected multiple clients")
    return Dataset(
        clients=clients,
        feature_dim=feature_dim,
        n_classes=n_classes,
        benign_label=benign_label,
        manifest_path=mp,
        partition_lock=str(d.get("partition_lock", "UNDECLARED")),
        data_origin=str(d.get("data_origin", "UNDECLARED")),
        scenario=str(d.get("scenario", mp.stem)),
        partition_seed=int(d.get("partition_seed", -1)),
    )


def pool_role(ds: Dataset, role: str) -> Split:
    xs = []
    ys = []
    for cid in sorted(ds.clients):
        split = ds.clients[cid].splits[role]
        xs.append(split.X)
        ys.append(split.y)
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return Split(X=X, y=y, source_path=Path(f"POOLED::{role}"))


def class_counts(y: np.ndarray, n_classes: int) -> np.ndarray:
    return np.bincount(np.asarray(y, dtype=np.int64), minlength=n_classes)


def row_key(x: np.ndarray, y: int) -> bytes:
    # Exact float32 row bytes + int16 label. Labels are < 32768 by contract.
    return memoryview(np.ascontiguousarray(x, dtype=np.float32)).cast("B").tobytes() + struct.pack("<h", int(y))


def overlap_against_reference(reference: Split, candidate: Split) -> dict:
    """Exact duplicate X+y audit.

    Builds a key dictionary only for the smaller candidate. Intended for R0 canonical
    development roles, not for use inside training.
    """
    t0 = time.perf_counter()
    keys = set()
    duplicate_rows_within_candidate = 0
    for i in range(len(candidate.y)):
        k = row_key(candidate.X[i], int(candidate.y[i]))
        if k in keys:
            duplicate_rows_within_candidate += 1
        keys.add(k)
    matched_reference_rows = 0
    matched_unique = set()
    for i in range(len(reference.y)):
        k = row_key(reference.X[i], int(reference.y[i]))
        if k in keys:
            matched_reference_rows += 1
            matched_unique.add(k)
    return {
        "reference_rows": int(len(reference.y)),
        "candidate_rows": int(len(candidate.y)),
        "candidate_unique_signatures": int(len(keys)),
        "candidate_duplicate_rows": int(duplicate_rows_within_candidate),
        "reference_rows_matching_candidate_signature": int(matched_reference_rows),
        "unique_overlapping_signatures": int(len(matched_unique)),
        "elapsed_seconds": float(time.perf_counter() - t0),
    }


def validation_duplicate_mask(train: Split, validation: Split) -> np.ndarray:
    """Return True for validation rows whose exact X+y signature exists in train."""
    index: Dict[bytes, list[int]] = {}
    for i in range(len(validation.y)):
        k = row_key(validation.X[i], int(validation.y[i]))
        index.setdefault(k, []).append(i)
    duplicate = np.zeros(len(validation.y), dtype=bool)
    remaining = len(index)
    if remaining == 0:
        return duplicate
    for i in range(len(train.y)):
        k = row_key(train.X[i], int(train.y[i]))
        hit = index.get(k)
        if hit:
            duplicate[np.asarray(hit, dtype=np.int64)] = True
            del index[k]
            remaining -= 1
            if remaining == 0:
                break
    return duplicate


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, benign_label: int) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    labels = np.arange(n_classes, dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    benign = int(benign_label)
    benign_rows = int((y_true == benign).sum())
    benign_false_alarm = int(((y_true == benign) & (y_pred != benign)).sum())
    far = float(benign_false_alarm / benign_rows) if benign_rows else float("nan")
    return {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "far": far,
        "benign_rows": benign_rows,
        "benign_false_alarm_rows": benign_false_alarm,
        "per_class": [
            {
                "class_id": int(i),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(n_classes)
        ],
        "confusion_matrix": cm.astype(int).tolist(),
    }


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def system_environment(python_executable: str | None = None) -> dict:
    import platform
    import sys
    out = {
        "python": sys.version,
        "python_executable": python_executable or sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import sklearn
        out["sklearn"] = sklearn.__version__
    except Exception as e:
        out["sklearn_error"] = repr(e)
    try:
        import torch
        out.update({
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except Exception as e:
        out["torch_error"] = repr(e)
    return out
