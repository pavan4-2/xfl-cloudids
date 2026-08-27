"""Shared, deterministic utilities used by the six XFL-CloudIDS F0 tests.

Students should not run this file directly. The numbered F0 files import it so
that data loading, model training, metrics, hashing and logging are identical
across all six tests.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    fractions = sum(cfg[k] for k in ("train_fraction", "utility_fraction", "gate_fraction", "test_fraction"))
    if not np.isclose(fractions, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, found {fractions}")
    return cfg


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)}")


def environment_record(device: torch.device) -> dict:
    return {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_used": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


class LayerNormMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden: Sequence[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        previous = input_dim
        for width in hidden:
            layers.extend([nn.Linear(previous, width), nn.LayerNorm(width), nn.ReLU(), nn.Dropout(dropout)])
            previous = width
        layers.append(nn.Linear(previous, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def model_from_meta(meta: dict, cfg: dict, device: torch.device) -> LayerNormMLP:
    return LayerNormMLP(meta["input_dim"], meta["num_classes"], cfg["hidden_dims"], cfg["dropout"]).to(device)


def load_npz(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return data["x"].astype(np.float32), data["y"].astype(np.int64)


def tenant_file(root: str | Path, seed: int, tenant: int, split: str) -> Path:
    return Path(root) / "f01" / "partitions" / f"seed_{seed}" / f"tenant_{tenant}_{split}.npz"


def load_tenant(root: str | Path, seed: int, tenant: int, split: str) -> Tuple[np.ndarray, np.ndarray]:
    return load_npz(tenant_file(root, seed, tenant, split))


def class_weights(y: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = len(y) / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    device: torch.device,
    seed: int,
    epochs: int | None = None,
    proximal_state: Dict[str, torch.Tensor] | None = None,
) -> nn.Module:
    set_seed(seed)
    model.train()
    batch = min(int(cfg["batch_size"]), max(1, len(y)))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss(weight=class_weights(y, model.network[-1].out_features, device))
    for _ in range(epochs if epochs is not None else int(cfg["local_epochs"])):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if proximal_state is not None:
                prox = torch.zeros((), device=device)
                for name, parameter in model.named_parameters():
                    prox = prox + torch.sum((parameter - proximal_state[name].to(device)) ** 2)
                loss = loss + 0.5 * float(cfg["fedprox_mu"]) * prox
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
    return model


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, probabilities = [], []
    for start in range(0, len(x), batch_size):
        logits = model(torch.from_numpy(x[start : start + batch_size]).to(device))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(predictions), np.concatenate(probabilities)


@torch.no_grad()
def class_losses(model: nn.Module, x: np.ndarray, y: np.ndarray, num_classes: int, device: torch.device) -> np.ndarray:
    model.eval()
    logits_parts = []
    for start in range(0, len(x), 4096):
        logits_parts.append(model(torch.from_numpy(x[start : start + 4096]).to(device)).cpu())
    logits = torch.cat(logits_parts)
    losses = nn.functional.cross_entropy(logits, torch.from_numpy(y), reduction="none").numpy()
    result = np.full(num_classes, np.nan, dtype=np.float64)
    for cls in range(num_classes):
        mask = y == cls
        if mask.any():
            result[cls] = float(losses[mask].mean())
    return result


def metrics_from_predictions(y: np.ndarray, pred: np.ndarray, num_classes: int, rare_classes: Sequence[int]) -> dict:
    labels = np.arange(num_classes)
    recalls = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    benign = 0
    benign_mask = y == benign
    far = float(np.mean(pred[benign_mask] != benign)) if benign_mask.any() else float("nan")
    eligible_recalls = [recalls[c] for c in range(num_classes) if np.any(y == c)]
    rare_recalls = [recalls[c] for c in rare_classes if c < num_classes and np.any(y == c)]
    return {
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
        "worst_class_recall": float(min(eligible_recalls)) if eligible_recalls else float("nan"),
        "rare_class_recall": float(np.mean(rare_recalls)) if rare_recalls else float("nan"),
        "far": far,
        "n": int(len(y)),
    }


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, meta: dict, device: torch.device) -> dict:
    pred, _ = predict(model, x, device)
    return metrics_from_predictions(y, pred, meta["num_classes"], meta["rare_classes"])


def state_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def load_state(model: nn.Module, state: Dict[str, torch.Tensor], device: torch.device) -> nn.Module:
    model.load_state_dict({name: tensor.to(device) for name, tensor in state.items()})
    return model


def average_states(states: Sequence[Dict[str, torch.Tensor]], weights: Sequence[float]) -> Dict[str, torch.Tensor]:
    weights_np = np.asarray(weights, dtype=np.float64)
    weights_np = weights_np / weights_np.sum()
    output = {}
    for name in states[0]:
        acc = torch.zeros_like(states[0][name], dtype=torch.float32)
        for weight, state in zip(weights_np, states):
            acc.add_(state[name].float(), alpha=float(weight))
        output[name] = acc.to(states[0][name].dtype)
    return output


def flatten_update(candidate: Dict[str, torch.Tensor], base: Dict[str, torch.Tensor]) -> np.ndarray:
    return torch.cat([(candidate[k].float() - base[k].float()).reshape(-1) for k in sorted(base)]).numpy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0 else 0.0


def bootstrap_metric_differences(
    y: np.ndarray,
    pred_candidate: np.ndarray,
    pred_reference: np.ndarray,
    num_classes: int,
    rare_classes: Sequence[int],
    resamples: int,
    seed: int,
    blocks: int = 20,
) -> dict:
    rng = np.random.default_rng(seed)
    block_edges = np.linspace(0, len(y), min(blocks, len(y)) + 1, dtype=int)
    block_indices = [np.arange(block_edges[i], block_edges[i + 1]) for i in range(len(block_edges) - 1)]
    deltas = {"macro_f1": [], "far": [], "worst_class_recall": []}
    for _ in range(resamples):
        selected = rng.integers(0, len(block_indices), size=len(block_indices))
        idx = np.concatenate([block_indices[i] for i in selected])
        cand = metrics_from_predictions(y[idx], pred_candidate[idx], num_classes, rare_classes)
        ref = metrics_from_predictions(y[idx], pred_reference[idx], num_classes, rare_classes)
        for key in deltas:
            deltas[key].append(cand[key] - ref[key])
    result = {}
    for key, values in deltas.items():
        clean = np.asarray(values, dtype=float)
        clean = clean[np.isfinite(clean)]
        if len(clean):
            result[key] = {
                "mean": float(np.mean(clean)),
                "lcb95": float(np.quantile(clean, 0.025)),
                "ucb95": float(np.quantile(clean, 0.975)),
                "eligible": True,
            }
        else:
            result[key] = {"mean": float("nan"), "lcb95": float("nan"),
                           "ucb95": float("nan"), "eligible": False}
    return result


def run_manifest(test_id: str, cfg_path: Path, artifact_root: Path, device: torch.device, inputs: Iterable[Path]) -> dict:
    return {
        "test_id": test_id,
        "config_path": str(cfg_path.resolve()),
        "config_sha256": sha256_file(cfg_path),
        "artifact_root": str(artifact_root.resolve()),
        "environment": environment_record(device),
        "inputs": [{"path": str(p.resolve()), "sha256": sha256_file(p)} for p in inputs if p.exists()],
    }
