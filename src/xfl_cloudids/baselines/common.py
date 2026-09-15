from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence
import numpy as np

ArrayState = Mapping[str, np.ndarray]


def _validate_states(states: Sequence[ArrayState]) -> None:
    if not states:
        raise ValueError("at least one state is required")
    keys = tuple(states[0].keys())
    for state in states[1:]:
        if tuple(state.keys()) != keys:
            raise ValueError("all states must expose identical ordered keys")
        for key in keys:
            if np.asarray(state[key]).shape != np.asarray(states[0][key]).shape:
                raise ValueError(f"shape mismatch for parameter {key}")


def weighted_average_states(states: Sequence[ArrayState], weights: Sequence[float]) -> Dict[str, np.ndarray]:
    _validate_states(states)
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) != len(states):
        raise ValueError("weights must be one-dimensional and match states")
    if np.any(w < 0) or not np.isfinite(w).all() or float(w.sum()) <= 0:
        raise ValueError("weights must be finite, non-negative, and have positive sum")
    w = w / w.sum()
    out: Dict[str, np.ndarray] = {}
    for key in states[0]:
        stack = np.stack([np.asarray(s[key], dtype=float) for s in states], axis=0)
        out[key] = np.tensordot(w, stack, axes=(0, 0))
    return out


def sample_count_weights(sample_counts: Sequence[int]) -> np.ndarray:
    n = np.asarray(sample_counts, dtype=float)
    if n.ndim != 1 or len(n) == 0 or np.any(n <= 0):
        raise ValueError("sample counts must be positive")
    return n / n.sum()


def state_l2_distance(a: ArrayState, b: ArrayState) -> float:
    if set(a) != set(b):
        raise ValueError("state keys differ")
    sq = 0.0
    for key in a:
        da = np.asarray(a[key], dtype=float)
        db = np.asarray(b[key], dtype=float)
        if da.shape != db.shape:
            raise ValueError(f"shape mismatch for {key}")
        sq += float(np.sum((da - db) ** 2))
    return float(np.sqrt(sq))
