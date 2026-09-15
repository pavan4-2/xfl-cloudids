from __future__ import annotations
from typing import Dict, Sequence
import numpy as np


def fedfomo_positive_improvement_weights(peer_ids: Sequence[int], improvements: Sequence[float]) -> Dict[int, float]:
    """Normalize positive local-validation improvements for FedFomo-style personalized mixing.

    The full experiment runner must compute each improvement on the receiver's validation split.
    Non-positive peers receive zero mass. If no peer improves validation utility, all peer masses are zero.
    """
    if len(peer_ids) != len(improvements):
        raise ValueError("peer_ids and improvements must have equal length")
    gains = np.maximum(np.asarray(improvements, dtype=float), 0.0)
    if not np.isfinite(gains).all():
        raise ValueError("improvements must be finite")
    total = float(gains.sum())
    if total <= 0:
        return {int(pid): 0.0 for pid in peer_ids}
    return {int(pid): float(g / total) for pid, g in zip(peer_ids, gains)}
