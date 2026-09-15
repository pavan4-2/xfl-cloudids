from __future__ import annotations
from typing import Dict, Mapping, Sequence
import numpy as np
from .common import sample_count_weights, weighted_average_states


def aggregate_fedavg(client_states: Sequence[Mapping[str, np.ndarray]], sample_counts: Sequence[int]) -> Dict[str, np.ndarray]:
    """Canonical sample-count weighted FedAvg aggregation."""
    return weighted_average_states(client_states, sample_count_weights(sample_counts))
