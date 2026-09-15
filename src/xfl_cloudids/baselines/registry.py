from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    family: str
    personalized: bool
    notes: str


BASELINES: Dict[str, BaselineSpec] = {
    "local_only": BaselineSpec("Local-only", "non_federated", True, "Receiver trains/evaluates its own model only."),
    "fedavg": BaselineSpec("FedAvg", "global_fl", False, "Sample-count weighted global aggregation."),
    "fedprox": BaselineSpec("FedProx", "global_fl", False, "FedAvg plus proximal local objective."),
    "scaffold": BaselineSpec("SCAFFOLD", "variance_reduced_fl", False, "Control-variate correction for client drift."),
    "ditto": BaselineSpec("Ditto", "personalized_fl", True, "Global model plus proximal personalized model."),
    "fedfomo": BaselineSpec("FedFomo", "personalized_fl", True, "Receiver validation-driven personalized peer weighting."),
}


def baseline_names():
    return tuple(BASELINES.keys())
