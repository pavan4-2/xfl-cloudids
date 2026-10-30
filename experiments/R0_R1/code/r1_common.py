from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import (
    classification_metrics,
    dump_json,
    load_manifest,
    pool_role,
    validation_duplicate_mask,
    write_csv,
)


def load_r1_data(bridge: Path, spec: dict):
    ds = load_manifest(bridge / spec["canonical_manifest"], bridge)
    train = pool_role(ds, "train")
    validation = pool_role(ds, "validation")
    duplicate_mask = validation_duplicate_mask(train, validation)
    clean_mask = ~duplicate_mask
    if clean_mask.sum() == 0:
        raise AssertionError("STOP: every validation row duplicates TRAIN; invalid recovery benchmark")
    return ds, train, validation, duplicate_mask, clean_mask


def emit_metrics(out: Path, model_name: str, ds, validation, y_pred: np.ndarray, duplicate_mask: np.ndarray, gate: dict):
    raw = classification_metrics(validation.y, y_pred, ds.n_classes, ds.benign_label)
    clean = classification_metrics(validation.y[~duplicate_mask], y_pred[~duplicate_mask], ds.n_classes, ds.benign_label)
    criteria = {
        "accuracy": clean["accuracy"] >= float(gate["accuracy_min"]),
        "weighted_f1": clean["weighted_f1"] >= float(gate["weighted_f1_min"]),
        "macro_f1": clean["macro_f1"] >= float(gate["macro_f1_min"]),
        "far": clean["far"] <= float(gate["far_max"]),
    }
    result = {
        "model": model_name,
        "primary_deduplicated_validation": clean,
        "secondary_raw_validation": raw,
        "train_duplicate_validation_rows": int(duplicate_mask.sum()),
        "deduplicated_validation_rows": int((~duplicate_mask).sum()),
        "gate_thresholds": gate,
        "gate_checks": criteria,
        "ceiling_gate_pass": bool(all(criteria.values())),
        "sealed_final_test_access": False,
    }
    dump_json(out / f"R1_{model_name}_METRICS.json", result)

    per_class_rows = []
    for scope, m in [("deduplicated_validation", clean), ("raw_validation", raw)]:
        for r in m["per_class"]:
            row = {"model": model_name, "scope": scope, **r}
            per_class_rows.append(row)
    write_csv(out / f"R1_{model_name}_PER_CLASS.csv", per_class_rows)

    for scope, m in [("deduplicated_validation", clean), ("raw_validation", raw)]:
        cm = np.asarray(m["confusion_matrix"], dtype=int)
        rows = []
        for i in range(cm.shape[0]):
            row = {"true_class": i}
            row.update({f"pred_{j}": int(cm[i, j]) for j in range(cm.shape[1])})
            rows.append(row)
        write_csv(out / f"R1_{model_name}_{scope}_CONFUSION_MATRIX.csv", rows)
    return result
