from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import (
    class_counts,
    dump_json,
    load_manifest,
    overlap_against_reference,
    pool_role,
    system_environment,
    write_csv,
)

ROLES = ("train", "utility", "gate", "validation")


def audit(bridge: Path, out: Path, spec: dict, system: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    env = system_environment()
    env["system"] = system
    env["step"] = "R0"
    dump_json(out / "R0_ENVIRONMENT.json", env)

    manifest_rows = []
    split_rows = []
    pooled_rows = []
    structural_signatures = []
    canonical = None

    for name in spec["audit_manifests"]:
        mp = bridge / name
        if not mp.exists():
            raise FileNotFoundError(f"STOP: required development manifest not found: {mp}")
        ds = load_manifest(mp, bridge)
        structural_signatures.append((ds.feature_dim, ds.n_classes, ds.benign_label, len(ds.clients), ds.data_origin))
        if name == spec["canonical_manifest"]:
            canonical = ds
        manifest_rows.append({
            "manifest": name,
            "scenario": ds.scenario,
            "partition_seed": ds.partition_seed,
            "partition_lock": ds.partition_lock,
            "data_origin": ds.data_origin,
            "feature_dim": ds.feature_dim,
            "n_classes": ds.n_classes,
            "benign_label": ds.benign_label,
            "clients": len(ds.clients),
        })
        for cid in sorted(ds.clients):
            for role in ROLES:
                sp = ds.clients[cid].splits[role]
                counts = class_counts(sp.y, ds.n_classes)
                row = {
                    "manifest": name,
                    "scenario": ds.scenario,
                    "partition_seed": ds.partition_seed,
                    "client": cid,
                    "role": role,
                    "rows": len(sp.y),
                    "source_file": sp.source_path.name,
                    "source_file_bytes": int(sp.source_path.stat().st_size),
                    "x_rows": int(sp.X.shape[0]),
                    "x_features": int(sp.X.shape[1]),
                    "x_dtype": str(sp.X.dtype),
                    "y_rows": int(sp.y.shape[0]),
                    "y_dtype": str(sp.y.dtype),
                }
                row.update({f"class_{i}": int(counts[i]) for i in range(ds.n_classes)})
                split_rows.append(row)
        for role in ROLES:
            p = pool_role(ds, role)
            counts = class_counts(p.y, ds.n_classes)
            row = {
                "manifest": name,
                "scenario": ds.scenario,
                "partition_seed": ds.partition_seed,
                "role": role,
                "rows": len(p.y),
                "present_classes": int((counts > 0).sum()),
                "min_nonzero_class_support": int(counts[counts > 0].min()) if np.any(counts > 0) else 0,
                "max_class_support": int(counts.max()) if len(counts) else 0,
            }
            row.update({f"class_{i}": int(counts[i]) for i in range(ds.n_classes)})
            pooled_rows.append(row)

    # Structural consistency across the four frozen manifests.
    dims = set(structural_signatures)
    if len(dims) != 1:
        raise AssertionError(f"R0 structural inconsistency across development manifests: {dims}")

    canonical_name = spec["canonical_manifest"]
    if canonical is None:
        raise AssertionError("canonical manifest was not audited")
    pooled = {role: pool_role(canonical, role) for role in ROLES}

    # Exact duplicate signature audit. These are reported, not silently removed from history.
    overlap = {
        "train_vs_validation": overlap_against_reference(pooled["train"], pooled["validation"]),
        "train_vs_utility": overlap_against_reference(pooled["train"], pooled["utility"]),
        "train_vs_gate": overlap_against_reference(pooled["train"], pooled["gate"]),
    }
    dump_json(out / "R0_CANONICAL_EXACT_DUPLICATE_AUDIT.json", overlap)

    train_counts = class_counts(pooled["train"].y, canonical.n_classes)
    val_counts = class_counts(pooled["validation"].y, canonical.n_classes)
    class_support = []
    for i in range(canonical.n_classes):
        class_support.append({
            "class_id": i,
            "train_rows": int(train_counts[i]),
            "validation_rows": int(val_counts[i]),
            "rare_train_lt_100": bool(train_counts[i] < 100),
            "very_rare_train_lt_20": bool(train_counts[i] < 20),
            "unstable_validation_support_lt_10": bool(val_counts[i] < 10),
        })
    write_csv(out / "R0_CANONICAL_CLASS_SUPPORT.csv", class_support)

    # Verify that global role totals/class supports are stable across the four manifests.
    role_signatures = {}
    for row in pooled_rows:
        role = row["role"]
        sig = (row["rows"],) + tuple(row[f"class_{i}"] for i in range(canonical.n_classes))
        role_signatures.setdefault(role, set()).add(sig)
    global_role_consistency = {
        role: {
            "unique_global_count_signatures": len(sigs),
            "consistent": len(sigs) == 1,
        }
        for role, sigs in role_signatures.items()
    }

    write_csv(out / "R0_MANIFEST_SUMMARY.csv", manifest_rows)
    write_csv(out / "R0_CLIENT_SPLIT_STRUCTURE.csv", split_rows)
    write_csv(out / "R0_POOLED_ROLE_CLASS_COUNTS.csv", pooled_rows)

    summary = {
        "step": "R0",
        "system": system,
        "status": "TECHNICAL_PASS",
        "scope": "development_only",
        "sealed_final_test_access": False,
        "bridge": str(bridge),
        "canonical_manifest": canonical_name,
        "feature_dim": canonical.feature_dim,
        "n_classes": canonical.n_classes,
        "benign_label": canonical.benign_label,
        "canonical_rows": {role: int(len(pooled[role].y)) for role in ROLES},
        "global_role_count_consistency": global_role_consistency,
        "rare_classes_train_lt_100": [int(i) for i, n in enumerate(train_counts) if n < 100],
        "very_rare_classes_train_lt_20": [int(i) for i, n in enumerate(train_counts) if n < 20],
        "validation_support_lt_10": [int(i) for i, n in enumerate(val_counts) if n < 10],
        "duplicate_audit": overlap,
        "elapsed_seconds": time.perf_counter() - started,
        "scientific_interpretation": "SUPERVISOR_ONLY",
    }
    dump_json(out / "R0_SUMMARY.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--system", choices=["SYSTEM5", "SYSTEM6"], required=True)
    args = ap.parse_args()
    spec = json.loads((Path(__file__).resolve().parent / "RECOVERY_SPECIFICATION.json").read_text())
    summary = audit(Path(args.bridge).resolve(), Path(args.out).resolve(), spec, args.system)
    print(json.dumps({"R0": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
