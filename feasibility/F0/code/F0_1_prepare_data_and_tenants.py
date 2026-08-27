#!/usr/bin/env python3
"""
TEST F0.1 — Prepare safe data and five non-IID tenants
WHY THIS TEST IS NEEDED:
This test creates the exact training, utility-validation and gate data used by
the later feasibility tests. It also seals an untouched final-test file. Doing
this first prevents data leakage and ensures every method receives identical
data. Students only need to provide the folder containing CIC-IDS2017 CSV files.

MAIN OUTPUT: artifacts/f0/f01/F0_1_PASS.json plus tenant NPZ files and hashes.
IMPORTANT: Later F0 tests never open untouched_final_test.npz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from f0_common import load_config, run_manifest, set_seed, sha256_file, write_json


LABEL_MAP = {
    "BENIGN": "BENIGN",
    "DDOS": "DDOS",
    "PORTSCAN": "PORTSCAN",
    "BOT": "BOT",
    "INFILTRATION": "INFILTRATION",
    "HEARTBLEED": "HEARTBLEED",
    "FTP-PATATOR": "BRUTE_FORCE",
    "SSH-PATATOR": "BRUTE_FORCE",
    "WEB ATTACK – BRUTE FORCE": "WEB_ATTACK",
    "WEB ATTACK – XSS": "WEB_ATTACK",
    "WEB ATTACK – SQL INJECTION": "WEB_ATTACK",
    "WEB ATTACK - BRUTE FORCE": "WEB_ATTACK",
    "WEB ATTACK - XSS": "WEB_ATTACK",
    "WEB ATTACK - SQL INJECTION": "WEB_ATTACK",
}


def normalize_label(value: object) -> str:
    text = str(value).strip().upper().replace("�", "-")
    if text in LABEL_MAP:
        return LABEL_MAP[text]
    if text.startswith("DOS ") or text.startswith("DOS-"):
        return "DOS"
    if "WEB ATTACK" in text:
        return "WEB_ATTACK"
    return text.replace(" ", "_")


def synthetic_frame(seed: int = 17, rows: int = 6000) -> pd.DataFrame:
    """Small fixture for code verification only; never valid as research evidence."""
    rng = np.random.default_rng(seed)
    labels = np.array(["BENIGN", "DOS", "DDOS", "PORTSCAN", "BRUTE_FORCE"])
    probabilities = np.array([0.55, 0.18, 0.12, 0.10, 0.05])
    y = rng.choice(labels, rows, p=probabilities)
    centers = {name: idx * 0.75 for idx, name in enumerate(labels)}
    data = {f"Feature {j}": rng.normal([centers[v] for v in y], 1.0 + j * 0.02) for j in range(12)}
    data["Flow ID"] = [f"fixture-{i}" for i in range(rows)]
    data["Timestamp"] = np.arange(rows)
    data["Label"] = y
    return pd.DataFrame(data)


def read_cicids(raw_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    files = sorted(raw_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")
    frames = []
    for file_index, path in enumerate(files):
        frame = pd.read_csv(path, low_memory=False)
        frame.columns = [str(c).strip() for c in frame.columns]
        frame["__file_order"] = file_index
        frame["__row_order"] = np.arange(len(frame))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), files


def chronological_class_split(y: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    output = {"train": [], "utility": [], "gate": [], "test": []}
    fractions = np.cumsum([
        cfg["train_fraction"], cfg["utility_fraction"], cfg["gate_fraction"], cfg["test_fraction"]
    ])
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        boundaries = (fractions[:-1] * len(idx)).astype(int)
        pieces = np.split(idx, boundaries)
        for name, part in zip(output, pieces):
            output[name].extend(part.tolist())
    return {name: np.array(sorted(idx), dtype=np.int64) for name, idx in output.items()}


def dirichlet_partition(y: np.ndarray, tenants: int, alpha: float, seed: int,
                        eligibility_floor: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    allocations = [[] for _ in range(tenants)]
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        # Give every tenant a small eligibility floor when the class is large
        # enough, then allocate the remainder non-IID. This prevents undefined
        # gate metrics without making the overall distribution IID.
        floor = min(eligibility_floor, len(idx) // tenants)
        cursor = 0
        if floor > 0:
            for tenant in range(tenants):
                allocations[tenant].extend(idx[cursor : cursor + floor].tolist())
                cursor += floor
        idx = idx[cursor:]
        if len(idx) == 0:
            continue
        proportions = rng.dirichlet(np.full(tenants, alpha))
        counts = rng.multinomial(len(idx), proportions)
        cursor = 0
        for tenant, count in enumerate(counts):
            allocations[tenant].extend(idx[cursor : cursor + count].tolist())
            cursor += count
    for tenant in range(tenants):
        rng.shuffle(allocations[tenant])
    return [np.asarray(values, dtype=np.int64) for values in allocations]


def main() -> None:
    parser = argparse.ArgumentParser(description="F0.1 leakage-safe CIC-IDS2017 preparation")
    parser.add_argument("--raw-dir", type=Path, help="Folder containing CIC-IDS2017 CSV files")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("f0_config.json"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/f0"))
    parser.add_argument("--synthetic-smoke", action="store_true", help="Use a small fake fixture only to test code")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["development_seeds"][0])
    out = args.artifact_root / "f01"
    out.mkdir(parents=True, exist_ok=True)

    if args.synthetic_smoke:
        frame, source_files = synthetic_frame(), []
        data_source = "SYNTHETIC_SMOKE_NOT_RESEARCH_EVIDENCE"
    else:
        if args.raw_dir is None:
            raise SystemExit("Provide --raw-dir or use --synthetic-smoke for code verification only.")
        frame, source_files = read_cicids(args.raw_dir)
        data_source = "CIC-IDS2017"

    frame.columns = [str(c).strip() for c in frame.columns]
    label_candidates = [c for c in frame.columns if c.lower() in {"label", "class"}]
    if len(label_candidates) != 1:
        raise ValueError(f"Expected exactly one Label/Class column, found {label_candidates}")
    label_col = label_candidates[0]
    labels_text = frame[label_col].map(normalize_label)
    order_columns = [c for c in ("__file_order", "__row_order") if c in frame]
    if order_columns:
        frame = frame.assign(__label=labels_text).sort_values(order_columns, kind="stable")
        labels_text = frame.pop("__label")

    if len(frame) > int(cfg["max_rows"]):
        selected = []
        per_class = max(1, int(cfg["max_rows"]) // labels_text.nunique())
        for label in sorted(labels_text.unique()):
            idx = np.flatnonzero(labels_text.to_numpy() == label)
            selected.extend(idx[np.linspace(0, len(idx) - 1, min(per_class, len(idx)), dtype=int)].tolist())
        selected = np.asarray(sorted(selected[: int(cfg["max_rows"])]), dtype=np.int64)
        frame = frame.iloc[selected].reset_index(drop=True)
        labels_text = labels_text.iloc[selected].reset_index(drop=True)

    class_names = ["BENIGN"] + sorted(x for x in labels_text.unique() if x != "BENIGN")
    label_to_id = {name: idx for idx, name in enumerate(class_names)}
    y = labels_text.map(label_to_id).to_numpy(np.int64)

    excluded = {label_col, "Flow ID", "Source IP", "Destination IP", "Timestamp", "__file_order", "__row_order"}
    features = frame.drop(columns=[c for c in excluded if c in frame], errors="ignore")
    features = features.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    all_nan = features.columns[features.isna().all()].tolist()
    features = features.drop(columns=all_nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    constant = [c for c in features.columns if features[c].nunique(dropna=False) <= 1]
    features = features.drop(columns=constant)
    if features.shape[1] == 0:
        raise ValueError("No usable numeric features remain after leakage filtering")
    x = features.to_numpy(np.float32)

    split_idx = chronological_class_split(y, cfg)
    scaler = StandardScaler().fit(x[split_idx["train"]])
    x = scaler.transform(x).astype(np.float32)
    np.savez_compressed(out / "untouched_final_test.npz", x=x[split_idx["test"]], y=y[split_idx["test"]])
    write_json(out / "untouched_final_test.SEAL.json", {
        "status": "SEALED_DO_NOT_OPEN_DURING_F0",
        "sha256": sha256_file(out / "untouched_final_test.npz"),
        "rows": int(len(split_idx["test"])),
    })

    split_arrays = {name: (x[idx], y[idx]) for name, idx in split_idx.items() if name != "test"}
    counts_rows = []
    for seed in cfg["development_seeds"]:
        seed_dir = out / "partitions" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for split_number, (split, (split_x, split_y)) in enumerate(split_arrays.items()):
            allocations = dirichlet_partition(split_y, cfg["num_tenants"], cfg["dirichlet_alpha"],
                                               seed + split_number * 1000,
                                               int(cfg["min_class_examples"]))
            for tenant, idx in enumerate(allocations):
                if len(idx) == 0:
                    raise RuntimeError(f"Tenant {tenant} has no {split} rows; change only the documented partition seed")
                path = seed_dir / f"tenant_{tenant}_{split}.npz"
                np.savez_compressed(path, x=split_x[idx], y=split_y[idx])
                bincount = np.bincount(split_y[idx], minlength=len(class_names))
                for cls, count in enumerate(bincount):
                    counts_rows.append({"seed": seed, "tenant": tenant, "split": split, "class_id": cls,
                                        "class_name": class_names[cls], "count": int(count)})

    counts = pd.DataFrame(counts_rows)
    counts.to_csv(out / "tenant_class_counts.csv", index=False)
    class_total = np.bincount(y[split_idx["train"]], minlength=len(class_names))
    non_benign = [(idx, count) for idx, count in enumerate(class_total) if idx != 0 and count > 0]
    rare_classes = [idx for idx, _ in sorted(non_benign, key=lambda item: item[1])[: max(1, min(2, len(non_benign)))]]
    feature_hash = hashlib.sha256("\n".join(features.columns).encode()).hexdigest()
    meta = {
        "data_source": data_source,
        "input_dim": int(features.shape[1]),
        "num_classes": len(class_names),
        "class_names": class_names,
        "label_to_id": label_to_id,
        "rare_classes": rare_classes,
        "feature_names": features.columns.tolist(),
        "feature_schema_sha256": feature_hash,
        "excluded_columns": sorted(excluded),
        "dropped_all_nan": all_nan,
        "dropped_constant": constant,
        "split_rows": {name: int(len(idx)) for name, idx in split_idx.items()},
        "raw_files": [{"name": p.name, "sha256": sha256_file(p)} for p in source_files],
    }
    write_json(out / "data_metadata.json", meta)
    manifest = run_manifest("F0.1", args.config, args.artifact_root, __import__("torch").device("cpu"), source_files)
    manifest.update({"status": "PASS", "data_source": data_source, "metadata_sha256": sha256_file(out / "data_metadata.json")})
    write_json(out / "run_manifest.json", manifest)
    write_json(out / "F0_1_PASS.json", {
        "test": "F0.1", "status": "PASS", "research_evidence": not args.synthetic_smoke,
        "next_test": "F0_2_baselines_and_negative_transfer.py"
    })
    print(json.dumps({"status": "PASS", "data_source": data_source, "rows": len(frame),
                      "features": features.shape[1], "classes": class_names,
                      "sealed_test_sha256": sha256_file(out / "untouched_final_test.npz")}, indent=2))


if __name__ == "__main__":
    main()
