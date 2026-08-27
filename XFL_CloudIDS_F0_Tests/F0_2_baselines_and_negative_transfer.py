#!/usr/bin/env python3
"""
TEST F0.2 — Check whether ordinary federation harms any tenant
WHY THIS TEST IS NEEDED:
RUTRA-Gate is useful only if FedAvg/FedProx sometimes transfer harmful knowledge
between different tenants. This test runs Local-only, Centralized, FedAvg and
FedProx with the same data, model and seeds, then measures how often federation
reduces a tenant's macro-F1. Do not continue if F0.1 has not passed.

MAIN OUTPUT: artifacts/f0/f02/metrics.csv and negative_transfer_summary.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from f0_common import (average_states, evaluate, load_config, load_tenant,
                       model_from_meta, run_manifest, set_seed, state_cpu,
                       train_model, write_json)


def federated_model(method: str, base_state: dict, seed: int, cfg: dict, meta: dict,
                    artifact_root: Path, device: torch.device, rounds: int) -> torch.nn.Module:
    global_model = model_from_meta(meta, cfg, device)
    global_model.load_state_dict(base_state)
    for round_id in range(rounds):
        global_state = state_cpu(global_model)
        candidates, sizes = [], []
        for tenant in range(cfg["num_tenants"]):
            x, y = load_tenant(artifact_root, seed, tenant, "train")
            local = model_from_meta(meta, cfg, device)
            local.load_state_dict(global_state)
            proximal = global_state if method == "FedProx" else None
            train_model(local, x, y, cfg, device, seed + round_id * 100 + tenant,
                        proximal_state=proximal)
            candidates.append(state_cpu(local))
            sizes.append(len(y))
        global_model.load_state_dict(average_states(candidates, sizes))
    return global_model


def main() -> None:
    parser = argparse.ArgumentParser(description="F0.2 baseline and negative-transfer test")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("f0_config.json"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/f0"))
    parser.add_argument("--rounds", type=int, help="Override only for code smoke testing")
    parser.add_argument("--local-epochs", type=int, help="Override only for code smoke testing")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.local_epochs is not None:
        cfg["local_epochs"] = args.local_epochs
    rounds = args.rounds or int(cfg["federated_rounds"])
    pass_file = args.artifact_root / "f01" / "F0_1_PASS.json"
    if not pass_file.exists():
        raise SystemExit("STOP: F0.1 PASS file is missing.")
    with open(args.artifact_root / "f01" / "data_metadata.json", encoding="utf-8") as handle:
        meta = json.load(handle)
    device = torch.device("cuda" if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    out = args.artifact_root / "f02"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    start_time = time.time()

    for seed in cfg["development_seeds"]:
        set_seed(seed)
        base = model_from_meta(meta, cfg, device)
        base_state = state_cpu(base)

        local_models = {}
        for tenant in range(cfg["num_tenants"]):
            x_train, y_train = load_tenant(args.artifact_root, seed, tenant, "train")
            model = model_from_meta(meta, cfg, device)
            model.load_state_dict(base_state)
            train_model(model, x_train, y_train, cfg, device, seed + tenant,
                        epochs=rounds * int(cfg["local_epochs"]))
            local_models[tenant] = model

        pooled_x, pooled_y = [], []
        for tenant in range(cfg["num_tenants"]):
            x_train, y_train = load_tenant(args.artifact_root, seed, tenant, "train")
            pooled_x.append(x_train); pooled_y.append(y_train)
        centralized = model_from_meta(meta, cfg, device)
        centralized.load_state_dict(base_state)
        train_model(centralized, np.concatenate(pooled_x), np.concatenate(pooled_y), cfg, device, seed + 9000,
                    epochs=rounds * int(cfg["local_epochs"]))
        fedavg = federated_model("FedAvg", base_state, seed, cfg, meta, args.artifact_root, device, rounds)
        fedprox = federated_model("FedProx", base_state, seed, cfg, meta, args.artifact_root, device, rounds)

        for tenant in range(cfg["num_tenants"]):
            x_dev, y_dev = load_tenant(args.artifact_root, seed, tenant, "utility")
            for method, model in (("Local-only", local_models[tenant]), ("Centralized", centralized),
                                  ("FedAvg", fedavg), ("FedProx", fedprox)):
                metric = evaluate(model, x_dev, y_dev, meta, device)
                rows.append({"seed": seed, "tenant": tenant, "method": method, "split": "utility", **metric})
        torch.save({"seed": seed, "fedavg": state_cpu(fedavg), "fedprox": state_cpu(fedprox),
                    "local": {k: state_cpu(v) for k, v in local_models.items()}}, out / f"models_seed_{seed}.pt")

    metrics = pd.DataFrame(rows)
    local = metrics[metrics.method == "Local-only"][["seed", "tenant", "macro_f1", "rare_class_recall"]].rename(
        columns={"macro_f1": "local_macro_f1", "rare_class_recall": "local_rare_recall"})
    metrics = metrics.merge(local, on=["seed", "tenant"], how="left")
    metrics["macro_f1_delta_vs_local"] = metrics.macro_f1 - metrics.local_macro_f1
    metrics["negative_transfer"] = metrics.macro_f1_delta_vs_local < -float(cfg["negative_transfer_margin"])
    metrics.to_csv(out / "metrics.csv", index=False)

    federation = metrics[metrics.method.isin(["FedAvg", "FedProx"])]
    negative = federation[federation.negative_transfer]
    summary = {
        "status": "PASS_EXECUTION",
        "rounds": rounds,
        "local_epochs": cfg["local_epochs"],
        "tenant_seed_cases": int(len(federation)),
        "negative_transfer_cases": int(len(negative)),
        "negative_transfer_rate": float(federation.negative_transfer.mean()),
        "worst_macro_f1_loss": float(federation.macro_f1_delta_vs_local.min()),
        "mean_loss_among_negative_cases": float(negative.macro_f1_delta_vs_local.mean()) if len(negative) else 0.0,
        "worst_tenant_macro_f1": federation.groupby(["seed", "method"]).macro_f1.min().reset_index().to_dict("records"),
        "suggested_signal_met": bool(
            federation.negative_transfer.mean() >= 0.20 or federation.macro_f1_delta_vs_local.min() <= -0.02
        ),
        "elapsed_seconds": time.time() - start_time,
        "note": "Suggested signal is a decision aid, not a target to engineer."
    }
    write_json(out / "negative_transfer_summary.json", summary)
    manifest = run_manifest("F0.2", args.config, args.artifact_root, device, [pass_file])
    manifest.update(summary)
    write_json(out / "run_manifest.json", manifest)
    write_json(out / "F0_2_PASS.json", {"test": "F0.2", "status": "PASS_EXECUTION",
                                         "suggested_signal_met": summary["suggested_signal_met"]})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
