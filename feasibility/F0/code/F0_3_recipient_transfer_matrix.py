#!/usr/bin/env python3
"""
TEST F0.3 — Build the receiver–contributor transfer-utility matrix
WHY THIS TEST IS NEEDED:
A contributor may help Tenant A but harm Tenant B. RUTRA-Gate needs this
receiver-specific pattern to exist. This test trains one candidate per tenant,
evaluates every candidate on every receiver's utility data, and records the
benefit or harm separately for each attack class and time window.

MAIN OUTPUT: artifacts/f0/f03/utility.csv, matrix_summary.json and checkpoints.
Positive utility means helpful; negative utility means harmful.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from f0_common import (average_states, class_losses, flatten_update, load_config, load_tenant,
                       model_from_meta, run_manifest, set_seed, state_cpu,
                       train_model, write_json)


def main() -> None:
    parser = argparse.ArgumentParser(description="F0.3 recipient-conditioned utility matrix")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("f0_config.json"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/f0"))
    parser.add_argument("--rounds", type=int, help="Override only for code smoke testing")
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, help="Override only for code smoke testing")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.local_epochs is not None:
        cfg["local_epochs"] = args.local_epochs
    rounds = args.rounds or int(cfg["federated_rounds"])
    prerequisite = args.artifact_root / "f02" / "F0_2_PASS.json"
    if not prerequisite.exists():
        raise SystemExit("STOP: F0.2 execution PASS file is missing.")
    with open(args.artifact_root / "f01" / "data_metadata.json", encoding="utf-8") as handle:
        meta = json.load(handle)
    device = torch.device("cuda" if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    out = args.artifact_root / "f03"
    checkpoint_root = out / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.time()

    for seed in cfg["development_seeds"]:
        set_seed(seed)
        global_model = model_from_meta(meta, cfg, device)
        for round_id in range(rounds):
            base_state = state_cpu(global_model)
            candidate_states, train_sizes = [], []
            for contributor in range(cfg["num_tenants"]):
                x_train, y_train = load_tenant(args.artifact_root, seed, contributor, "train")
                candidate = model_from_meta(meta, cfg, device)
                candidate.load_state_dict(base_state)
                train_model(candidate, x_train, y_train, cfg, device,
                            seed + round_id * 100 + contributor)
                candidate_states.append(state_cpu(candidate))
                train_sizes.append(len(y_train))

            update_norms = np.asarray([np.linalg.norm(flatten_update(state, base_state))
                                       for state in candidate_states], dtype=float)
            norm_median = float(np.median(update_norms))
            norm_mad = float(np.median(np.abs(update_norms - norm_median)))
            norm_limit = norm_median + 3.0 * max(norm_mad, 1e-12)
            integrity_pass = np.isfinite(update_norms) & (update_norms <= norm_limit)

            checkpoint_dir = checkpoint_root / f"seed_{seed}" / f"round_{round_id}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"base": base_state, "candidates": candidate_states,
                        "train_sizes": train_sizes}, checkpoint_dir / "candidates.pt")

            for receiver in range(cfg["num_tenants"]):
                x_val, y_val = load_tenant(args.artifact_root, seed, receiver, "utility")
                window_indices = np.array_split(np.arange(len(y_val)), args.windows)
                reference = model_from_meta(meta, cfg, device)
                reference.load_state_dict(candidate_states[receiver])
                for window, idx in enumerate(window_indices):
                    if len(idx) == 0:
                        continue
                    reference_loss = class_losses(reference, x_val[idx], y_val[idx], meta["num_classes"], device)
                    for contributor, state in enumerate(candidate_states):
                        candidate = model_from_meta(meta, cfg, device)
                        candidate.load_state_dict(state)
                        candidate_loss = class_losses(candidate, x_val[idx], y_val[idx], meta["num_classes"], device)
                        for cls in range(meta["num_classes"]):
                            count = int(np.sum(y_val[idx] == cls))
                            eligible = count >= int(cfg["min_class_examples"])
                            utility = float(reference_loss[cls] - candidate_loss[cls]) if eligible else np.nan
                            rows.append({"seed": seed, "round": round_id, "receiver": receiver,
                                         "contributor": contributor, "class_id": cls,
                                         "class_name": meta["class_names"][cls], "window": window,
                                         "examples": count, "eligible": eligible, "utility": utility,
                                         "update_norm": float(update_norms[contributor]),
                                         "integrity_pass": bool(integrity_pass[contributor])})
            global_model.load_state_dict(average_states(candidate_states, train_sizes))

    utility = pd.DataFrame(rows)
    utility.to_csv(out / "utility.csv", index=False)
    eligible = utility[(utility.eligible) & (utility.receiver != utility.contributor)].copy()
    eligible["sign"] = np.sign(eligible.utility)

    recipient_groups = eligible.groupby(["seed", "round", "contributor", "class_id", "window"]).sign
    recipient_variation = recipient_groups.apply(lambda s: (s.min() < 0) and (s.max() > 0))
    class_groups = eligible.groupby(["seed", "round", "receiver", "contributor", "window"]).sign
    class_variation = class_groups.apply(lambda s: (s.min() < 0) and (s.max() > 0))
    means = eligible.groupby(["seed", "round", "receiver", "contributor"], as_index=False).utility.mean()
    top = means.loc[means.groupby(["seed", "round", "receiver"]).utility.idxmax()]
    ranking_var = top.groupby(["seed", "round"]).contributor.nunique()
    temporal = eligible.sort_values("round")
    temporal["previous_sign"] = temporal.groupby(
        ["seed", "receiver", "contributor", "class_id", "window"]
    ).sign.shift(1)
    comparable = temporal[(temporal.sign != 0) & temporal.previous_sign.notna() & (temporal.previous_sign != 0)]
    persistence = float((comparable.sign == comparable.previous_sign).mean()) if len(comparable) else float("nan")
    summary = {
        "status": "PASS_EXECUTION",
        "rounds": rounds,
        "eligible_utility_rows": int(len(eligible)),
        "positive_fraction": float((eligible.utility > 0).mean()) if len(eligible) else 0.0,
        "negative_fraction": float((eligible.utility < 0).mean()) if len(eligible) else 0.0,
        "receiver_sign_variation_fraction": float(recipient_variation.mean()) if len(recipient_variation) else 0.0,
        "class_sign_variation_fraction": float(class_variation.mean()) if len(class_variation) else 0.0,
        "rounds_with_different_top_contributors_fraction": float((ranking_var > 1).mean()) if len(ranking_var) else 0.0,
        "adjacent_round_sign_persistence": persistence,
        "recipient_conditioning_signal": bool(
            len(eligible) > 0 and (recipient_variation.mean() > 0 or (ranking_var > 1).mean() >= 0.25)
        ),
        "elapsed_seconds": time.time() - start,
    }
    write_json(out / "matrix_summary.json", summary)
    manifest = run_manifest("F0.3", args.config, args.artifact_root, device, [prerequisite])
    manifest.update(summary)
    write_json(out / "run_manifest.json", manifest)
    write_json(out / "F0_3_PASS.json", {"test": "F0.3", "status": "PASS_EXECUTION",
                                         "recipient_conditioning_signal": summary["recipient_conditioning_signal"]})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
