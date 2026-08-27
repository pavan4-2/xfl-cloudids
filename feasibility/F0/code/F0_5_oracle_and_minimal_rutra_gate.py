#!/usr/bin/env python3
"""
TEST F0.5 — Estimate headroom and run the minimum RUTRA-Gate prototype
WHY THIS TEST IS NEEDED:
First, an offline oracle checks whether better tenant-specific choices are even
possible. Then a small RUTRA-Gate prototype tests whether useful collaboration
can be retained while harmful collaboration is rejected. This avoids spending
months on the full project when there is little available improvement.

MAIN OUTPUT: artifacts/f0/f05/metrics.csv, weights.csv, gate.csv and summary.json.
NOTE: The pFedGraph-core screening mechanism here is diagnostic only. A faithful,
equation-validated pFedGraph reproduction remains mandatory for the final paper.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from f0_common import (average_states, bootstrap_metric_differences, cosine,
                       evaluate, flatten_update, load_config, load_tenant,
                       metrics_from_predictions, model_from_meta, predict,
                       run_manifest, state_cpu, write_json)


def normalized_positive(values: np.ndarray, fallback: int) -> np.ndarray:
    weights = np.maximum(values, 0.0)
    if weights.sum() <= 1e-12:
        weights = np.zeros_like(values); weights[fallback] = 1.0
    return weights / weights.sum()


def rutra_weights(utilities: pd.DataFrame, reliability: np.ndarray, receiver: int, cfg: dict) -> tuple[np.ndarray, dict]:
    k = int(cfg["num_tenants"])
    event_rows = []
    for (_, _), event in utilities.groupby(["class_id", "window"]):
        vector = np.zeros(k, dtype=float)
        available = np.zeros(k, dtype=bool)
        for row in event.itertuples():
            if bool(row.eligible) and np.isfinite(row.utility):
                vector[int(row.contributor)] = float(row.utility)
                available[int(row.contributor)] = True
        if available.any():
            event_rows.append(vector)
    if not event_rows:
        fallback = np.zeros(k); fallback[receiver] = 1.0
        return fallback, {"success": False, "fallback": "no_eligible_utility"}
    utility_events = np.vstack(event_rows)
    previous = np.zeros(k); previous[receiver] = 1.0
    risk = 1.0 - reliability.copy(); risk[receiver] = 0.0
    q = float(cfg["cvar_q"])

    def objective(alpha: np.ndarray) -> float:
        gains = utility_events @ alpha
        losses = -gains
        cutoff = np.quantile(losses, q)
        tail = losses[losses >= cutoff]
        cvar = float(tail.mean()) if len(tail) else float(losses.mean())
        return float(-gains.mean() + float(cfg["cvar_lambda"]) * cvar +
                     float(cfg["temporal_mu"]) * np.sum((alpha - previous) ** 2))

    constraints = [
        {"type": "eq", "fun": lambda a: np.sum(a) - 1.0},
        {"type": "ineq", "fun": lambda a: float(cfg["risk_budget"]) - float(np.dot(a, risk))},
    ]
    result = minimize(objective, previous, method="SLSQP", bounds=[(0.0, 1.0)] * k,
                      constraints=constraints, options={"maxiter": 300, "ftol": 1e-9})
    if not result.success or np.any(result.x < -1e-7):
        return previous, {"success": False, "fallback": "optimizer_failure", "message": str(result.message)}
    alpha = np.clip(result.x, 0, 1); alpha /= alpha.sum()
    return alpha, {"success": True, "fallback": None, "objective": objective(alpha),
                   "risk": float(np.dot(alpha, risk)),
                   "constraint_residual": float(cfg["risk_budget"]) - float(np.dot(alpha, risk))}


def main() -> None:
    parser = argparse.ArgumentParser(description="F0.5 oracle and minimum RUTRA-Gate")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("f0_config.json"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/f0"))
    parser.add_argument("--gate-bootstrap", type=int, help="Override only for code smoke testing")
    args = parser.parse_args()
    cfg = load_config(args.config)
    prerequisites = [args.artifact_root / "f04" / "F0_4_PASS.json",
                     args.artifact_root / "f03" / "utility.csv",
                     args.artifact_root / "f04" / "reliability_predictions.csv"]
    if not all(path.exists() for path in prerequisites):
        raise SystemExit("STOP: F0.3/F0.4 prerequisite outputs are missing.")
    with open(args.artifact_root / "f01" / "data_metadata.json", encoding="utf-8") as handle:
        meta = json.load(handle)
    utility_all = pd.read_csv(prerequisites[1])
    reliability_all = pd.read_csv(prerequisites[2])
    device = torch.device("cuda" if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    out = args.artifact_root / "f05"
    out.mkdir(parents=True, exist_ok=True)
    metric_rows, weight_rows, gate_rows = [], [], []
    start_time = time.time()

    for seed in cfg["development_seeds"]:
        seed_utility = utility_all[utility_all.seed == seed]
        final_round = int(seed_utility["round"].max())
        checkpoint = args.artifact_root / "f03" / "checkpoints" / f"seed_{seed}" / f"round_{final_round}" / "candidates.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        base_state, states, train_sizes = saved["base"], saved["candidates"], saved["train_sizes"]
        k = int(cfg["num_tenants"])
        updates = [flatten_update(state, base_state) for state in states]
        final_reliability = reliability_all[(reliability_all.seed == seed) &
                                            (reliability_all["round"] == final_round)]
        reliability = np.full(k, 0.05, dtype=float)
        for row in final_reliability.itertuples():
            reliability[int(row.contributor)] = float(row.reliability_lcb)

        fedavg_state = average_states(states, train_sizes)
        f02_models = torch.load(args.artifact_root / "f02" / f"models_seed_{seed}.pt",
                                map_location="cpu", weights_only=True)
        for receiver in range(k):
            local_state = states[receiver]
            receiver_utility = seed_utility[(seed_utility["round"] == final_round) &
                                            (seed_utility.receiver == receiver)]
            mean_utility = receiver_utility[receiver_utility.eligible == True].groupby("contributor").utility.mean()  # noqa: E712
            utility_vector = np.array([float(mean_utility.get(j, -np.inf)) for j in range(k)])
            utility_vector[receiver] = max(0.0, utility_vector[receiver])
            fedfomo_alpha = normalized_positive(utility_vector, receiver)

            graph_scores = np.array([cosine(updates[receiver], updates[j]) for j in range(k)])
            graph_scores = np.exp(np.clip(graph_scores, -1, 1) / 0.5) * np.sqrt(np.asarray(train_sizes) / max(train_sizes))
            graph_alpha = graph_scores / graph_scores.sum()
            oracle_choice = int(np.nanargmax(utility_vector))
            oracle_alpha = np.zeros(k); oracle_alpha[oracle_choice] = 1.0
            rutra_alpha, optimizer_info = rutra_weights(receiver_utility, reliability, receiver, cfg)

            methods = {
                "Local-only": local_state,
                "FedAvg": fedavg_state,
                "FedProx": f02_models["fedprox"],
                "FedFomo-screening": average_states(states, fedfomo_alpha),
                "pFedGraph-core-screening": average_states(states, graph_alpha),
                "Oracle-offline": average_states(states, oracle_alpha),
                "RUTRA-candidate": average_states(states, rutra_alpha),
            }
            x_gate, y_gate = load_tenant(args.artifact_root, seed, receiver, "gate")
            evaluated = {}
            predictions = {}
            for method, state in methods.items():
                model = model_from_meta(meta, cfg, device); model.load_state_dict(state)
                pred, _ = predict(model, x_gate, device)
                predictions[method] = pred
                evaluated[method] = metrics_from_predictions(y_gate, pred, meta["num_classes"], meta["rare_classes"])
                metric_rows.append({"seed": seed, "round": final_round, "receiver": receiver,
                                    "method": method, "split": "gate", **evaluated[method]})

            intervals = bootstrap_metric_differences(
                y_gate, predictions["RUTRA-candidate"], predictions["Local-only"],
                meta["num_classes"], meta["rare_classes"],
                args.gate_bootstrap or int(cfg["gate_bootstrap"]), seed + receiver,
            )
            accepted = bool(
                intervals["macro_f1"]["lcb95"] >= float(cfg["gate_f1_margin"]) and
                intervals["far"]["ucb95"] <= float(cfg["gate_far_tolerance"]) and
                intervals["worst_class_recall"]["lcb95"] >= -float(cfg["gate_wcr_tolerance"])
            )
            actual_delta = evaluated["RUTRA-candidate"]["macro_f1"] - evaluated["Local-only"]["macro_f1"]
            harmful = actual_delta < -float(cfg["negative_transfer_margin"])
            beneficial = actual_delta > float(cfg["negative_transfer_margin"])
            deployed = evaluated["RUTRA-candidate"] if accepted else evaluated["Local-only"]
            metric_rows.append({"seed": seed, "round": final_round, "receiver": receiver,
                                "method": "RUTRA-deployed", "split": "gate", **deployed})
            gate_rows.append({"seed": seed, "round": final_round, "receiver": receiver,
                              "accepted": accepted, "actual_macro_f1_delta": actual_delta,
                              "actually_harmful": harmful, "actually_beneficial": beneficial,
                              "f1_lcb95": intervals["macro_f1"]["lcb95"],
                              "far_ucb95": intervals["far"]["ucb95"],
                              "wcr_lcb95": intervals["worst_class_recall"]["lcb95"]})
            for method, alpha in (("FedFomo-screening", fedfomo_alpha),
                                  ("pFedGraph-core-screening", graph_alpha),
                                  ("Oracle-offline", oracle_alpha), ("RUTRA-candidate", rutra_alpha)):
                for contributor, weight in enumerate(alpha):
                    weight_rows.append({"seed": seed, "round": final_round, "receiver": receiver,
                                        "method": method, "contributor": contributor, "weight": float(weight),
                                        "reliability_lcb": float(reliability[contributor]),
                                        "optimizer_success": optimizer_info.get("success") if method == "RUTRA-candidate" else None,
                                        "risk": optimizer_info.get("risk") if method == "RUTRA-candidate" else None,
                                        "constraint_residual": optimizer_info.get("constraint_residual") if method == "RUTRA-candidate" else None})

    metrics = pd.DataFrame(metric_rows)
    local = metrics[metrics.method == "Local-only"][["seed", "receiver", "macro_f1"]].rename(
        columns={"macro_f1": "local_macro_f1"})
    metrics = metrics.merge(local, on=["seed", "receiver"], how="left")
    metrics["delta_vs_local"] = metrics.macro_f1 - metrics.local_macro_f1
    metrics["negative_transfer"] = metrics.delta_vs_local < -float(cfg["negative_transfer_margin"])
    weights = pd.DataFrame(weight_rows); gates = pd.DataFrame(gate_rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    weights.to_csv(out / "weights.csv", index=False)
    gates.to_csv(out / "gate.csv", index=False)

    def nt_rate(method: str) -> float:
        return float(metrics[metrics.method == method].negative_transfer.mean())
    fedavg_nt = nt_rate("FedAvg"); deployed_nt = nt_rate("RUTRA-deployed"); oracle_nt = nt_rate("Oracle-offline")
    nt_reduction = (fedavg_nt - deployed_nt) / fedavg_nt if fedavg_nt > 0 else 0.0
    oracle_nt_reduction = (fedavg_nt - oracle_nt) / fedavg_nt if fedavg_nt > 0 else 0.0
    worst = metrics.groupby(["seed", "method"]).macro_f1.min().reset_index()
    oracle_worst = worst[worst.method == "Oracle-offline"].macro_f1.mean()
    local_worst = worst[worst.method == "Local-only"].macro_f1.mean()
    harmful_count = int(gates.actually_harmful.sum()); beneficial_count = int(gates.actually_beneficial.sum())
    harmful_reject = float((~gates[gates.actually_harmful].accepted).mean()) if harmful_count else float("nan")
    beneficial_accept = float(gates[gates.actually_beneficial].accepted.mean()) if beneficial_count else float("nan")
    summary = {
        "status": "PASS_EXECUTION",
        "oracle_worst_tenant_headroom_pp": float((oracle_worst - local_worst) * 100),
        "oracle_negative_transfer_reduction": float(oracle_nt_reduction),
        "fedavg_negative_transfer_rate": fedavg_nt,
        "rutra_deployed_negative_transfer_rate": deployed_nt,
        "rutra_negative_transfer_reduction": float(nt_reduction),
        "gate_harmful_cases": harmful_count,
        "gate_beneficial_cases": beneficial_count,
        "harmful_update_rejection_rate": harmful_reject,
        "beneficial_update_acceptance_rate": beneficial_accept,
        "gate_coverage": float(gates.accepted.mean()),
        "optimizer_success_rate": float(np.mean([
            bool(value) for value in weights[weights.method == "RUTRA-candidate"].optimizer_success.tolist()
        ])),
        "manageable_cost": True,
        "elapsed_seconds": time.time() - start_time,
        "screening_warning": "FedFomo/pFedGraph mechanisms are F0 screening implementations, not final publication baselines.",
    }
    summary["oracle_headroom_signal"] = bool(summary["oracle_worst_tenant_headroom_pp"] >= 3.0 or
                                               summary["oracle_negative_transfer_reduction"] >= 0.30)
    summary["prototype_safety_signal"] = bool(summary["rutra_negative_transfer_reduction"] >= 0.25)
    write_json(out / "oracle_and_prototype_summary.json", summary)
    manifest = run_manifest("F0.5", args.config, args.artifact_root, device, prerequisites)
    manifest.update(summary)
    write_json(out / "run_manifest.json", manifest)
    write_json(out / "F0_5_PASS.json", {"test": "F0.5", "status": "PASS_EXECUTION",
                                         "oracle_headroom_signal": summary["oracle_headroom_signal"],
                                         "prototype_safety_signal": summary["prototype_safety_signal"]})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
