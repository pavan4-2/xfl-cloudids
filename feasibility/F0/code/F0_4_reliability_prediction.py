#!/usr/bin/env python3
"""
TEST F0.4 — Check whether past reliability predicts future harm
WHY THIS TEST IS NEEDED:
RUTRA-Gate uses a contributor's earlier behaviour to decide how much to trust
its next update. This test strictly uses rounds before t to predict what happens
in round t. It reports ROC-AUC, Brier score and Spearman correlation. If past
behaviour has no forward signal, the reliability component is not justified.

MAIN OUTPUT: artifacts/f0/f04/reliability_predictions.csv and summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import beta as beta_distribution
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, roc_auc_score

from f0_common import load_config, run_manifest, write_json


def safe_metrics(frame: pd.DataFrame) -> dict:
    if len(frame) < 4:
        return {"roc_auc_harm": float("nan"), "brier_success": float("nan"),
                "spearman_reliability_utility": float("nan")}
    harm = frame.harmful.astype(int).to_numpy()
    success = 1 - harm
    auc = roc_auc_score(harm, 1.0 - frame.reliability_lcb) if len(np.unique(harm)) == 2 else float("nan")
    brier = brier_score_loss(success, frame.reliability_mean)
    rho = spearmanr(frame.reliability_lcb, frame.next_utility, nan_policy="omit").statistic
    return {"roc_auc_harm": float(auc), "brier_success": float(brier),
            "spearman_reliability_utility": float(rho)}


def bootstrap_intervals(frame: pd.DataFrame, resamples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    values = {key: [] for key in safe_metrics(frame)}
    for _ in range(resamples):
        sampled = frame.iloc[rng.integers(0, len(frame), len(frame))]
        metrics = safe_metrics(sampled)
        for key, value in metrics.items():
            if np.isfinite(value):
                values[key].append(value)
    output = {}
    for key, samples in values.items():
        output[key] = {
            "lcb95": float(np.quantile(samples, 0.025)) if samples else float("nan"),
            "ucb95": float(np.quantile(samples, 0.975)) if samples else float("nan"),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="F0.4 prior-round reliability prediction")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("f0_config.json"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/f0"))
    parser.add_argument("--bootstrap", type=int, help="Override only for code smoke testing")
    args = parser.parse_args()
    cfg = load_config(args.config)
    prerequisite = args.artifact_root / "f03" / "F0_3_PASS.json"
    utility_path = args.artifact_root / "f03" / "utility.csv"
    if not prerequisite.exists() or not utility_path.exists():
        raise SystemExit("STOP: F0.3 outputs are missing.")
    utility = pd.read_csv(utility_path)
    utility = utility[(utility.eligible == True) & (utility.receiver != utility.contributor)].copy()  # noqa: E712
    out = args.artifact_root / "f04"
    out.mkdir(parents=True, exist_ok=True)
    gamma = float(cfg["reliability_gamma"])
    quantile = float(cfg["reliability_quantile"])
    records = []

    for seed in cfg["development_seeds"]:
        seed_data = utility[utility.seed == seed]
        posterior = {j: [1.0, 1.0] for j in range(cfg["num_tenants"])}
        for round_id in sorted(seed_data["round"].unique()):
            current = seed_data[seed_data["round"] == round_id]
            for contributor in range(cfg["num_tenants"]):
                a, b = posterior[contributor]
                rel_mean = a / (a + b)
                rel_lcb = float(beta_distribution.ppf(quantile, a, b))
                contributor_rows = current[current.contributor == contributor]
                observations = contributor_rows.utility.dropna().to_numpy()
                if len(observations) == 0:
                    continue
                next_utility = float(np.mean(observations))
                records.append({"seed": seed, "round": int(round_id), "contributor": contributor,
                                "prior_a": a, "prior_b": b, "reliability_mean": rel_mean,
                                "reliability_lcb": rel_lcb, "next_utility": next_utility,
                                "harmful": bool(next_utility < 0), "observations": len(observations)})
            # Update only after predictions for this round have been recorded.
            for contributor in range(cfg["num_tenants"]):
                contributor_rows = current[current.contributor == contributor]
                observations = contributor_rows.utility.dropna().to_numpy()
                passed = bool(contributor_rows.integrity_pass.all()) if "integrity_pass" in contributor_rows else True
                successes = float(np.sum(observations > 0)) if passed else 0.0
                failures = float(np.sum(observations <= 0)) if passed else float(len(observations))
                a, b = posterior[contributor]
                posterior[contributor] = [gamma * a + successes, gamma * b + failures]

    predictions = pd.DataFrame(records)
    # Round 0 contains only the identical Beta(1,1) prior and is not evidence of predictability.
    evaluated = predictions[predictions["round"] > predictions["round"].min()].copy()
    if evaluated.empty:
        raise RuntimeError("At least two utility rounds are required for reliability prediction")
    predictions.to_csv(out / "reliability_predictions.csv", index=False)
    metrics = safe_metrics(evaluated)
    intervals = bootstrap_intervals(evaluated, args.bootstrap or int(cfg["reliability_bootstrap"]),
                                    cfg["development_seeds"][0])
    meaningful = bool(
        (np.isfinite(metrics["roc_auc_harm"]) and metrics["roc_auc_harm"] >= 0.70) or
        (np.isfinite(metrics["spearman_reliability_utility"]) and
         abs(metrics["spearman_reliability_utility"]) >= 0.40)
    )
    summary = {
        "status": "PASS_EXECUTION",
        "evaluated_contributor_rounds": int(len(evaluated)),
        **metrics,
        "bootstrap_95_intervals": intervals,
        "meaningful_forward_signal": meaningful,
        "prior_round_only_verified": True,
        "note": "AUC near 0.70 or absolute Spearman near 0.40 is a decision aid, not a target to engineer."
    }
    write_json(out / "reliability_summary.json", summary)
    manifest = run_manifest("F0.4", args.config, args.artifact_root, torch.device("cpu"),
                            [prerequisite, utility_path])
    manifest.update(summary)
    write_json(out / "run_manifest.json", manifest)
    write_json(out / "F0_4_PASS.json", {"test": "F0.4", "status": "PASS_EXECUTION",
                                         "meaningful_forward_signal": meaningful})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
