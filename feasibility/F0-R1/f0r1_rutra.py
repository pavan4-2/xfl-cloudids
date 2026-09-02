from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist

from f0r1_core import (
    aggregate_states,
    artifact_dir,
    checkpoint_path,
    clone_state,
    cosine,
    evaluate_state,
    get_device,
    load_checkpoint,
    load_config,
    load_partitions,
    load_r1_1,
    new_model,
    safe_auc,
    safe_spearman,
    stable_seed,
    state_update_vector,
    write_json,
)


def _predict_with_losses(state: Mapping[str, torch.Tensor], X: np.ndarray, y: np.ndarray,
                         cfg: dict, n_classes: int, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    if len(y) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    model = new_model(X.shape[1], n_classes, cfg, device)
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    model.eval()
    preds, losses = [], []
    bs = min(4096, max(1, cfg["training"]["batch_size"] * 2))
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(np.asarray(X[i:i + bs], dtype=np.float32)).to(device)
            yb = torch.from_numpy(np.asarray(y[i:i + bs], dtype=np.int64)).to(device)
            logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            losses.append(F.cross_entropy(logits, yb, reduction="none").cpu().numpy())
    return np.concatenate(preds), np.concatenate(losses).astype(np.float64)


def _macro_f1_np(y: np.ndarray, pred: np.ndarray) -> float:
    classes = np.unique(y)
    if len(classes) == 0:
        return 0.0
    vals = []
    for c in classes:
        tp = np.sum((y == c) & (pred == c))
        fp = np.sum((y != c) & (pred == c))
        fn = np.sum((y == c) & (pred != c))
        den = 2 * tp + fp + fn
        vals.append(0.0 if den == 0 else (2.0 * tp / den))
    return float(np.mean(vals))


def _recall_np(y: np.ndarray, pred: np.ndarray, cls: int) -> float:
    m = y == cls
    return float(np.mean(pred[m] == cls)) if m.any() else 0.0


def run_r1_3(artifact_root: str, rounds: int | None, windows: int | None) -> dict:
    cfg = load_config()
    meta, data = load_r1_1(artifact_root)
    r2sum = json.loads((Path(artifact_root) / "r1_2" / "negative_transfer_summary.json").read_text())
    rounds = r2sum["rounds"] if rounds is None else rounds
    windows = cfg["utility_windows"] if windows is None else windows
    if not meta.get("synthetic_smoke"):
        if rounds != cfg["training"]["rounds"]:
            raise SystemExit("Refusing R1.3 round override on real data.")
        if windows != cfg["utility_windows"]:
            raise SystemExit("Refusing R1.3 window override on real data.")

    out = artifact_dir(artifact_root, "r1_3")
    device = get_device()
    n_classes = len(meta["labels"])
    n_min = int(cfg["utility_min_examples"])
    rows = []

    for sc in cfg["scenarios"]:
        for seed in cfg["seeds"]:
            parts = load_partitions(artifact_root, sc["name"], seed)
            for rnd in range(rounds):
                ckpt = load_checkpoint(checkpoint_path(artifact_root, sc["name"], seed, rnd))
                local_states = ckpt["local_states"]
                for receiver in range(cfg["tenants"]):
                    ui = parts[f"utility_t{receiver}"]
                    if len(ui) == 0:
                        continue
                    # Preserve source-time order for the windowed utility audit.
                    order = data["order_utility"][ui]
                    ui = ui[np.argsort(order, kind="stable")]
                    y_all = data["y_utility"][ui]
                    local_pred, local_loss = _predict_with_losses(
                        local_states[receiver], data["X_utility"][ui], y_all, cfg, n_classes, device
                    )
                    win_pos = {"overall": np.arange(len(ui), dtype=np.int64)}
                    for wi, arr in enumerate(np.array_split(np.arange(len(ui), dtype=np.int64), windows)):
                        win_pos[str(wi)] = arr
                    for contributor in range(cfg["tenants"]):
                        if contributor == receiver:
                            continue
                        cont_pred, cont_loss = _predict_with_losses(
                            local_states[contributor], data["X_utility"][ui], y_all, cfg, n_classes, device
                        )
                        for win, pos in win_pos.items():
                            if len(pos) == 0:
                                continue
                            yw = y_all[pos]
                            lp, cp = local_pred[pos], cont_pred[pos]
                            ll, cl = local_loss[pos], cont_loss[pos]
                            macro_delta = _macro_f1_np(yw, cp) - _macro_f1_np(yw, lp)
                            rows.append({
                                "scenario": sc["name"], "alpha": sc["alpha"], "seed": seed, "round": rnd,
                                "receiver": receiver, "contributor": contributor, "window": win, "class": "__ALL__",
                                "n_examples": int(len(pos)), "eligible": True,
                                "raw_utility": float(np.mean(ll) - np.mean(cl)),
                                "normalized_utility": np.nan,
                                "local_metric": _macro_f1_np(yw, lp), "contributor_metric": _macro_f1_np(yw, cp),
                                "utility_delta": float(macro_delta),
                            })
                            for cid in sorted(np.unique(yw).tolist()):
                                cm = yw == cid
                                n = int(cm.sum())
                                eligible = n >= n_min
                                raw_u = float(np.mean(ll[cm]) - np.mean(cl[cm])) if eligible else np.nan
                                local_recall = _recall_np(yw, lp, cid)
                                cont_recall = _recall_np(yw, cp, cid)
                                rows.append({
                                    "scenario": sc["name"], "alpha": sc["alpha"], "seed": seed, "round": rnd,
                                    "receiver": receiver, "contributor": contributor, "window": win, "class": meta["labels"][cid],
                                    "n_examples": n, "eligible": bool(eligible), "raw_utility": raw_u,
                                    "normalized_utility": np.nan,
                                    "local_metric": local_recall, "contributor_metric": cont_recall,
                                    "utility_delta": float(cont_recall - local_recall),
                                })
                print(f"R1.3 {sc['name']} seed={seed} round={rnd} complete")

    df = pd.DataFrame(rows)
    # Robust normalization is receiver- and round-specific; cosine is not used here.
    class_mask = (df["class"] != "__ALL__") & df["eligible"].astype(bool) & df["raw_utility"].notna()
    for _, idx in df[class_mask].groupby(["scenario", "seed", "round", "receiver"]).groups.items():
        vals = df.loc[idx, "raw_utility"].to_numpy(dtype=float)
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        scale = 1.4826 * mad
        if scale < 1e-9:
            scale = max(float(np.std(vals)), 1.0)
        df.loc[idx, "normalized_utility"] = (vals - med) / scale
    # Overall rows retain raw utility; normalization is not used for reliability outcomes.
    all_mask = df["class"] == "__ALL__"
    df.loc[all_mask, "normalized_utility"] = df.loc[all_mask, "raw_utility"]
    df.to_csv(out / "utility.csv", index=False)

    eligible = df[(df["class"] != "__ALL__") & df["eligible"].astype(bool) & (df["window"].astype(str) != "overall")]
    scenario_summaries = []
    different_groups, total_groups = 0, 0
    for sc in cfg["scenarios"]:
        gsc = eligible[eligible["scenario"] == sc["name"]]

        # A receiver group is one (scenario, seed, receiver).  The manual's
        # "different top contributors" criterion is evaluated inside that group:
        # if the best contributor changes across eligible class/window contexts,
        # the group contributes one positive count.
        sc_different, sc_groups = 0, 0
        for (seed, receiver), gr in gsc.groupby(["seed", "receiver"]):
            context_tops = []
            for (_, _), gc in gr.groupby(["class", "window"]):
                means = gc.groupby("contributor")["raw_utility"].mean()
                if len(means):
                    context_tops.append(int(means.idxmax()))
            if context_tops:
                sc_groups += 1
                total_groups += 1
                if len(set(context_tops)) >= 2:
                    sc_different += 1
                    different_groups += 1

        # Recipient signal asks whether the same contributors have measurably
        # different utility across receiving tenants in this scenario.
        pivot = gsc.groupby(["receiver", "contributor"])["raw_utility"].mean().unstack()
        dispersion = float(pivot.std(axis=0, ddof=0).mean()) if not pivot.empty else 0.0
        recipient_signal = bool(dispersion > 1e-6)
        scenario_summaries.append({
            "scenario": sc["name"], "recipient_signal": recipient_signal,
            "receiver_groups": int(sc_groups),
            "receiver_groups_with_different_top_contributors": int(sc_different),
            "top_contributor_difference_fraction": float(sc_different / sc_groups) if sc_groups else 0.0,
            "receiver_dispersion": dispersion,
        })
    rare_rows = {
        lab: int(((df["class"] == lab) & df["eligible"].astype(bool)).sum())
        for lab in ["HEARTBLEED", "INFILTRATION"] if lab in meta["labels"]
    }
    summary = {
        "mode": meta["mode"], "windows": windows, "utility_definition": "receiver loss(local) - loss(contributor)",
        "normalization": "receiver/round median-MAD robust normalization", "scenario_summaries": scenario_summaries,
        "receiver_groups": int(total_groups), "different_top_contributor_groups": int(different_groups),
        "different_top_contributor_fraction": float(different_groups / total_groups) if total_groups else 0.0,
        "eligible_rare_class_rows": rare_rows,
    }
    write_json(out / "matrix_summary.json", summary)
    structural_checks = {
        "utility_rows": len(df) > 0,
        "both_scenarios": len(scenario_summaries) == 2,
        "robust_normalization_present": bool(df["normalized_utility"].notna().any()),
        "rare_rows_recorded": bool(rare_rows),
    }
    write_json(out / "R1_3_PASS.json", {
        "status": "PASS" if all(structural_checks.values()) else "FAIL",
        "structural_checks": structural_checks,
    })
    print(f"R1.3 complete -> {out}")
    return summary


def _posterior_from_history(util: pd.DataFrame, scenario: str, seed: int, current_round: int,
                            current_receiver: int, contributor: int, gamma: float, cfg: dict) -> Tuple[float, float, float, int]:
    hist = util[
        (util["scenario"] == scenario) & (util["seed"] == seed) &
        (util["contributor"] == contributor) & (util["receiver"] != current_receiver) &
        (util["round"] < current_round) & (util["class"] != "__ALL__") &
        (util["window"].astype(str) != "overall") & util["eligible"].astype(bool)
    ].copy()
    A = 1.0
    B = 1.0
    n_evidence = 0
    tau = float(cfg["utility_success_threshold"])
    for rr in sorted(hist["round"].unique().tolist()):
        g = hist[hist["round"] == rr]
        # Every class/window item contributes equal evidence mass during F0-R1.
        successes = float(np.sum(g["raw_utility"].to_numpy(dtype=float) > tau))
        total = float(len(g))
        failures = total - successes
        A = gamma * A + successes
        B = gamma * B + failures
        n_evidence += int(total)
    delta = float(cfg["reliability_lower_quantile"])
    lcb = float(beta_dist.ppf(delta, A, B)) if A > 0 and B > 0 else 0.0
    return A, B, lcb, n_evidence


def _prospective_beta_predictions(util: pd.DataFrame, scenario: str, seed: int, gamma: float, cfg: dict) -> pd.DataFrame:
    overall = util[
        (util["scenario"] == scenario) & (util["seed"] == seed) &
        (util["class"] == "__ALL__") & (util["window"].astype(str) == "overall")
    ].copy()
    recs = []
    for _, row in overall.iterrows():
        A, B, lcb, n = _posterior_from_history(
            util, scenario, seed, int(row["round"]), int(row["receiver"]), int(row["contributor"]), gamma, cfg
        )
        if n == 0:
            continue
        actual_delta = float(row["utility_delta"])
        harm = int(actual_delta <= -float(cfg["negative_transfer_margin"]))
        risk = 1.0 - lcb
        recs.append({
            "scenario": scenario, "seed": int(seed), "round": int(row["round"]),
            "receiver": int(row["receiver"]), "contributor": int(row["contributor"]), "gamma": gamma,
            "posterior_A": A, "posterior_B": B, "reliability_lcb": lcb,
            "predicted_utility": lcb, "predicted_harm_risk": risk,
            "actual_utility": actual_delta, "actual_harm": harm, "history_rows": n,
        })
    return pd.DataFrame(recs)


def _metric_pack(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0, "harm_cases": 0, "roc_auc": None, "spearman": None, "brier": None}
    auc = safe_auc(df["actual_harm"], df["predicted_harm_risk"])
    sp = safe_spearman(df["reliability_lcb"], df["actual_utility"])
    brier = float(np.mean((df["predicted_harm_risk"].to_numpy(dtype=float) - df["actual_harm"].to_numpy(dtype=float)) ** 2))
    return {
        "n": int(len(df)), "harm_cases": int(df["actual_harm"].sum()),
        "roc_auc": None if not np.isfinite(auc) else float(auc),
        "spearman": None if not np.isfinite(sp) else float(sp), "brier": brier,
    }


def _bootstrap_reliability(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    if df.empty:
        return {"auc_ci95": [None, None], "spearman_ci95": [None, None], "brier_ci95": [None, None]}
    rng = np.random.default_rng(seed)
    aucs, sps, brs = [], [], []
    n = len(df)
    for _ in range(n_boot):
        b = df.iloc[rng.integers(0, n, n)]
        a = safe_auc(b["actual_harm"], b["predicted_harm_risk"])
        s = safe_spearman(b["reliability_lcb"], b["actual_utility"])
        br = float(np.mean((b["predicted_harm_risk"].to_numpy(dtype=float) - b["actual_harm"].to_numpy(dtype=float)) ** 2))
        if np.isfinite(a):
            aucs.append(float(a))
        if np.isfinite(s):
            sps.append(float(s))
        brs.append(br)

    def ci(v):
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if v else [None, None]

    return {"auc_ci95": ci(aucs), "spearman_ci95": ci(sps), "brier_ci95": ci(brs)}


def run_r1_4(artifact_root: str, bootstrap: int | None) -> dict:
    cfg = load_config()
    meta, _ = load_r1_1(artifact_root)
    bootstrap = cfg["bootstrap"] if bootstrap is None else bootstrap
    if bootstrap != cfg["bootstrap"] and not meta.get("synthetic_smoke"):
        raise SystemExit("Refusing R1.4 bootstrap override on real data.")
    out = artifact_dir(artifact_root, "r1_4")
    util = pd.read_csv(Path(artifact_root) / "r1_3" / "utility.csv")
    util["eligible"] = util["eligible"].astype(str).str.lower().isin(["true", "1"])
    allpred, selection_rows, summaries = [], [], []

    for sc in cfg["scenarios"]:
        candidates = []
        for gamma in cfg["reliability_gamma"]:
            p = _prospective_beta_predictions(util, sc["name"], 17, float(gamma), cfg)
            m = _metric_pack(p)
            candidates.append((float(gamma), p, m))
            selection_rows.append({"scenario": sc["name"], "gamma": gamma, **m})

        def key(item):
            m = item[2]
            auc = m["roc_auc"] if m["roc_auc"] is not None else -1.0
            sp = abs(m["spearman"]) if m["spearman"] is not None else -1.0
            brier = -(m["brier"] if m["brier"] is not None else 9.0)
            return (auc, sp, brier, -item[0])

        gamma, p_sel, m_sel = max(candidates, key=key)
        p_sel = p_sel.copy(); p_sel["phase"] = "selection_seed17"; allpred.append(p_sel)
        p_val = _prospective_beta_predictions(util, sc["name"], 29, gamma, cfg)
        p_val = p_val.copy(); p_val["phase"] = "heldout_seed29"; allpred.append(p_val)
        m_val = _metric_pack(p_val)
        ci = _bootstrap_reliability(p_val, bootstrap, stable_seed("r1_4", sc["name"], gamma))
        summaries.append({
            "scenario": sc["name"], "selected_gamma": gamma,
            "selection_seed17": m_sel, "heldout_seed29": {**m_val, **ci},
        })

    pd.DataFrame(selection_rows).to_csv(out / "gamma_selection.csv", index=False)
    pred = pd.concat(allpred, ignore_index=True) if allpred else pd.DataFrame()
    pred.to_csv(out / "reliability_predictions.csv", index=False)
    summary = {
        "mode": meta["mode"], "bootstrap": bootstrap, "gamma_candidates": cfg["reliability_gamma"],
        "posterior": "discounted Beta(1,1) with lower credible bound",
        "history_rule": "prior rounds only; current receiver excluded from contributor history",
        "scenarios": summaries,
    }
    write_json(out / "reliability_summary.json", summary)
    print(f"R1.4 complete -> {out}")
    return summary


def _reliability_lookup(pred: pd.DataFrame) -> Dict[Tuple[str, int, int, int, int], float]:
    out = {}
    for _, r in pred.iterrows():
        out[(str(r["scenario"]), int(r["seed"]), int(r["round"]), int(r["receiver"]), int(r["contributor"]))] = float(r["reliability_lcb"])
    return out


def _current_utility_matrix(util: pd.DataFrame, scenario: str, seed: int, rnd: int, receiver: int,
                            n_tenants: int) -> Tuple[np.ndarray, np.ndarray, Dict[int, float]]:
    g = util[
        (util["scenario"] == scenario) & (util["seed"] == seed) & (util["round"] == rnd) &
        (util["receiver"] == receiver) & (util["class"] != "__ALL__") &
        (util["window"].astype(str) != "overall") & util["eligible"].astype(bool)
    ].copy()
    contributors = [c for c in range(n_tenants) if c != receiver]
    if g.empty:
        return np.zeros((1, n_tenants), dtype=float), np.array(["none"], dtype=object), {c: 0.0 for c in contributors}
    g["context"] = g["window"].astype(str) + "::" + g["class"].astype(str)
    p = g.pivot_table(index="context", columns="contributor", values="normalized_utility", aggfunc="mean")
    p = p.reindex(columns=contributors).fillna(0.0)
    U = np.zeros((len(p), n_tenants), dtype=float)
    for c in contributors:
        U[:, c] = p[c].to_numpy(dtype=float)
    means = {c: float(g[g["contributor"] == c]["raw_utility"].mean()) if len(g[g["contributor"] == c]) else 0.0 for c in contributors}
    return U, p.index.to_numpy(dtype=object), means


def _cvar(losses: np.ndarray, q: float) -> float:
    if len(losses) == 0:
        return 0.0
    eta = float(np.quantile(losses, q))
    tail = losses[losses >= eta]
    return float(np.mean(tail)) if len(tail) else eta


def _optimize_rutra_weights(U: np.ndarray, reliability: np.ndarray, receiver: int, prev_alpha: np.ndarray,
                            cfg: dict) -> Tuple[np.ndarray, dict]:
    n = len(reliability)
    eps = float(cfg["risk"]["budget"])
    lam = float(cfg["risk"]["cvar_lambda"])
    q = float(cfg["risk"]["q"])
    mu = float(cfg["temporal_mu"])
    maxiter = int(cfg["optimizer"]["max_iterations"])
    tol = float(cfg["optimizer"]["tolerance"])
    risk = 1.0 - np.clip(reliability, 0.0, 1.0)
    risk[receiver] = 0.0

    def objective(a):
        benefit = U @ a
        val_term = -float(np.mean(benefit))
        tail = _cvar(-benefit, q)
        temp = mu * float(np.sum((a - prev_alpha) ** 2))
        return val_term + lam * tail + temp

    cons = [
        {"type": "eq", "fun": lambda a: float(np.sum(a) - 1.0)},
        {"type": "ineq", "fun": lambda a: float(eps - np.dot(a, risk))},
    ]
    x0 = np.asarray(prev_alpha, dtype=float).copy()
    if len(x0) != n or x0.sum() <= 0:
        x0 = np.zeros(n, dtype=float); x0[receiver] = 1.0
    x0 = np.clip(x0, 0.0, 1.0); x0 /= x0.sum()
    if np.dot(x0, risk) > eps:
        x0[:] = 0.0; x0[receiver] = 1.0
    res = minimize(objective, x0, method="SLSQP", bounds=[(0.0, 1.0)] * n, constraints=cons,
                   options={"maxiter": maxiter, "ftol": tol, "disp": False})
    if (not res.success) or not np.isfinite(res.x).all():
        a = np.zeros(n, dtype=float); a[receiver] = 1.0
        status = "FALLBACK_LOCAL"
    else:
        a = np.clip(res.x, 0.0, 1.0)
        if a.sum() <= 0:
            a[:] = 0.0; a[receiver] = 1.0
        else:
            a /= a.sum()
        status = "OPTIMAL" if np.dot(a, risk) <= eps + 1e-6 else "FALLBACK_LOCAL"
        if status == "FALLBACK_LOCAL":
            a[:] = 0.0; a[receiver] = 1.0
    risk_value = float(np.dot(a, risk))
    return a, {
        "optimizer_status": status,
        "optimizer_message": str(getattr(res, "message", "")),
        "objective": float(objective(a)),
        "risk_value": risk_value,
        "risk_budget": eps,
        "constraint_residual": float(eps - risk_value),
    }


def _confusion(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    return np.bincount(y.astype(np.int64) * n_classes + pred.astype(np.int64), minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def _metrics_from_cm(cm: np.ndarray, benign_id: int) -> Tuple[float, float, float]:
    support = cm.sum(axis=1)
    pred_support = cm.sum(axis=0)
    tp = np.diag(cm).astype(float)
    den = 2 * tp + (pred_support - tp) + (support - tp)
    f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
    present = support > 0
    macro = float(np.mean(f1[present])) if present.any() else 0.0
    recalls = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    worst = float(np.min(recalls[present])) if present.any() else 0.0
    if 0 <= benign_id < len(support) and support[benign_id] > 0:
        far = float((support[benign_id] - cm[benign_id, benign_id]) / support[benign_id])
    else:
        far = 0.0
    return macro, far, worst


def _temporal_block_bootstrap_gate(y: np.ndarray, local_pred: np.ndarray, cand_pred: np.ndarray,
                                   n_classes: int, benign_id: int, n_boot: int, seed: int, cfg: dict) -> dict:
    n = len(y)
    if n == 0:
        return {"accept": False, "macro_f1_diff_ci95": [None, None], "far_diff_ci95": [None, None],
                "worst_class_recall_diff_ci95": [None, None], "blocks": 0}
    block_len = max(10, int(math.sqrt(n)))
    blocks = [np.arange(i, min(i + block_len, n), dtype=np.int64) for i in range(0, n, block_len)]
    lcms = [_confusion(y[b], local_pred[b], n_classes) for b in blocks]
    ccms = [_confusion(y[b], cand_pred[b], n_classes) for b in blocks]
    rng = np.random.default_rng(seed)
    md, fd, wd = [], [], []
    B = len(blocks)
    for _ in range(n_boot):
        pick = rng.integers(0, B, B)
        lcm = np.sum([lcms[k] for k in pick], axis=0)
        ccm = np.sum([ccms[k] for k in pick], axis=0)
        lm, lf, lw = _metrics_from_cm(lcm, benign_id)
        cm, cf, cw = _metrics_from_cm(ccm, benign_id)
        md.append(cm - lm); fd.append(cf - lf); wd.append(cw - lw)
    mci = [float(np.quantile(md, .025)), float(np.quantile(md, .975))]
    fci = [float(np.quantile(fd, .025)), float(np.quantile(fd, .975))]
    wci = [float(np.quantile(wd, .025)), float(np.quantile(wd, .975))]
    tol = cfg["gate_tolerances"]
    accept = bool(mci[0] >= float(tol["f1_margin"]) and fci[1] <= float(tol["far"]) and wci[0] >= -float(tol["worst_class_recall"]))
    return {
        "accept": accept,
        "macro_f1_diff_ci95": mci,
        "far_diff_ci95": fci,
        "worst_class_recall_diff_ci95": wci,
        "blocks": B,
    }


def run_r1_5(artifact_root: str, rounds: int | None, gate_bootstrap: int | None) -> dict:
    cfg = load_config()
    meta, data = load_r1_1(artifact_root)
    frozen_rounds = len(cfg["decision_rounds"])
    rounds = frozen_rounds if rounds is None else rounds
    gate_bootstrap = cfg["bootstrap"] if gate_bootstrap is None else gate_bootstrap
    if not meta.get("synthetic_smoke"):
        if rounds != frozen_rounds:
            raise SystemExit("Refusing R1.5 round override on real data.")
        if gate_bootstrap != cfg["bootstrap"]:
            raise SystemExit("Refusing R1.5 gate-bootstrap override on real data.")

    r2sum = json.loads((Path(artifact_root) / "r1_2" / "negative_transfer_summary.json").read_text())
    available_rounds = int(r2sum["rounds"])
    decision = [r for r in cfg["decision_rounds"] if r < available_rounds]
    if meta.get("synthetic_smoke"):
        decision = list(range(max(0, available_rounds - rounds), available_rounds))
    else:
        decision = decision[:rounds]

    util = pd.read_csv(Path(artifact_root) / "r1_3" / "utility.csv")
    util["eligible"] = util["eligible"].astype(str).str.lower().isin(["true", "1"])
    pred = pd.read_csv(Path(artifact_root) / "r1_4" / "reliability_predictions.csv")
    rmap = _reliability_lookup(pred)
    out = artifact_dir(artifact_root, "r1_5")
    device = get_device()
    n_classes = len(meta["labels"])
    benign_id = meta["label_to_id"].get("BENIGN", 0)

    gate_rows, metric_rows, weight_rows = [], [], []
    prev_alpha: Dict[Tuple[str, int, int], np.ndarray] = {}

    for sc in cfg["scenarios"]:
        for seed in cfg["seeds"]:
            parts = load_partitions(artifact_root, sc["name"], seed)
            for rnd in decision:
                ck = load_checkpoint(checkpoint_path(artifact_root, sc["name"], seed, rnd))
                local_states = ck["local_states"]
                base = ck["base_state"]
                for receiver in range(cfg["tenants"]):
                    reliability = np.ones(cfg["tenants"], dtype=float)
                    prior_lcb = float(beta_dist.ppf(cfg["reliability_lower_quantile"], 1.0, 1.0))
                    for c in range(cfg["tenants"]):
                        if c == receiver:
                            reliability[c] = 1.0
                        else:
                            reliability[c] = float(rmap.get((sc["name"], seed, rnd, receiver, c), prior_lcb))
                    U, contexts, mean_raw = _current_utility_matrix(util, sc["name"], seed, rnd, receiver, cfg["tenants"])
                    key = (sc["name"], seed, receiver)
                    prev = prev_alpha.get(key)
                    if prev is None:
                        prev = np.zeros(cfg["tenants"], dtype=float); prev[receiver] = 1.0
                    alpha, opt = _optimize_rutra_weights(U, reliability, receiver, prev, cfg)
                    prev_alpha[key] = alpha.copy()
                    candidate = aggregate_states(local_states, alpha)

                    ui = parts[f"utility_t{receiver}"]
                    gi = parts[f"gate_t{receiver}"]
                    # Gate is ordered temporally by preserved source order before block bootstrap.
                    go = data["order_gate"][gi]
                    gi = gi[np.argsort(go, kind="stable")]
                    local_u = evaluate_state(local_states[receiver], data["X_utility"][ui], data["y_utility"][ui], cfg, n_classes, benign_id, device)
                    cand_u = evaluate_state(candidate, data["X_utility"][ui], data["y_utility"][ui], cfg, n_classes, benign_id, device)
                    local_g = evaluate_state(local_states[receiver], data["X_gate"][gi], data["y_gate"][gi], cfg, n_classes, benign_id, device)
                    cand_g = evaluate_state(candidate, data["X_gate"][gi], data["y_gate"][gi], cfg, n_classes, benign_id, device)
                    fed_g = evaluate_state(ck["fedavg_state"], data["X_gate"][gi], data["y_gate"][gi], cfg, n_classes, benign_id, device)
                    gate_boot = _temporal_block_bootstrap_gate(
                        data["y_gate"][gi], local_g["predictions"], cand_g["predictions"], n_classes, benign_id,
                        gate_bootstrap, stable_seed("gate", sc["name"], seed, rnd, receiver), cfg
                    )
                    accept = bool(gate_boot["accept"] and opt["optimizer_status"] == "OPTIMAL")
                    dep_g = cand_g if accept else local_g
                    pred_change = float(np.mean(cand_g["predictions"] != local_g["predictions"])) if len(gi) else 0.0
                    external_mass = float(1.0 - alpha[receiver])

                    # FedFomo-screening uses current receiver utility; pFedGraph-core uses similarity only.
                    current_overall = util[
                        (util["scenario"] == sc["name"]) & (util["seed"] == seed) & (util["round"] == rnd) &
                        (util["receiver"] == receiver) & (util["class"] == "__ALL__") &
                        (util["window"].astype(str) == "overall")
                    ]
                    fomo_w = np.zeros(cfg["tenants"], dtype=float); fomo_w[receiver] = 0.8
                    positives = []
                    for c in range(cfg["tenants"]):
                        if c == receiver: continue
                        rr = current_overall[current_overall["contributor"] == c]
                        positives.append((c, max(0.0, float(rr["raw_utility"].iloc[0])) if len(rr) else 0.0))
                    s = sum(v for _, v in positives)
                    if s > 0:
                        for c, v in positives: fomo_w[c] = 0.2 * v / s
                    else:
                        fomo_w[receiver] = 1.0
                    fomo = aggregate_states(local_states, fomo_w)
                    fomo_g = evaluate_state(fomo, data["X_gate"][gi], data["y_gate"][gi], cfg, n_classes, benign_id, device)

                    rv = state_update_vector(local_states[receiver], base)
                    sim_w = np.zeros(cfg["tenants"], dtype=float); sim_w[receiver] = 0.8
                    sims = []
                    for c in range(cfg["tenants"]):
                        if c == receiver: continue
                        sims.append((c, max(0.0, cosine(rv, state_update_vector(local_states[c], base)))))
                    ss = sum(v for _, v in sims)
                    if ss > 0:
                        for c, v in sims: sim_w[c] = 0.2 * v / ss
                    else:
                        sim_w[receiver] = 1.0
                    pgraph = aggregate_states(local_states, sim_w)
                    pgraph_g = evaluate_state(pgraph, data["X_gate"][gi], data["y_gate"][gi], cfg, n_classes, benign_id, device)

                    # Offline oracle: unattainable upper-bound screening on held-out gate for headroom only.
                    oracle_ev = local_g; oracle_name = "Local-only"
                    for c in range(cfg["tenants"]):
                        if c == receiver: continue
                        ow = np.zeros(cfg["tenants"], dtype=float); ow[receiver] = 0.8; ow[c] = 0.2
                        st = aggregate_states(local_states, ow)
                        ev = evaluate_state(st, data["X_gate"][gi], data["y_gate"][gi], cfg, n_classes, benign_id, device)
                        if ev["macro_f1"] > oracle_ev["macro_f1"]:
                            oracle_ev = ev; oracle_name = f"pair_t{c}"

                    candidate_delta = float(cand_g["macro_f1"] - local_g["macro_f1"])
                    harmful = candidate_delta <= -float(cfg["negative_transfer_margin"])
                    beneficial = candidate_delta > 0.0
                    nontrivial = bool(
                        pred_change >= float(cfg["nontrivial"]["prediction_change"]) and
                        external_mass >= float(cfg["nontrivial"]["external_weight_mass"])
                    )
                    gate_rows.append({
                        "scenario": sc["name"], "alpha": sc["alpha"], "seed": seed, "round": rnd, "receiver": receiver,
                        "local_f1": local_g["macro_f1"], "candidate_f1": cand_g["macro_f1"], "deployed_f1": dep_g["macro_f1"],
                        "fedavg_f1": fed_g["macro_f1"], "oracle_f1": oracle_ev["macro_f1"], "oracle_choice": oracle_name,
                        "candidate_delta": candidate_delta, "accepted": accept, "harmful": harmful, "beneficial": beneficial,
                        "risk_value": opt["risk_value"], "risk_budget": opt["risk_budget"], "constraint_residual": opt["constraint_residual"],
                        "optimizer_status": opt["optimizer_status"], "external_weight_mass": external_mass,
                        "prediction_change_fraction": pred_change, "nontrivial": nontrivial,
                        "utility_local_f1": local_u["macro_f1"], "utility_candidate_f1": cand_u["macro_f1"],
                        "gate_macro_f1_lcb": gate_boot["macro_f1_diff_ci95"][0], "gate_macro_f1_ucb": gate_boot["macro_f1_diff_ci95"][1],
                        "gate_far_lcb": gate_boot["far_diff_ci95"][0], "gate_far_ucb": gate_boot["far_diff_ci95"][1],
                        "gate_wcr_lcb": gate_boot["worst_class_recall_diff_ci95"][0], "gate_wcr_ucb": gate_boot["worst_class_recall_diff_ci95"][1],
                        "gate_blocks": gate_boot["blocks"],
                        "fedavg_negative_transfer": bool(fed_g["macro_f1"] <= local_g["macro_f1"] - float(cfg["negative_transfer_margin"])),
                    })

                    for method, ev in [
                        ("Local-only", local_g), ("FedAvg", fed_g), ("RUTRA-candidate", cand_g),
                        ("RUTRA-deployed", dep_g), ("Oracle", oracle_ev),
                        ("FedFomo-screening", fomo_g), ("pFedGraph-core-screening", pgraph_g),
                    ]:
                        metric_rows.append({
                            "scenario": sc["name"], "seed": seed, "round": rnd, "receiver": receiver, "method": method,
                            "macro_f1": ev["macro_f1"], "accuracy": ev["accuracy"], "far": ev["far"],
                            "worst_class_recall": ev["worst_class_recall"],
                        })

                    for c in range(cfg["tenants"]):
                        sim = 1.0 if c == receiver else cosine(rv, state_update_vector(local_states[c], base))
                        weight_rows.append({
                            "scenario": sc["name"], "seed": seed, "round": rnd, "receiver": receiver, "contributor": c,
                            "alpha": float(alpha[c]), "reliability_lcb": float(reliability[c]),
                            "reliability_risk": 0.0 if c == receiver else float(1.0 - reliability[c]),
                            "risk_contribution": 0.0 if c == receiver else float(alpha[c] * (1.0 - reliability[c])),
                            "mean_raw_utility": 0.0 if c == receiver else float(mean_raw.get(c, 0.0)),
                            "cosine_diagnostic": float(sim), "risk_value": opt["risk_value"], "risk_budget": opt["risk_budget"],
                            "constraint_residual": opt["constraint_residual"], "optimizer_status": opt["optimizer_status"],
                        })
                print(f"R1.5 {sc['name']} seed={seed} round={rnd} complete")

    gate = pd.DataFrame(gate_rows)
    metrics = pd.DataFrame(metric_rows)
    weights = pd.DataFrame(weight_rows)
    gate.to_csv(out / "gate.csv", index=False)
    metrics.to_csv(out / "metrics.csv", index=False)
    weights.to_csv(out / "weights.csv", index=False)

    harmful = int(gate["harmful"].sum())
    rejected_harm = int((gate["harmful"] & ~gate["accepted"]).sum())
    beneficial = int(gate["beneficial"].sum())
    accepted_ben = int((gate["beneficial"] & gate["accepted"]).sum())
    deployed_harm = int((gate["harmful"] & gate["accepted"]).sum())
    rutra_nt_red = float(1.0 - deployed_harm / harmful) if harmful else 0.0
    fed_harm = int(gate["fedavg_negative_transfer"].sum())
    oracle_harm = int((gate["oracle_f1"] <= gate["local_f1"] - float(cfg["negative_transfer_margin"])).sum())
    oracle_nt_red = float(1.0 - oracle_harm / fed_harm) if fed_harm else 0.0
    head_by_tenant = {int(r): float(np.mean(g["oracle_f1"] - g["local_f1"])) for r, g in gate.groupby("receiver")}
    worst_head = float(min(head_by_tenant.values())) if head_by_tenant else 0.0

    summary = {
        "mode": meta["mode"], "decision_rounds": decision, "receiver_round_cases": int(len(gate)),
        "gate_bootstrap": gate_bootstrap, "harmful_cases": harmful, "harmful_rejected": rejected_harm,
        "harmful_rejection_rate": float(rejected_harm / harmful) if harmful else 0.0,
        "rutra_nt_reduction": rutra_nt_red, "beneficial_cases": beneficial, "beneficial_accepted": accepted_ben,
        "beneficial_acceptance_rate": float(accepted_ben / beneficial) if beneficial else 0.0,
        "accepted_nontrivial_cases": int((gate["accepted"] & gate["nontrivial"]).sum()),
        "fedavg_harmful_cases": fed_harm, "oracle_nt_reduction": oracle_nt_red,
        "worst_tenant_headroom": worst_head, "headroom_by_tenant": head_by_tenant,
        "aggregation": "recipient utility + discounted reliability + CVaR + temporal penalty under reliability-risk budget",
        "gate": "held-out temporal block bootstrap against local reference",
        "screening_note": "FedFomo-screening and pFedGraph-core-screening are diagnostics, not publication-faithful final baselines.",
    }
    write_json(out / "oracle_gate_summary.json", summary)
    print(f"R1.5 complete -> {out}; cases={len(gate)}")
    return summary
