from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier

from common import dump_json, seed_all, system_environment
from r1_common import emit_metrics, load_r1_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--system", choices=["SYSTEM5"], default="SYSTEM5")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    spec = json.loads((root / "RECOVERY_SPECIFICATION.json").read_text())
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    seed = int(spec["calibration_seed"]); seed_all(seed)

    ds, train, validation, duplicate_mask, clean_mask = load_r1_data(Path(args.bridge).resolve(), spec)
    cfg = spec["r1_system5_model"]
    env = system_environment(); env.update({"step":"R1", "system":"SYSTEM5", "role":"centralized_tree_ceiling"})
    dump_json(out / "R1_SYSTEM5_ENVIRONMENT.json", env)

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=float(cfg["learning_rate"]),
        max_iter=int(cfg["max_iter"]),
        max_leaf_nodes=int(cfg["max_leaf_nodes"]),
        min_samples_leaf=int(cfg["min_samples_leaf"]),
        l2_regularization=float(cfg["l2_regularization"]),
        early_stopping=bool(cfg["early_stopping"]),
        validation_fraction=float(cfg["validation_fraction"]),
        n_iter_no_change=int(cfg["n_iter_no_change"]),
        random_state=seed,
    )
    t0 = time.perf_counter()
    model.fit(train.X, train.y)
    fit_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    pred = model.predict(validation.X)
    predict_seconds = time.perf_counter() - t1

    result = emit_metrics(
        out,
        "HIST_GRADIENT_BOOSTING",
        ds,
        validation,
        pred,
        duplicate_mask,
        spec["r1_ceiling_gates"]["system5_tree"],
    )
    summary = {
        "step":"R1",
        "system":"SYSTEM5",
        "status":"TECHNICAL_PASS",
        "experiment":"centralized_performance_ceiling",
        "calibration_seed":seed,
        "canonical_manifest":spec["canonical_manifest"],
        "train_rows":int(len(train.y)),
        "raw_validation_rows":int(len(validation.y)),
        "deduplicated_validation_rows":int(clean_mask.sum()),
        "model":"HistGradientBoostingClassifier",
        "fit_seconds":fit_seconds,
        "predict_seconds":predict_seconds,
        "n_iter_":int(getattr(model,"n_iter_",-1)),
        "primary_metrics":result["primary_deduplicated_validation"],
        "ceiling_gate_pass":result["ceiling_gate_pass"],
        "sealed_final_test_access":False,
        "automatic_r2_authorization":False,
        "scientific_interpretation":"SUPERVISOR_ONLY"
    }
    dump_json(out / "R1_SYSTEM5_SUMMARY.json", summary)
    print(json.dumps({"R1_SYSTEM5":summary}, sort_keys=True))

if __name__ == "__main__":
    main()
