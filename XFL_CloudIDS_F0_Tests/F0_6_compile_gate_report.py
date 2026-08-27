#!/usr/bin/env python3
"""
TEST F0.6 — Compile the evidence and recommend GREEN, CONDITIONAL or RED
WHY THIS TEST IS NEEDED:
The earlier tests produce many files. This final test checks that the evidence
is complete, verifies that the sealed final-test file has not changed, applies
the pre-declared decision aids, and creates one report for the supervisor.
The script's recommendation is not final: only the supervisor signs Gate F0.

MAIN OUTPUT: artifacts/f0/f06/F0_GATE_DECISION_REPORT.md and
F0_RESULTS_FOR_SUPERVISOR.zip. Never include the sealed final-test data in ZIP.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd
import torch

from f0_common import load_config, run_manifest, sha256_file, write_json


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="F0.6 feasibility evidence and decision report")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("f0_config.json"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/f0"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    required = {
        "F0.1": args.artifact_root / "f01" / "F0_1_PASS.json",
        "F0.2": args.artifact_root / "f02" / "negative_transfer_summary.json",
        "F0.3": args.artifact_root / "f03" / "matrix_summary.json",
        "F0.4": args.artifact_root / "f04" / "reliability_summary.json",
        "F0.5": args.artifact_root / "f05" / "oracle_and_prototype_summary.json",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise SystemExit("STOP: missing required evidence:\n" + "\n".join(missing))
    f01, f02, f03, f04, f05 = [read_json(required[key]) for key in required]
    seal_path = args.artifact_root / "f01" / "untouched_final_test.SEAL.json"
    test_path = args.artifact_root / "f01" / "untouched_final_test.npz"
    seal = read_json(seal_path)
    seal_unchanged = test_path.exists() and sha256_file(test_path) == seal["sha256"]

    criteria = {
        "Real CIC-IDS2017 evidence (not synthetic smoke data)": bool(f01.get("research_evidence", False)),
        "Material negative transfer exists": bool(f02["suggested_signal_met"]),
        "Utility depends on receiver/class": bool(f03["recipient_conditioning_signal"]),
        "Reliability has forward predictive value": bool(f04["meaningful_forward_signal"]),
        "Oracle shows useful headroom": bool(f05["oracle_headroom_signal"]),
        "Minimal prototype reduces negative transfer": bool(f05["prototype_safety_signal"]),
        "Sealed final-test hash is unchanged": bool(seal_unchanged),
        "Computational cost is manageable": bool(f05["manageable_cost"]),
    }
    rate_checks = {}
    if f05["gate_harmful_cases"] > 0:
        rate_checks["At least 70% harmful-update rejection"] = bool(f05["harmful_update_rejection_rate"] >= 0.70)
    else:
        rate_checks["At least 70% harmful-update rejection"] = None
    if f05["gate_beneficial_cases"] > 0:
        rate_checks["At least 60% beneficial-update acceptance"] = bool(f05["beneficial_update_acceptance_rate"] >= 0.60)
    else:
        rate_checks["At least 60% beneficial-update acceptance"] = None

    core_green = all(criteria.values())
    rate_green = all(value is True for value in rate_checks.values())
    if core_green and rate_green:
        recommendation = "GREEN"
    elif sum(value is True for value in criteria.values()) >= 5 and seal_unchanged:
        recommendation = "CONDITIONAL"
    else:
        recommendation = "RED"

    out = args.artifact_root / "f06"
    out.mkdir(parents=True, exist_ok=True)
    decision = {
        "automated_recommendation": recommendation,
        "final_supervisor_decision": "PENDING",
        "criteria": criteria,
        "rate_checks": rate_checks,
        "no_final_test_data_used_declaration": "STUDENT MUST CONFIRM AND SIGN",
        "green_authorization": "Only a supervisor-signed GREEN authorizes Phases 1-10 unchanged.",
    }
    write_json(out / "gate_recommendation.json", decision)

    lines = [
        "# XFL-CloudIDS Phase F0 Gate Decision Report",
        "",
        f"Automated recommendation: **{recommendation}**",
        "",
        "> This is an evidence-based recommendation, not the final decision. The supervisor must inspect the attached files and sign below.",
        "",
        "## Evidence checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for name, value in {**criteria, **rate_checks}.items():
        result = "PASS" if value is True else ("FAIL" if value is False else "INSUFFICIENT CASES")
        lines.append(f"| {name} | {result} |")
    lines.extend([
        "", "## Key numerical evidence", "",
        f"- Negative-transfer rate in F0.2: {f02['negative_transfer_rate']:.3f}",
        f"- Worst F0.2 macro-F1 loss: {f02['worst_macro_f1_loss']:.4f}",
        f"- Receiver sign-variation fraction: {f03['receiver_sign_variation_fraction']:.3f}",
        f"- Reliability harm ROC-AUC: {f04['roc_auc_harm']:.3f}",
        f"- Reliability/next-utility Spearman: {f04['spearman_reliability_utility']:.3f}",
        f"- Oracle worst-tenant headroom: {f05['oracle_worst_tenant_headroom_pp']:.2f} percentage points",
        f"- Prototype negative-transfer reduction: {f05['rutra_negative_transfer_reduction']:.3f}",
        f"- Harmful-update rejection: {f05['harmful_update_rejection_rate']}",
        f"- Beneficial-update acceptance: {f05['beneficial_update_acceptance_rate']}",
        f"- Gate coverage: {f05['gate_coverage']:.3f}",
        "", "## Mandatory declarations", "",
        "- [ ] Students confirm that no untouched final-test records or metrics were opened or used during F0.",
        "- [ ] Students confirm that unsuccessful and failed runs are included in the submitted logs.",
        "- [ ] Students confirm that thresholds were not changed after viewing results.",
        "- [ ] Supervisor has verified the sealed final-test SHA-256 value.",
        "", "## Supervisor decision", "",
        "Final decision: GREEN / CONDITIONAL / RED (circle one)",
        "", "Reason and conditions: ________________________________________________",
        "", "Supervisor name/signature: __________________  Date: ________________",
        "", "A signed GREEN decision authorizes Phases 1-10 without changing the frozen manual.",
    ])
    report_path = out / "F0_GATE_DECISION_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = run_manifest("F0.6", args.config, args.artifact_root, torch.device("cpu"), list(required.values()))
    manifest.update({"status": "PASS_EXECUTION", **decision})
    write_json(out / "run_manifest.json", manifest)
    write_json(out / "F0_6_PASS.json", {"test": "F0.6", "status": "PASS_EXECUTION",
                                         "automated_recommendation": recommendation,
                                         "final_supervisor_decision": "PENDING"})

    evidence_files = []
    for folder in ("f01", "f02", "f03", "f04", "f05", "f06"):
        for path in sorted((args.artifact_root / folder).glob("*")):
            if (path.is_file() and path.name not in {"untouched_final_test.npz", "evidence_checksums.csv"}
                    and path.suffix in {".json", ".csv", ".md"}):
                evidence_files.append(path)
    checksum_rows = [{"relative_path": str(p.relative_to(args.artifact_root)), "sha256": sha256_file(p),
                      "bytes": p.stat().st_size} for p in evidence_files]
    pd.DataFrame(checksum_rows).to_csv(out / "evidence_checksums.csv", index=False)
    evidence_files.append(out / "evidence_checksums.csv")
    zip_path = out / "F0_RESULTS_FOR_SUPERVISOR.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in evidence_files:
            archive.write(path, path.relative_to(args.artifact_root))
    (out / "F0_RESULTS_FOR_SUPERVISOR.sha256").write_text(
        sha256_file(zip_path) + "  " + zip_path.name + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS_EXECUTION", "recommendation": recommendation,
                      "report": str(report_path), "results_zip": str(zip_path)}, indent=2))


if __name__ == "__main__":
    main()
