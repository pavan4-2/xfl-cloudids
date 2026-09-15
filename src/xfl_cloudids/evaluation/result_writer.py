from __future__ import annotations
import csv, json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REQUIRED_RESULT_FIELDS = (
    "method", "scenario", "seed", "round", "receiver", "split",
    "accuracy", "macro_f1", "far", "worst_class_recall",
)


def validate_result_record(record: Mapping) -> None:
    missing = [k for k in REQUIRED_RESULT_FIELDS if k not in record]
    if missing:
        raise ValueError(f"missing result fields: {missing}")


def write_result_csv(records: Sequence[Mapping], path) -> Path:
    if not records:
        raise ValueError("at least one result record is required")
    for r in records:
        validate_result_record(r)
    keys = list(dict.fromkeys([*REQUIRED_RESULT_FIELDS, *[k for r in records for k in r.keys()]]))
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(records)
    return p


def write_run_manifest(payload: Mapping, path) -> Path:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    return p
