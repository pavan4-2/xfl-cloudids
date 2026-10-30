from __future__ import annotations

import argparse, json, zipfile
from pathlib import Path

FORBIDDEN_SUFFIXES={".npz",".pt",".pth",".ckpt"}
FORBIDDEN_NAMES=("sealed","untouched","final_test","final-test")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--work",required=True);ap.add_argument("--system",choices=["SYSTEM5","SYSTEM6"],required=True);a=ap.parse_args()
    work=Path(a.work).resolve(); results=work/"results"; submission=work/"submission"; submission.mkdir(parents=True,exist_ok=True)
    if not results.exists():raise FileNotFoundError("results directory missing")
    files=[]
    for p in sorted(results.rglob("*")):
        if not p.is_file():continue
        low=p.name.lower()
        if p.suffix.lower() in FORBIDDEN_SUFFIXES or any(x in low for x in FORBIDDEN_NAMES):
            raise RuntimeError(f"STOP: forbidden artifact in submission results: {p}")
        files.append(p)
    required=[results/"r0"/"R0_SUMMARY.json",results/"r1"/f"R1_{a.system}_SUMMARY.json"]
    for p in required:
        if not p.exists():raise FileNotFoundError(f"required output missing: {p}")
    z=submission/f"XFL_R0_R1_{a.system}_RESULTS.zip"
    with zipfile.ZipFile(z,"w",zipfile.ZIP_DEFLATED) as zf:
        for p in files:zf.write(p,"results/"+p.relative_to(results).as_posix())
        root=Path(__file__).resolve().parent
        for name in ["RECOVERY_SPECIFICATION.json","CLASS_LABELS.json","REPRODUCIBILITY_README.md"]:zf.write(root/name,name)
    print(json.dumps({"status":"TECHNICAL_PASS","system":a.system,"zip":str(z),"files":len(files)},sort_keys=True))

if __name__=="__main__":main()
