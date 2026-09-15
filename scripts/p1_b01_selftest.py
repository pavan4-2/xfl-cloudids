from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from xfl_cloudids.baselines import aggregate_fedavg, baseline_names, fedfomo_positive_improvement_weights
from xfl_cloudids.evaluation import classification_metrics, write_result_csv, write_run_manifest


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--artifact-dir',required=True); args=ap.parse_args()
    out=Path(args.artifact_dir); out.mkdir(parents=True,exist_ok=True)
    states=[{"w":np.array([0.,1.])},{"w":np.array([2.,3.])}]
    avg=aggregate_fedavg(states,[1,3])
    metrics=classification_metrics([0,0,1,1],[0,1,1,1],[0,1],0)
    peer=fedfomo_positive_improvement_weights([1,2],[-0.2,0.3])
    rec={"method":"fedavg","scenario":"selftest","seed":0,"round":0,"receiver":0,"split":"selftest",**metrics}
    write_result_csv([rec],out/'selftest_results.csv')
    write_run_manifest({"step":"P1-B01","baselines":list(baseline_names()),"sealed_final_test_access":False},out/'selftest_manifest.json')
    result={
        "technical_status":"PASS",
        "baseline_count":len(baseline_names()),
        "required_baselines_present":all(x in baseline_names() for x in ["local_only","fedavg","fedprox","scaffold","ditto","fedfomo"]),
        "fedavg_expected":bool(np.allclose(avg['w'],[1.5,2.5])),
        "fedfomo_rejects_harmful_peer":peer[1] == 0.0 and peer[2] == 1.0,
        "common_metrics_keys":sorted(metrics),
        "sealed_final_test_access":False,
    }
    (out/'selftest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not all([result['required_baselines_present'],result['fedavg_expected'],result['fedfomo_rejects_harmful_peer']]):
        raise SystemExit(2)

if __name__=='__main__': main()
