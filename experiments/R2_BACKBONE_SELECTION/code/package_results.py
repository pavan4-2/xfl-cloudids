from __future__ import annotations
import argparse
import shutil
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--work',required=True); ap.add_argument('--system',choices=['SYSTEM5'],default='SYSTEM5'); args=ap.parse_args()
    work=Path(args.work).resolve(); results=work/'results'/'r2'
    if not (results/'R2_SYSTEM5_SUMMARY.json').exists() or not (results/'R2_SELECTION.json').exists():
        raise RuntimeError('STOP: completed R2 summary/selection not found; do not fabricate results.')
    sub=work/'submission'; sub.mkdir(parents=True,exist_ok=True)
    stage=sub/'_stage';
    if stage.exists(): shutil.rmtree(stage)
    stage.mkdir()
    shutil.copytree(results, stage/'results')
    root=Path(__file__).resolve().parent
    for name in ['R2_SPECIFICATION.json','REFERENCE_R1_SYSTEM5.json','CLASS_LABELS.json','REPRODUCIBILITY_README.md']:
        shutil.copy2(root/name, stage/name)
    zipbase=sub/'XFL_R2_SYSTEM5_RESULTS'
    zippath=Path(shutil.make_archive(str(zipbase),'zip',stage))
    shutil.rmtree(stage)
    print(f'PACKAGED_RESULTS {zippath}')

if __name__=='__main__': main()
