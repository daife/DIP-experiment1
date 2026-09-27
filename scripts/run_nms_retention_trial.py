"""Compare broader first-NMS retention using the best expanded-data model."""
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    record=ROOT/'results/cascade_auto_scan_positive_nms05_iteration.json'
    if record.exists():
        raise ValueError('existing immutable trial')
    directory=ROOT/'datasets/derived/cascade_auto_scan_positive_localization_v1'
    cascade=directory/'model.json'
    cache=directory/'nms05_cache'
    reference='results/cascade_auto_scan_positive_nms05_validation20.json'
    commands=[
        ['scripts/evaluate_candidate_verifier.py','--pages','20','--pages-per-work','2','--cascade',str(cascade),'--verifier','models/candidate_hog_rbf_step3_v1.joblib','--nms-iou','.5','--output',reference,'--cache-dir',str(cache)],
        ['scripts/evaluate_combined_improvements.py','--pages','20','--pages-per-work','2','--cascade',str(cascade),'--cache-dir',str(cache),'--first-nms','.5','--reference',reference,'--tight-verifier','datasets/derived/cascade_auto_proposal_verifier_v4/model.joblib','--extra-thresholds','0','.25','.5','.75','1','1.25','2','--output','results/cascade_auto_scan_positive_nms05_v4_combined20.json','--prediction-dir',str(directory/'nms05_combined20_v4')],
    ]
    report={'status':'running','scope':'inference retention experiment; model/data/GT unchanged; not acceptance','stages':[]}
    def save():
        record.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        for i,command in enumerate(commands):
            entry={'command':[sys.executable,'-u',*command],'status':'running','stdout':f'tmp/cascade_auto_scan_positive_nms05_stage{i}.stdout.log','stderr':f'tmp/cascade_auto_scan_positive_nms05_stage{i}.stderr.log'}
            report['stages'].append(entry);save()
            with (ROOT/entry['stdout']).open('w') as stdout,(ROOT/entry['stderr']).open('w') as stderr:
                result=subprocess.run(entry['command'],cwd=ROOT,stdout=stdout,stderr=stderr)
            entry.update(exit_code=result.returncode,status='completed' if result.returncode==0 else 'failed');save()
            if result.returncode:
                raise RuntimeError(f'stage {i} failed')
        report['status']='experiments_finished_not_acceptance';save()
    except Exception as error:
        report.update(status='failed',error=str(error));save();raise


if __name__=='__main__':
    main()
