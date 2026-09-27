"""Continue the verified live context guard into new Cascade/HOG comparisons."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from run_auto_coverage_iteration import wait_for_process

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid',type=int,required=True)
    args=parser.parse_args()
    record=ROOT/'results/cascade_auto_context_iteration.json'
    if record.exists():
        raise ValueError('existing record; inspect before continuing')
    report={'status':'waiting_for_context_guard','guard_pid':args.wait_pid,'stages':[]}
    def save():
        record.write_text(json.dumps(report,indent=2)+'\n')
    save()
    output=ROOT/'datasets/derived/cascade_auto_context_localization_v1'
    guard=ROOT/'datasets/derived/cascade_auto_background_context_guard_v1'
    verifier=ROOT/'datasets/derived/cascade_auto_proposal_verifier_v3/model.joblib'
    try:
        wait_for_process(args.wait_pid)
        if not (guard/'summary.json').exists():
            raise ValueError('guard exited without final summary')
        commands=[
            ['scripts/train_context_cleaned_verifier.py'],
            ['scripts/evaluate_combined_improvements.py','--pages','20','--pages-per-work','2','--cascade','datasets/derived/cascade_auto_curated_localization_v1/model.json','--reference','results/cascade_auto_curated_localization_v1_validation20.json','--tight-verifier',str(verifier),'--extra-thresholds','0','.25','.5','.75','1','1.25','2','--output','results/cascade_auto_proposal_verifier_v3_validation20.json','--prediction-dir','datasets/derived/cascade_auto_proposal_verifier_v3/validation20'],
            ['scripts/train_auto_mined_cascade.py','--mined','datasets/derived/cascade_auto_background_merged_v1','--mined-guard',str(guard),'--negative-quarantine',str(guard/'provenance.json'),'--localization-negatives','datasets/derived/cascade_localization_negatives_v1','--base-negative-count','0','--positive-hog-min','0','--output',str(output),'--stages','3','--trees','64','--root-candidates','64','--child-candidates','32','--stage-recall','.98','--histogram-splits'],
            ['scripts/evaluate_candidate_verifier.py','--pages','20','--pages-per-work','2','--cascade',str(output/'model.json'),'--verifier','models/candidate_hog_rbf_step3_v1.joblib','--output','results/cascade_auto_context_localization_v1_validation20.json','--cache-dir',str(output/'validation_cache')],
            ['scripts/evaluate_combined_improvements.py','--pages','20','--pages-per-work','2','--cascade',str(output/'model.json'),'--reference','results/cascade_auto_context_localization_v1_validation20.json','--tight-verifier',str(verifier),'--extra-thresholds','0','.25','.5','.75','1','1.25','2','--output','results/cascade_auto_context_localization_v1_combined20.json','--prediction-dir',str(output/'combined20')],
        ]
        for i,command in enumerate(commands):
            entry={'command':[sys.executable,'-u',*command],'status':'running','stdout':f'tmp/cascade_auto_context_stage{i}.stdout.log','stderr':f'tmp/cascade_auto_context_stage{i}.stderr.log'}
            report['stages'].append(entry);report['status']='running';save()
            with (ROOT/entry['stdout']).open('w') as stdout,(ROOT/entry['stderr']).open('w') as stderr:
                completed=subprocess.run(entry['command'],cwd=ROOT,stdout=stdout,stderr=stderr)
            entry.update(exit_code=completed.returncode,status='completed' if completed.returncode==0 else 'failed');save()
            if completed.returncode:
                raise RuntimeError(f'stage {i} failed; inspect logs')
        report['status']='experiments_finished_not_acceptance';save()
    except Exception as error:
        report.update(status='failed',error=str(error));save();raise


if __name__=='__main__':
    main()
