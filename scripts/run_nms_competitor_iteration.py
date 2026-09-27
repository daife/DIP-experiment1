"""Continue a live NMS competitor miner into a fixed-data Cascade comparison."""
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
    record=ROOT/'results/cascade_auto_nms_competitor_iteration.json'
    if record.exists():
        raise ValueError('existing run record; inspect without overwriting')
    report={'status':'waiting_for_competitor_mining','miner_pid':args.wait_pid,'stages':[]}
    def save():
        record.write_text(json.dumps(report,indent=2)+'\n')
    save()
    output=ROOT/'datasets/derived/cascade_auto_nms_competitor_localization_v1'
    pool=ROOT/'datasets/derived/cascade_auto_nms_competitors_v1'
    ordinary='results/cascade_auto_nms_competitor_localization_v1_validation20.json'
    try:
        wait_for_process(args.wait_pid)
        assert (pool/'provenance.json').exists(), 'mining has not completed'
        commands=[
            ['scripts/train_auto_mined_cascade.py','--mined','datasets/derived/cascade_auto_background_merged_v1','--mined-guard','datasets/derived/cascade_auto_background_scale_context_guard_v1','--negative-quarantine','datasets/derived/cascade_auto_background_context_guard_v1/provenance.json','--localization-negatives','datasets/derived/cascade_localization_negatives_v1','--proposal-positives','datasets/derived/cascade_auto_scan_positives_v1','--nms-competitors',str(pool),'--base-negative-count','0','--positive-hog-min','0','--output',str(output),'--stages','3','--trees','64','--root-candidates','64','--child-candidates','32','--stage-recall','.98','--histogram-splits'],
            ['scripts/evaluate_candidate_verifier.py','--pages','20','--pages-per-work','2','--cascade',str(output/'model.json'),'--verifier','models/candidate_hog_rbf_step3_v1.joblib','--output',ordinary,'--cache-dir',str(output/'validation_cache')],
            ['scripts/evaluate_combined_improvements.py','--pages','20','--pages-per-work','2','--cascade',str(output/'model.json'),'--reference',ordinary,'--tight-verifier','datasets/derived/cascade_auto_proposal_verifier_v4/model.joblib','--extra-thresholds','0','.25','.5','.75','1','1.25','2','--output','results/cascade_auto_nms_competitor_localization_v1_v4_combined20.json','--prediction-dir',str(output/'combined20_v4')],
        ]
        for i,command in enumerate(commands):
            entry={'command':[sys.executable,'-u',*command],'status':'running','stdout':f'tmp/cascade_auto_nms_competitor_stage{i}.stdout.log','stderr':f'tmp/cascade_auto_nms_competitor_stage{i}.stderr.log'}
            report['status']='running';report['stages'].append(entry);save()
            with (ROOT/entry['stdout']).open('w') as stdout,(ROOT/entry['stderr']).open('w') as stderr:
                result=subprocess.run(entry['command'],cwd=ROOT,stdout=stdout,stderr=stderr)
            entry.update(exit_code=result.returncode,status='completed' if result.returncode==0 else 'failed');save()
            if result.returncode:
                raise RuntimeError(f'stage {i} failed; inspect logs')
        report['status']='experiments_finished_not_acceptance';save()
    except Exception as error:
        report.update(status='failed',error=str(error));save();raise


if __name__=='__main__':
    main()
