"""Await collection, audit, merge and refit the second positive expansion."""
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
    record=ROOT/'results/cascade_auto_scan_positive_v2_iteration.json'
    assert not record.exists()
    report={'status':'waiting_for_collection','collector_pid':args.wait_pid,'stages':[]}
    def save():
        record.write_text(json.dumps(report,indent=2)+'\n')
    save()
    commands=[
        ['scripts/audit_scan_positive_shards.py','--pool','datasets/derived/cascade_auto_scan_positives_v2','--require-complete','--output','results/cascade_auto_scan_positives_v2_final_shard_audit.json'],
        ['scripts/merge_scan_positive_pools.py'],
        ['scripts/run_scan_positive_iteration.py','--fit-verifier','--pool','datasets/derived/cascade_auto_scan_positives_merged_v2','--parent-verifier','datasets/derived/cascade_auto_proposal_verifier_v5','--verifier-output','datasets/derived/cascade_auto_proposal_verifier_v6'],
        ['scripts/run_cascade_data_trial.py','--tag','cascade_auto_scan_positive_localization_v2','--trees','64','--root-candidates','64','--child-candidates','32','--mined','datasets/derived/cascade_auto_pipeline_backgrounds_merged_v2','--mined-guard','datasets/derived/cascade_auto_pipeline_backgrounds_merged_v2_guard','--negative-quarantine','datasets/derived/cascade_auto_pipeline_backgrounds_merged_v2/provenance.json','--proposal-positives','datasets/derived/cascade_auto_scan_positives_merged_v2','--verifier','datasets/derived/cascade_auto_proposal_verifier_v6/model.joblib','--verifier-name','v6'],
    ]
    try:
        wait_for_process(args.wait_pid)
        assert (ROOT/'datasets/derived/cascade_auto_scan_positives_v2/provenance.json').exists()
        for i,command in enumerate(commands):
            entry={'command':[sys.executable,'-u',*command],'status':'running',
                   'stdout':f'tmp/cascade_auto_scan_positive_v2_stage{i}.stdout.log',
                   'stderr':f'tmp/cascade_auto_scan_positive_v2_stage{i}.stderr.log'}
            report['status']='running';report['stages'].append(entry);save()
            with (ROOT/entry['stdout']).open('w') as stdout,(ROOT/entry['stderr']).open('w') as stderr:
                result=subprocess.run(entry['command'],cwd=ROOT,stdout=stdout,stderr=stderr)
            entry.update(exit_code=result.returncode,status='completed' if result.returncode==0 else 'failed');save()
            if result.returncode:
                raise RuntimeError(f'stage {i} failed; inspect saved logs')
        report['status']='experiments_finished_not_acceptance';save()
    except Exception as error:
        report.update(status='failed',error=str(error));save();raise


if __name__=='__main__':
    main()
