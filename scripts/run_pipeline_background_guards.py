"""Await fresh hard backgrounds, then run whole-page and local teacher vetoes."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from run_auto_coverage_iteration import wait_for_process

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wait-pid',type=int,required=True)
    args=p.parse_args()
    record=ROOT/'results/cascade_auto_pipeline_background_guards.json'
    if record.exists():
        raise ValueError('immutable run record exists')
    pool=ROOT/'datasets/derived/cascade_auto_pipeline_backgrounds_v1'
    whole=ROOT/'datasets/derived/cascade_auto_pipeline_backgrounds_v1_guard'
    local=ROOT/'datasets/derived/cascade_auto_pipeline_backgrounds_v1_context_guard'
    report={'status':'waiting_for_mining','miner_pid':args.wait_pid,'stages':[]}
    def save():
        record.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        wait_for_process(args.wait_pid)
        assert (pool/'provenance.json').exists(),'miner did not complete'
        commands=[
            ['scripts/filter_auto_background_faces.py','--dataset',str(pool),'--output',str(whole),'--threshold','.1'],
            ['scripts/filter_auto_background_contexts.py','--pool',str(pool),'--output',str(local),'--all-contexts','--preceding-guard',str(whole),'--quarantine-json',str(pool/'selection.json')],
        ]
        for i,command in enumerate(commands):
            entry={'command':[sys.executable,'-u',*command],'status':'running','stdout':f'tmp/cascade_auto_pipeline_background_guard_stage{i}.stdout.log','stderr':f'tmp/cascade_auto_pipeline_background_guard_stage{i}.stderr.log'}
            report['status']='running';report['stages'].append(entry);save()
            with (ROOT/entry['stdout']).open('w') as stdout,(ROOT/entry['stderr']).open('w') as stderr:
                result=subprocess.run(entry['command'],cwd=ROOT,stdout=stdout,stderr=stderr)
            entry.update(exit_code=result.returncode,status='completed' if result.returncode==0 else 'failed');save()
            if result.returncode:
                raise RuntimeError(f'guard stage {i} failed')
        report['status']='data_filtering_finished_not_acceptance';save()
    except Exception as error:
        report.update(status='failed',error=str(error));save();raise


if __name__=='__main__':
    main()
