"""Train an immutable capacity trial on the audited expanded data, then evaluate."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag',required=True)
    p.add_argument('--trees',type=int,default=128)
    p.add_argument('--root-candidates',type=int,default=128)
    p.add_argument('--child-candidates',type=int,default=48)
    p.add_argument('--mined',default='datasets/derived/cascade_auto_background_merged_v1')
    p.add_argument('--mined-guard',default='datasets/derived/cascade_auto_background_scale_context_guard_v1')
    p.add_argument('--negative-quarantine',default='datasets/derived/cascade_auto_background_context_guard_v1/provenance.json')
    p.add_argument('--verifier',default='datasets/derived/cascade_auto_proposal_verifier_v4/model.joblib')
    p.add_argument('--verifier-name',default='v4')
    p.add_argument('--proposal-positives',default='datasets/derived/cascade_auto_scan_positives_v1')
    args=p.parse_args()
    if not args.tag.replace('_','').isalnum() or not args.verifier_name.replace('_','').isalnum() or min(args.trees,args.root_candidates,args.child_candidates)<1:
        p.error('invalid tag or capacity')
    output=ROOT/'datasets/derived'/args.tag
    record=ROOT/'results'/f'{args.tag}_iteration.json'
    if output.exists() or record.exists():
        raise ValueError('immutable trial already exists')
    ordinary=f'results/{args.tag}_validation20.json'
    commands=[
        ['scripts/train_auto_mined_cascade.py','--mined',args.mined,'--mined-guard',args.mined_guard,'--negative-quarantine',args.negative_quarantine,'--localization-negatives','datasets/derived/cascade_localization_negatives_v1','--proposal-positives',args.proposal_positives,'--base-negative-count','0','--positive-hog-min','0','--output',str(output),'--stages','3','--trees',str(args.trees),'--root-candidates',str(args.root_candidates),'--child-candidates',str(args.child_candidates),'--stage-recall','.98','--histogram-splits'],
        ['scripts/evaluate_candidate_verifier.py','--pages','20','--pages-per-work','2','--cascade',str(output/'model.json'),'--verifier','models/candidate_hog_rbf_step3_v1.joblib','--output',ordinary,'--cache-dir',str(output/'validation_cache')],
        ['scripts/evaluate_combined_improvements.py','--pages','20','--pages-per-work','2','--cascade',str(output/'model.json'),'--reference',ordinary,'--tight-verifier',args.verifier,'--extra-thresholds','0','.25','.5','.75','1','1.25','2','--output',f'results/{args.tag}_{args.verifier_name}_combined20.json','--prediction-dir',str(output/f'combined20_{args.verifier_name}')],
    ]
    report={'status':'running','scope':'expanded-data Cascade trial; not acceptance','stages':[]}
    def save():
        record.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        for i,command in enumerate(commands):
            entry={'command':[sys.executable,'-u',*command],'status':'running','stdout':f'tmp/{args.tag}_stage{i}.stdout.log','stderr':f'tmp/{args.tag}_stage{i}.stderr.log'}
            report['stages'].append(entry);save()
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
