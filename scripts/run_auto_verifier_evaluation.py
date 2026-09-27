"""Wait for the existing train-only verifier job, then evaluate fixed pages."""
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
    parser.add_argument('--directory',type=Path,default=ROOT/'datasets/derived/cascade_auto_proposal_verifier_v1')
    args=parser.parse_args()
    directory=args.directory.resolve()
    record_path=ROOT/'results'/(directory.name+'_evaluation_job.json')
    if record_path.exists():
        raise ValueError('existing evaluation record; inspect before rerun')
    report={'status':'waiting_for_live_trainer','trainer_pid':args.wait_pid}
    def save():
        record_path.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        wait_for_process(args.wait_pid)
        if not (directory/'provenance.json').exists() or not (directory/'model.joblib').exists():
            raise ValueError('trainer exited without complete model/provenance')
        command=[sys.executable,'-u','scripts/evaluate_combined_improvements.py',
            '--pages','20','--pages-per-work','2',
            '--cascade','datasets/derived/cascade_auto_curated_localization_v1/model.json',
            '--reference','results/cascade_auto_curated_localization_v1_validation20.json',
            '--tight-verifier',str(directory/'model.joblib'),
            '--extra-thresholds','0','.25','.5','.75','1','1.25','2',
            '--output',str(ROOT/'results'/(directory.name+'_validation20.json')),
            '--prediction-dir',str(directory/'validation20')]
        report.update(status='running',command=command);save()
        subprocess.run(command,cwd=ROOT,check=True)
        report['status']='evaluation_finished_not_acceptance';save()
    except Exception as error:
        report.update(status='failed',error=str(error));save();raise


if __name__=='__main__':
    main()
