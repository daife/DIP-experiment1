"""Continue a live teacher-cleaning job into an auditable Cascade iteration."""
import argparse
import ctypes
from ctypes import wintypes
import datetime
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wait_for_process(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE,wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x100000 | 0x1000,False,pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), 'required cleaning process is unavailable')
    try:
        while True:
            status = kernel.WaitForSingleObject(handle,30000)
            if status == 0:
                code = wintypes.DWORD()
                if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):
                    raise OSError(ctypes.get_last_error(),'cannot read cleaning exit code')
                if code.value != 0:
                    raise RuntimeError(f'cleaning process exited with {code.value}; no training started')
                return
            if status != 258:
                raise OSError(ctypes.get_last_error(),'process wait failed')
            print('waiting on confirmed teacher-cleaning process',pid,flush=True)
    finally:
        kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid',type=int,required=True)
    args = parser.parse_args()
    output = ROOT/'datasets/derived/cascade_auto_coverage_localization_v1'
    if output.exists():
        raise ValueError('iteration output exists; do not rerun or overwrite')
    report_path = ROOT/'results/cascade_auto_coverage_iteration.json'
    if report_path.exists():
        raise ValueError('iteration record exists; inspect state before restarting')
    report = {'status':'waiting_for_teacher_cleaning','cleaning_pid':args.wait_pid,
              'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope':'automatic data expansion experiment; performance acceptance remains unproven',
              'stages':[]}
    def save():
        report_path.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        wait_for_process(args.wait_pid)
        guard = ROOT/'datasets/derived/cascade_auto_background_coverage_v1_guard'
        if not (guard/'summary.json').exists():
            raise ValueError('cleaning exited without final summary')
        commands = [
            ['scripts/merge_guarded_backgrounds.py'],
            ['scripts/verify_auto_window_data.py','--dataset','datasets/derived/cascade_auto_background_merged_v1','--report','results/cascade_auto_background_merged_v1_verification.json'],
            ['scripts/train_auto_mined_cascade.py','--mined','datasets/derived/cascade_auto_background_merged_v1','--localization-negatives','datasets/derived/cascade_localization_negatives_v1','--base-negative-count','0','--positive-hog-min','0','--output',str(output),'--stages','3','--trees','64','--root-candidates','64','--child-candidates','32','--stage-recall','0.98','--histogram-splits'],
            ['scripts/evaluate_candidate_verifier.py','--pages','20','--pages-per-work','2','--cascade',str(output/'model.json'),'--verifier','models/candidate_hog_rbf_step3_v1.joblib','--output','results/cascade_auto_coverage_localization_v1_validation20.json','--cache-dir',str(output/'validation_cache')],
            ['scripts/evaluate_combined_improvements.py','--pages','20','--pages-per-work','2','--cascade',str(output/'model.json'),'--reference','results/cascade_auto_coverage_localization_v1_validation20.json','--output','results/cascade_auto_coverage_localization_v1_validation20_combined.json','--prediction-dir',str(output/'combined20')],
        ]
        for index,command in enumerate(commands):
            entry = {'command':[sys.executable,'-u',*command],'status':'running',
                     'stdout':f'tmp/cascade_auto_coverage_iteration_stage{index}.stdout.log',
                     'stderr':f'tmp/cascade_auto_coverage_iteration_stage{index}.stderr.log'}
            report['stages'].append(entry);report['status']='running';save()
            print('starting',command,flush=True)
            with (ROOT/entry['stdout']).open('w') as stdout,(ROOT/entry['stderr']).open('w') as stderr:
                result = subprocess.run(entry['command'],cwd=ROOT,stdout=stdout,stderr=stderr)
            entry['exit_code']=result.returncode
            entry['status']='completed' if result.returncode==0 else 'failed';save()
            if result.returncode:
                raise RuntimeError(f'stage {index} failed; see recorded logs')
        report['status']='experiments_finished_not_acceptance';save()
    except Exception as error:
        report['status']='failed';report['error']=str(error);save()
        raise


if __name__=='__main__':
    main()
