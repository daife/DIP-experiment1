"""Isolate the original scale-mined background distribution after context vetoes."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT/'datasets/derived/cascade_auto_background_merged_v1'
    parent = ROOT/'datasets/derived/cascade_auto_background_context_guard_v1'
    guard = ROOT/'datasets/derived/cascade_auto_background_scale_context_guard_v1'
    output = ROOT/'datasets/derived/cascade_auto_scale_context_localization_v1'
    record = ROOT/'results/cascade_auto_scale_context_iteration.json'
    if guard.exists() or output.exists() or record.exists():
        raise ValueError('immutable experiment already exists; inspect its status')
    rows = [json.loads(line) for line in (source/'candidates.jsonl').read_text().splitlines()]
    metadata = json.loads((parent/'provenance.json').read_text())
    summary = json.loads((parent/'summary.json').read_text())
    assert metadata['source_sha256'] == sha(source/'candidates.jsonl')
    assert summary['accepted_indices_sha256'] == sha(parent/'accepted_indices.npy')
    accepted = np.load(parent/'accepted_indices.npy')
    assert len(set(accepted.tolist())) == len(accepted)
    assert np.all((accepted >= 0) & (accepted < len(rows)))
    indices = np.array([int(i) for i in accepted if rows[i]['source_dataset'] == 'cascade_auto_windows_scale_v2'], dtype=np.int64)
    assert len(indices) and all(rows[i]['split'] == 'train' and rows[i]['label'] == 0 for i in indices)
    guard.mkdir(parents=True)
    np.save(guard/'accepted_indices.npy', indices)
    provenance = {**metadata, 'parent_guard_sha256': sha(parent/'provenance.json'),
                  'parent_accepted_indices_sha256': sha(parent/'accepted_indices.npy'),
                  'selection': 'context-accepted intersect original scale-mined source; no new teacher inference',
                  'script_sha256': sha(Path(__file__)), 'human_reviewed': False}
    (guard/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    (guard/'summary.json').write_text(json.dumps({'accepted':len(indices), 'accepted_indices_sha256':sha(guard/'accepted_indices.npy')}, indent=2)+'\n')
    verifier = ROOT/'datasets/derived/cascade_auto_proposal_verifier_v3/model.joblib'
    ordinary = 'results/cascade_auto_scale_context_localization_v1_validation20.json'
    commands = [
        ['scripts/train_auto_mined_cascade.py', '--mined', str(source), '--mined-guard', str(guard), '--negative-quarantine', str(parent/'provenance.json'), '--localization-negatives', 'datasets/derived/cascade_localization_negatives_v1', '--base-negative-count', '0', '--positive-hog-min', '0', '--output', str(output), '--stages', '3', '--trees', '64', '--root-candidates', '64', '--child-candidates', '32', '--stage-recall', '.98', '--histogram-splits'],
        ['scripts/evaluate_candidate_verifier.py', '--pages', '20', '--pages-per-work', '2', '--cascade', str(output/'model.json'), '--verifier', 'models/candidate_hog_rbf_step3_v1.joblib', '--output', ordinary, '--cache-dir', str(output/'validation_cache')],
        ['scripts/evaluate_combined_improvements.py', '--pages', '20', '--pages-per-work', '2', '--cascade', str(output/'model.json'), '--reference', ordinary, '--tight-verifier', str(verifier), '--extra-thresholds', '0', '.25', '.5', '.75', '1', '1.25', '2', '--output', 'results/cascade_auto_scale_context_localization_v1_combined20.json', '--prediction-dir', str(output/'combined20')],
    ]
    report = {'status':'running', 'accepted_backgrounds':len(indices), 'guard_provenance_sha256':sha(guard/'provenance.json'), 'stages':[]}
    def save():
        record.write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        for i, command in enumerate(commands):
            entry = {'command':[sys.executable, '-u', *command], 'status':'running', 'stdout':f'tmp/cascade_auto_scale_context_stage{i}.stdout.log', 'stderr':f'tmp/cascade_auto_scale_context_stage{i}.stderr.log'}
            report['stages'].append(entry); save()
            with (ROOT/entry['stdout']).open('w') as stdout, (ROOT/entry['stderr']).open('w') as stderr:
                result = subprocess.run(entry['command'], cwd=ROOT, stdout=stdout, stderr=stderr)
            entry.update(exit_code=result.returncode, status='completed' if result.returncode == 0 else 'failed'); save()
            if result.returncode:
                raise RuntimeError(f'stage {i} failed; inspect logs')
        report['status'] = 'experiments_finished_not_acceptance'; save()
    except Exception as error:
        report.update(status='failed', error=str(error)); save(); raise


if __name__ == '__main__':
    main()
