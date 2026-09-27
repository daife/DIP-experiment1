"""Compare completed full-page runs on identical validation pages and gates."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    baseline = ROOT/'results/step8_cascade_search128_validation.json'
    paths = [baseline, *[path for i in range(1,4) if (path:=ROOT/f'results/cascade_auto_v1_round{i}_validation.json').exists()]]
    mined = ROOT/'datasets/derived/cascade_auto_mined_v1/validation.json'
    if mined.exists():
        paths.append(mined)
    capacity = ROOT/'datasets/derived/cascade_auto_mined_128trees/validation.json'
    if capacity.exists():
        paths.append(capacity)
    recalibrated = [ROOT/f'results/cascade_auto_recall{target}_validation.json' for target in ('098','095')]
    paths.extend(path for path in recalibrated if path.exists())
    curated = ROOT/'datasets/derived/cascade_auto_curated_localization_v1/validation.json'
    if curated.exists():
        paths.append(curated)
    balanced = ROOT/'datasets/derived/cascade_auto_curated_balanced_v1/validation.json'
    if balanced.exists():
        paths.append(balanced)
    reference = json.loads(baseline.read_text(encoding='utf-8'))
    records, sources = [], []
    for path in paths:
        run = json.loads(path.read_text(encoding='utf-8'))
        fields = ('split','step','scale_factor','nms_iou','post_filter_nms','post_filter_score','verifier_sha256','manifest_sha256')
        assert all(run[key] == reference[key] for key in fields), f'incompatible run: {path}'
        assert [(p['id'],p['gt']) for p in run['pages']] == [(p['id'],p['gt']) for p in reference['pages']]
        model = 'historical_search128' if path == baseline else ('fresh_mined_geometry_capacity' if path == mined else ('fresh_mined_128trees' if path == capacity else ('curated_localization' if path == curated else ('curated_balanced' if path == balanced else path.stem.replace('_validation','')))))
        for gate, result in run['summary'].items():
            assert gate in reference['summary']
            tp,fp,fn = (result[k] for k in ('tp','fp','fn'))
            assert sum(p['rules'][gate]['tp'] for p in run['pages']) == tp
            assert sum(p['rules'][gate]['fp'] for p in run['pages']) == fp
            assert sum(p['rules'][gate]['fn'] for p in run['pages']) == fn
            assert tp+fn == 73
            records.append({'model':model,'gate':gate,**{k:result[k] for k in ('tp','fp','fn','precision','recall','f1')}})
        sources.append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'cascade_sha256':run['cascade_sha256']})
    output = ROOT/'results/cascade_auto_comparison.csv'
    with output.open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    fig,axes = plt.subplots(1,2,figsize=(11,4.5))
    for ax,gate in zip(axes,('rbf1.5_side36','rbf1.75_side36')):
        for model in dict.fromkeys(row['model'] for row in records):
            row = next(r for r in records if r['model']==model and r['gate']==gate)
            ax.scatter(row['recall'],row['precision'],label=f"{model}: F1={row['f1']:.3f}",s=65)
        ax.set(xlabel='Recall (IoU >= 0.5)',ylabel='Precision (IoU >= 0.5)',title=gate,xlim=(0,1),ylim=(0,1))
        ax.grid(alpha=.2)
        ax.legend(fontsize=7,loc='upper right')
    fig.suptitle('Same 8 explored validation works, 73 faces; completed runs only')
    fig.tight_layout()
    fig.savefig(ROOT/'results/cascade_auto_comparison.png',dpi=180)
    plt.close(fig)
    pending = [f'round{i}' for i in range(1,4) if not (ROOT/f'results/cascade_auto_v1_round{i}_validation.json').exists()]
    if not mined.exists():
        pending.append('fresh_mined_geometry_capacity')
    if not capacity.exists():
        pending.append('fresh_mined_128trees')
    pending.extend(path.stem for path in recalibrated if not path.exists())
    if not curated.exists():
        pending.append('curated_localization')
    if not balanced.exists():
        pending.append('curated_balanced')
    report = {'sources':sources,'pending':pending,'records':records,'comparison_scope':'identical explored validation works and HOG gates; no box regression; not blind test',
              'plot':'results/cascade_auto_comparison.png','plot_legend':'color/legend identifies models; x=recall, y=precision; two panels fix HOG gates 1.5/1.75',
              'command':'.venv/Scripts/python.exe scripts/summarize_auto_cascade.py'}
    (ROOT/'results/cascade_auto_comparison.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'completed':len(paths)-1,'pending':pending,'same_page_identity_verified':True}))


if __name__=='__main__':
    main()
