"""Compare final pipelines on the fixed twenty-page validation expansion."""
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def metrics(counts):
    tp,fp,fn = (counts[k] for k in ('tp','fp','fn'))
    return {**counts,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),
            'f1':2*tp/max(1,2*tp+fp+fn)}


def main():
    reports = {name:ROOT/f'results/cascade_auto_{name}_validation20_combined.json'
               for name in ('historical','curated_localization_v1')}
    loaded = {name:json.loads(path.read_text()) for name,path in reports.items()}
    baseline = loaded['historical']
    page_ids = [r['image_id'] for r in baseline['pages']]
    assert len(page_ids)==len(set(page_ids))==20
    lookup = {r['image_id']:r for r in map(json.loads,(ROOT/'datasets/manifests/normalized/manga109_faces.jsonl').read_text(encoding='utf-8').splitlines())}
    assert sum(len(lookup[i]['annotations']) for i in page_ids)==252
    records = []
    for name,report in loaded.items():
        assert [r['image_id'] for r in report['pages']]==page_ids
        for field in ('split','step','scale_factor','first_nms','final_nms','thresholds'):
            assert report[field]==baseline[field]
        assert report['split']=='validation'
        for key,path_hash in baseline['sha256'].items():
            if Path(key).name != 'model.json':
                assert report['sha256'][key]==path_hash
        for gate in ('tight_postnms_1.5','tight_postnms_1.75','tight_refined_1.5','tight_refined_1.75'):
            grouped=defaultdict(lambda:dict(tp=0,fp=0,fn=0))
            for page in report['pages']:
                result=page['results'][gate]
                assert result['tp']+result['fn']==len(lookup[page['image_id']]['annotations'])
                work=lookup[page['image_id']]['source_group']
                for k in ('tp','fp','fn'):
                    grouped[work][k]+=result[k]
            totals={k:sum(r[k] for r in grouped.values()) for k in ('tp','fp','fn')}
            assert metrics(totals)==report['summary'][gate]
            records.append({'model':name,'gate':gate,'work':'ALL',**metrics(totals)})
            records.extend({'model':name,'gate':gate,'work':work,**metrics(counts)} for work,counts in sorted(grouped.items()))
    output=ROOT/'results/cascade_auto_validation20_comparison'
    with output.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    output.with_suffix('.json').write_text(json.dumps({'records':records,'page_ids':page_ids,
        'sources':{name:{'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in reports.items()},
        'scope':'20 fixed validation pages, two per ten works, 252 faces, IoU>=.5; explored validation, not blind test'},indent=2)+'\n')
    works=sorted({r['work'] for r in records if r['work']!='ALL'})
    fig,axes=plt.subplots(2,2,figsize=(18,11),sharex=True)
    gates=('tight_postnms_1.5','tight_postnms_1.75','tight_refined_1.5','tight_refined_1.75')
    for ax,gate in zip(axes.flat,gates):
        for offset,(name,color) in zip((-.18,.18),(('historical','#2874a6'),('curated_localization_v1','#c66b26'))):
            values=[next(r['f1'] for r in records if r['model']==name and r['gate']==gate and r['work']==work) for work in works]
            total=next(r['f1'] for r in records if r['model']==name and r['gate']==gate and r['work']=='ALL')
            ax.barh([i+offset for i in range(len(works))],values,height=.34,color=color,label=f'{name}: overall F1={total:.3f}')
        ax.set_yticks(range(len(works)),works)
        ax.set_xlim(0,1);ax.set_xlabel('F1 at IoU >= 0.5')
        ax.set_title(gate);ax.invert_yaxis();ax.legend(fontsize=9,loc='lower right')
    fig.suptitle('Same 20 explored validation pages; 2 pages per work; 252 faces',fontsize=18)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(output.with_suffix('.png'),dpi=140);plt.close(fig)
    print(json.dumps([r for r in records if r['work']=='ALL']))


if __name__ == '__main__':
    main()
