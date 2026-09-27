"""Replay both background veto decisions from saved teacher predictions."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from prepare_detection_dataset import near_face

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report_path=ROOT/'results/cascade_auto_pipeline_background_guard_audit.json'
    assert not report_path.exists()
    state=json.loads((ROOT/'results/cascade_auto_pipeline_background_guards.json').read_text())
    assert state['status']=='data_filtering_finished_not_acceptance'
    assert len(state['stages'])==2 and all(s['exit_code']==0 for s in state['stages'])
    pool=ROOT/'datasets/derived/cascade_auto_pipeline_backgrounds_v1'
    rows=list(map(json.loads,(pool/'candidates.jsonl').read_text().splitlines()))
    whole=ROOT/'datasets/derived/cascade_auto_pipeline_backgrounds_v1_guard'
    local=ROOT/'datasets/derived/cascade_auto_pipeline_backgrounds_v1_context_guard'
    outputs=[]
    for directory in [whole,local]:
        meta=json.loads((directory/'provenance.json').read_text())
        summary=json.loads((directory/'summary.json').read_text())
        assert meta['source_sha256']==sha(pool/'candidates.jsonl')
        assert summary['accepted_indices_sha256']==sha(directory/'accepted_indices.npy')
        assert summary['decisions_sha256']==sha(directory/'decisions.jsonl')
        decisions=list(map(json.loads,(directory/'decisions.jsonl').read_text().splitlines()))
        indices=np.load(directory/'accepted_indices.npy')
        assert len(decisions)==summary['total']==len(rows)
        assert summary['accepted']==len(indices) and summary['rejected']==len(rows)-len(indices)
        assert indices.dtype==np.int64 and len(indices)==len(set(indices.tolist()))
        assert set(indices.tolist())=={d['array_index'] for d in decisions if d['decision']=='keep_auto'}
        assert all(d['array_index']==i and d['id']==r['id'] and d['page_id']==r['page_id'] for i,(d,r) in enumerate(zip(decisions,rows,strict=True)))
        outputs.append((meta,summary,decisions,set(indices.tolist())))
    wm,ws,wd,wa=outputs[0];lm,ls,ld,la=outputs[1]
    assert la<=wa and lm['all_contexts']
    assert lm['preceding_guard_sha256']==sha(whole/'provenance.json')
    assert lm['quarantine_sha256']==sha(pool/'selection.json')
    cache=list(map(json.loads,(whole/'teacher_pages.jsonl').read_text().splitlines()))
    assert len(cache)==len({c['page_id'] for c in cache})
    cached={c['page_id']:c for c in cache}
    paths={r['page_id']:ROOT/'datasets'/r['parent_image_path'] for r in rows}
    shapes={}
    for page,path in paths.items():
        assert cached[page]['source_image_sha256']==sha(path)
        image=cv2.imread(str(path));assert image is not None
        shapes[page]=image.shape[:2]
    for i,row in enumerate(rows):
        boxes=[d[:4] for detections in cached[row['page_id']]['predictions'].values() for d in detections if d[4]>=wm['threshold']]
        reason='quarantined_source_page' if row['page_id'] in wm['quarantine_pages'] else ('teacher_face_overlap' if near_face(row['bbox_xyxy'],boxes) else None)
        assert wd[i]['reason']==(reason or 'source_and_teacher_exclusion')
        assert (i in wa)==(reason is None)
        if row['page_id'] in lm['quarantined_pages']:
            reason='confirmed_face_source_page_quarantine'
        elif i not in wa:
            reason='preceding_teacher_guard_rejected'
        else:
            h,w=shapes[row['page_id']];box=row['bbox_xyxy']
            side=max(box[2]-box[0],box[3]-box[1]);half=max(192,3*side)/2
            cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
            expected=[max(0,int(cx-half)),max(0,int(cy-half)),min(w,int(np.ceil(cx+half))),min(h,int(np.ceil(cy+half)))]
            assert ld[i]['context']==expected
            assert all(len(d)==5 and all(np.isfinite(d)) and d[4]>=lm['threshold'] for d in ld[i]['teacher_detections'])
            reason='local_teacher_face_overlap' if near_face(box,[d[:4] for d in ld[i]['teacher_detections']]) else None
        assert ld[i]['reason']==reason and (i in la)==(reason is None)
    review=json.loads((ROOT/'results/cascade_auto_pipeline_backgrounds_visual_review.json').read_text())
    suspect_set=set(review['suspect_ids'])
    suspects=[{'id':r['id'],'page':r['page_id'],'whole_keep':i in wa,'local_keep':i in la,
               'forced_page_quarantine':r['page_id'] in review['quarantined_pages']} for i,r in enumerate(rows) if r['id'] in suspect_set]
    assert len(suspects)==len(suspect_set) and all(s['forced_page_quarantine'] for s in suspects)
    report={'samples':len(rows),'whole_accepted':len(wa),'local_accepted':len(la),'teacher_pages':len(cache),
            'all_saved_decisions_replayed':True,'local_acceptance_subset_of_whole':True,'suspects':suspects,
            'source_sha256':sha(pool/'candidates.jsonl'),'script_sha256':sha(Path(__file__)),
            'guard_shas':{d.name:sha(d/'provenance.json') for d in [whole,local]},
            'limitation':'Replays saved predictions and verifies source image hashes; does not independently rerun neural teachers or certify all accepted windows.'}
    report_path.write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


if __name__=='__main__':
    main()
