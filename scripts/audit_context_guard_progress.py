"""Audit a complete-line snapshot of the live context guard; render six vetoes."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from prepare_detection_dataset import near_face

ROOT=Path(__file__).resolve().parents[1]


def main():
    directory=ROOT/'datasets/derived/cascade_auto_background_context_guard_v1'
    payload=(directory/'decisions.jsonl').read_bytes()
    payload=payload[:payload.rfind(b'\n')+1]
    decisions=list(map(json.loads,payload.decode().splitlines()))
    source=ROOT/'datasets/derived/cascade_auto_background_merged_v1/candidates.jsonl'
    rows=list(map(json.loads,source.read_text().splitlines()))
    metadata=json.loads((directory/'provenance.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==metadata['source_sha256']
    quarantine=set(metadata['quarantined_pages'])
    for index,decision in enumerate(decisions):
        row=rows[index]
        assert decision['array_index']==index and decision['id']==row['id'] and decision['page_id']==row['page_id']
        assert row['split']=='train'
        boxes=decision['teacher_detections']
        assert all(len(d)==5 and np.isfinite(d).all() and d[4]>=.1 for d in boxes)
        if row['page_id'] in quarantine:
            expected='confirmed_face_source_page_quarantine'
        elif near_face(row['bbox_xyxy'],[d[:4] for d in boxes]):
            expected='local_teacher_face_overlap'
        else:
            expected=None
        assert decision['reason']==expected
        assert decision['decision']==('reject' if expected else 'keep_auto')
    rejected=[r for r in decisions if r['reason']=='local_teacher_face_overlap']
    positions=np.linspace(0,len(rejected)-1,min(6,len(rejected)),dtype=int)
    cases=[rejected[i] for i in positions]
    sheet=Image.new('RGB',(1260,720),'white');labels=ImageDraw.Draw(sheet)
    for cell,case in enumerate(cases):
        row=rows[case['array_index']];bounds=case['context']
        image=Image.open(ROOT/'datasets'/row['parent_image_path']).convert('RGB').crop(bounds)
        draw=ImageDraw.Draw(image)
        for box,color in [(row['bbox_xyxy'],'red'),*[(b[:4],'blue') for b in case['teacher_detections']]]:
            draw.rectangle([box[0]-bounds[0],box[1]-bounds[1],box[2]-bounds[0],box[3]-bounds[1]],outline=color,width=2)
        scale=min(410/image.width,320/image.height)
        image=image.resize((round(image.width*scale),round(image.height*scale)))
        x,y=cell%3*420,cell//3*360;sheet.paste(image,(x,y))
        labels.text((x+3,y+323),row['page_id'].split(':',1)[1],fill='black')
        labels.text((x+3,y+339),row.get('source_candidate_id',row['id']),fill='black')
    sheet.save(directory/'progress_vetoes6.png')
    report={'snapshot_count':len(decisions),'snapshot_sha256':hashlib.sha256(payload).hexdigest(),
        'local_contexts_in_snapshot':sum(r['context'] is not None for r in decisions),
        'reasons':dict(Counter(r['reason'] or 'keep_auto' for r in decisions)),
        'checks':'complete-line snapshot IDs, train labels, finite thresholded teacher boxes, expanded overlap rule and recorded decisions; not model accuracy',
        'sheet':str((directory/'progress_vetoes6.png').relative_to(ROOT)),
        'sheet_ids':[rows[r['array_index']]['id'] for r in cases],
        'colors':'red background candidate; blue local teacher face boxes',
        'visual_review':'pending; snapshot only, final guard not completed'}
    (ROOT/'results/cascade_auto_context_guard_progress_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:report[k] for k in ('snapshot_count','local_contexts_in_snapshot','reasons')})


if __name__=='__main__':
    main()
