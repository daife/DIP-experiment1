"""Check a second teacher on the six already inspected local-veto contexts."""
import hashlib
import json
from pathlib import Path
import cv2
import torch
from anime_face_detector import get_checkpoint_path
from anime_face_detector._face import load_face_detector
from prepare_detection_dataset import near_face

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    directory=ROOT/'datasets/derived/cascade_auto_background_context_guard_v1'
    audit_path=ROOT/'results/cascade_auto_context_guard_progress_audit.json'
    audit=json.loads(audit_path.read_text())
    payload=(directory/'decisions.jsonl').read_bytes()
    payload=payload[:payload.rfind(b'\n')+1]
    lines=payload.splitlines(keepends=True)
    assert hashlib.sha256(b''.join(lines[:audit['snapshot_count']])).hexdigest()==audit['snapshot_sha256']
    decisions={r['id']:r for r in map(json.loads,payload.decode().splitlines())}
    source=ROOT/'datasets/derived/cascade_auto_background_merged_v1/candidates.jsonl'
    rows={r['id']:r for r in map(json.loads,source.read_text().splitlines())}
    torch.set_num_threads(2);cv2.setNumThreads(1)
    checkpoint=get_checkpoint_path('faster-rcnn');model=load_face_detector('faster-rcnn',checkpoint,'cpu')
    result=[]
    for identifier in audit['sheet_ids']:
        row=rows[identifier];previous=decisions[identifier]
        path=ROOT/'datasets'/row['parent_image_path']
        image=cv2.imread(str(path));x1,y1,x2,y2=previous['context']
        crop=image[y1:y2,x1:x2];detections=[]
        for detection in model.detect(crop):
            if detection[4]>=.1:
                detections.append([float(detection[0]+x1),float(detection[1]+y1),float(detection[2]+x1),float(detection[3]+y1),float(detection[4])])
        item={'id':identifier,'page_id':row['page_id'],'candidate':row['bbox_xyxy'],
            'context':previous['context'],'source_image_sha256':sha(path),
            'context_pixel_sha256':hashlib.sha256(crop.tobytes()).hexdigest(),
            'yolo_boxes':previous['teacher_detections'],'faster_boxes':detections,
            'faster_overlap_veto':near_face(row['bbox_xyxy'],[d[:4] for d in detections])}
        result.append(item);print({'id':identifier,'faster_overlap_veto':item['faster_overlap_veto']},flush=True)
    report={'cases':result,'cases_count':len(result),'both_teacher_veto':sum(r['faster_overlap_veto'] for r in result),
        'second_teacher':'faster-rcnn','threshold':.1,'teacher_sha256':sha(checkpoint),
        'snapshot_audit_sha256':sha(audit_path),'script_sha256':sha(Path(__file__)),
        'scope':'six already inspected evenly spaced snapshot vetoes; not a representative accuracy estimate or a modification of running guard',
        'limitation':'Teacher agreement can still be wrong; this diagnostic does not relabel or approve any data.'}
    (ROOT/'results/cascade_auto_context_teacher_agreement6.json').write_text(json.dumps(report,indent=2)+'\n')
    print({'cases_count':len(result),'both_teacher_veto':report['both_teacher_veto']},flush=True)


if __name__=='__main__':
    main()
