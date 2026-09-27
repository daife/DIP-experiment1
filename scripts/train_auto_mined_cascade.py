"""Train expanded Cascade plus automatically mined train windows.

Calibrate on annotation-derived validation crops with the same geometry as
training. Keep the historical reviewed-crop evaluation separately for comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cascade import fit_cascade, cascade_to_dict, stage_statistics
from src.channels11 import compute_11_channels
from prepare_detection_dataset import ordered_round_robin, square_box, near_face
from train_cascade import load_reviewed_data


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=ROOT/'datasets/derived/cascade_auto_v1')
    p.add_argument('--mined', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stages', type=int, default=5)
    p.add_argument('--trees', type=int, default=32)
    p.add_argument('--seed', type=int, default=20260926)
    p.add_argument('--histogram-splits', action='store_true', help='Use bounded-integer weighted split search')
    p.add_argument('--mined-guard', type=Path)
    p.add_argument('--localization-negatives', type=Path)
    p.add_argument('--base-negative-count', type=int, default=20000)
    p.add_argument('--positive-hog-min', type=float, help='Fixed teacher margin for automatic train/calibration positive selection')
    p.add_argument('--root-candidates', type=int, default=128)
    p.add_argument('--child-candidates', type=int, default=48)
    p.add_argument('--stage-recall', type=float, default=.99)
    p.add_argument('--balance-negative-types', action='store_true')
    p.add_argument('--negative-quarantine',type=Path,help='JSON with quarantined_pages; exclude all negatives from these train pages')
    p.add_argument('--proposal-positives',type=Path,help='GT-matched train proposal/jitter positives, without old HOG selection')
    p.add_argument('--nms-competitors',type=Path,help='Train nonmatching NMS competitors with positive suppression witnesses')
    args = p.parse_args()
    if min(args.stages,args.trees,args.root_candidates,args.child_candidates)<1 or args.base_negative_count<0 or not 0<args.stage_recall<=1:
        p.error('invalid training counts or recall')
    if args.histogram_splits:
        import src.weak_tree_training as training_backend
        from benchmark_cascade_split import histogram_split
        training_backend._best_split = histogram_split
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'model.json').exists():
        p.error('output model already exists')
    started = perf_counter()
    manifest = ROOT/'datasets/manifests/normalized/manga109_faces.jsonl'
    pages = list(map(json.loads, manifest.read_text(encoding='utf-8').splitlines()))
    lookup = {page['image_id']:page for page in pages}
    quarantined=set(json.loads(args.negative_quarantine.read_text())['quarantined_pages']) if args.negative_quarantine else set()
    assert all(lookup[page]['split']=='train' for page in quarantined)
    base_rows = list(map(json.loads,(args.dataset/'samples.jsonl').read_text(encoding='utf-8').splitlines()))
    labels = np.load(args.dataset/'labels.npy')
    base = np.load(args.dataset/'channels.npy',mmap_mode='r')
    assert len(base_rows) == len(base) == len(labels)
    assert all(r['split']==lookup[r['page_id']]['split']=='train' and r['label']==labels[i] for i,r in enumerate(base_rows))
    rng = np.random.default_rng(args.seed)
    positives=np.flatnonzero(labels==1)
    teacher_models=[]
    if args.positive_hog_min is not None:
        import joblib
        from src.candidate_verifier import hog
        teacher_models=[joblib.load(ROOT/f'models/{name}') for name in ('candidate_hog_rbf_step3_v1.joblib','candidate_hog_rbf_tight_v1.joblib')]
        teacher_x=np.stack([hog(base[i,0]) for i in positives])
        teacher_scores=np.max(np.stack([m.decision_function(teacher_x) for m in teacher_models]),axis=0)
        np.savez(args.output/'positive_teacher_scores.npz',base_indices=positives,scores=teacher_scores)
        positives=positives[teacher_scores>=args.positive_hog_min]
    eligible_base=np.asarray([i for i in np.flatnonzero(labels==0) if base_rows[i]['page_id'] not in quarantined],dtype=np.int64)
    selected = np.concatenate([positives,rng.choice(eligible_base,args.base_negative_count,replace=False)])
    rows = list(map(json.loads,(args.mined/'candidates.jsonl').read_text(encoding='utf-8').splitlines()))
    hard = np.load(args.mined/'channels11.npy',mmap_mode='r')
    assert hard.shape == (len(rows),11,24,24)
    eligible=set(range(len(rows)))
    if args.mined_guard:
        guard=json.loads((args.mined_guard/'provenance.json').read_text())
        assert guard['source_sha256']==sha(args.mined/'candidates.jsonl')
        guard_summary=json.loads((args.mined_guard/'summary.json').read_text())
        assert guard_summary['accepted_indices_sha256']==sha(args.mined_guard/'accepted_indices.npy')
        allowed=np.load(args.mined_guard/'accepted_indices.npy')
        assert len(set(allowed))==len(allowed) and np.all((allowed>=0)&(allowed<len(rows)))
        eligible=set(allowed.tolist())
    seen = {base_rows[i]['crop_sha256'] for i in selected}
    mined_indices = []
    for i,row in enumerate(rows):
        page = lookup[row['page_id']]
        assert row['split']==page['split']=='train' and row['label']==0 and row['review_decision']=='unreviewed'
        assert not near_face(row['bbox_xyxy'],[a['bbox'] for a in page['annotations']])
        digest = hashlib.sha256(hard[i,0].tobytes()).hexdigest()
        assert digest == row['crop_sha256']
        if i in eligible and row['page_id'] not in quarantined and digest not in seen:
            mined_indices.append(i)
            seen.add(digest)
    x = np.concatenate([base[selected],hard[mined_indices]])
    y = np.concatenate([labels[selected],np.zeros(len(mined_indices),dtype=np.uint8)])
    negative_groups=np.where(y==1,-1,0).astype(np.int8)
    localization_indices=[]
    if args.localization_negatives:
        from build_localization_negatives import geometry
        local_rows=list(map(json.loads,(args.localization_negatives/'candidates.jsonl').read_text().splitlines()))
        local_x=np.load(args.localization_negatives/'channels11.npy',mmap_mode='r')
        assert local_x.shape==(len(local_rows),11,24,24)
        for i,row in enumerate(local_rows):
            page=lookup[row['page_id']]
            assert row['split']==page['split']=='train' and row['label']==0 and row['label_basis']=='nonmatching_localization_window'
            iou,coverage=geometry(row['bbox_xyxy'],[a['bbox'] for a in page['annotations']])
            assert 0<iou<=.3 and coverage<=.6
            digest=hashlib.sha256(local_x[i,0].tobytes()).hexdigest()
            assert digest==row['crop_sha256']
            if row['page_id'] not in quarantined and digest not in seen:
                localization_indices.append(i);seen.add(digest)
        x=np.concatenate([x,local_x[localization_indices]])
        y=np.concatenate([y,np.zeros(len(localization_indices),dtype=np.uint8)])
        negative_groups=np.concatenate([negative_groups,np.ones(len(localization_indices),dtype=np.int8)])
        np.save(args.output/'localization_indices.npy',np.asarray(localization_indices))
    proposal_indices=[]
    if args.proposal_positives:
        from train_candidate_verifier import iou_max
        proposal_rows=list(map(json.loads,(args.proposal_positives/'candidates.jsonl').read_text().splitlines()))
        proposal_x=np.load(args.proposal_positives/'channels11.npy',mmap_mode='r')
        assert proposal_x.shape==(len(proposal_rows),11,24,24) and proposal_x.dtype==np.uint8
        for i,row in enumerate(proposal_rows):
            page=lookup[row['page_id']]
            assert row['split']==page['split']=='train' and row['label']==1
            box=row['bbox_xyxy'];x1,y1,x2,y2=box
            assert 0<=x1<x2<=page['width'] and 0<=y1<y2<=page['height']
            overlap=float(iou_max(np.asarray([box]),np.asarray([a['bbox'] for a in page['annotations']]))[0])
            assert overlap>=.5 and abs(overlap-row['max_annotation_IoU'])<1e-10
            digest=hashlib.sha256(proposal_x[i,0].tobytes()).hexdigest()
            assert digest==row['crop_sha256']
            if digest not in seen:
                proposal_indices.append(i);seen.add(digest)
        assert proposal_indices, 'no unique additional positive samples'
        x=np.concatenate([x,proposal_x[proposal_indices]])
        y=np.concatenate([y,np.ones(len(proposal_indices),dtype=np.uint8)])
        negative_groups=np.concatenate([negative_groups,np.full(len(proposal_indices),-1,dtype=np.int8)])
        np.save(args.output/'proposal_positive_indices.npy',np.asarray(proposal_indices))
    competitor_indices=[]
    if args.nms_competitors:
        from build_localization_negatives import geometry
        from train_candidate_verifier import iou_max
        from train_box_refiner import ious
        competitor_rows=list(map(json.loads,(args.nms_competitors/'candidates.jsonl').read_text().splitlines()))
        competitor_x=np.load(args.nms_competitors/'channels11.npy',mmap_mode='r')
        competitor_meta=json.loads((args.nms_competitors/'provenance.json').read_text())
        assert competitor_meta['candidate_sha256']==sha(args.nms_competitors/'candidates.jsonl')
        assert competitor_meta['channels_sha256']==sha(args.nms_competitors/'channels11.npy')
        assert competitor_x.shape==(len(competitor_rows),11,24,24) and competitor_x.dtype==np.uint8
        for i,row in enumerate(competitor_rows):
            page=lookup[row['page_id']];box=row['bbox_xyxy'];x1,y1,x2,y2=box
            gt=np.asarray([a['bbox'] for a in page['annotations']])
            assert row['split']==page['split']=='train' and row['label']==0
            assert row['label_basis']=='nonmatching_NMS_competitor'
            assert 0<=x1<x2<=page['width'] and 0<=y1<y2<=page['height']
            overlap,coverage=geometry(box,gt)
            assert .1<=overlap<.5 and coverage<=.8
            assert abs(overlap-row['max_annotation_IoU'])<1e-10 and abs(coverage-row['max_gt_coverage'])<1e-10
            assert row['witnesses']
            for witness in row['witnesses']:
                positive=np.asarray([witness['positive_bbox']]);index=witness['annotation_index']
                match=float(ious(positive,gt[index:index+1])[0,0])
                competition=float(ious(np.asarray([box]),positive)[0,0])
                assert match>=.5 and competition>.3
                assert abs(match-witness['positive_GT_IoU'])<1e-10 and abs(competition-witness['competitor_IoU'])<1e-10
                assert row['cascade_score']>=witness['positive_score']
            digest=hashlib.sha256(competitor_x[i,0].tobytes()).hexdigest()
            assert digest==row['crop_sha256']
            if row['page_id'] not in quarantined and digest not in seen:
                competitor_indices.append(i);seen.add(digest)
        assert competitor_indices, 'no unique eligible NMS competitors'
        x=np.concatenate([x,competitor_x[competitor_indices]])
        y=np.concatenate([y,np.zeros(len(competitor_indices),dtype=np.uint8)])
        negative_groups=np.concatenate([negative_groups,np.full(len(competitor_indices),2,dtype=np.int8)])
        np.save(args.output/'nms_competitor_indices.npy',np.asarray(competitor_indices))
    validation, validation_rows = [], []
    for page,index in ordered_round_robin([r for r in pages if r['split']=='validation'],'auto-calibration-v1'):
        box = square_box(page['annotations'][index]['bbox'],margin=1.0)
        x1,y1,x2,y2 = box
        if min(x1,y1)<0 or x2>page['width'] or y2>page['height']:
            continue
        gray = cv2.imread(str(ROOT/'datasets'/page['image_path']),0)
        if gray is None:
            raise ValueError(page['image_path'])
        crop = cv2.resize(gray[y1:y2,x1:x2],(24,24),interpolation=cv2.INTER_AREA)
        validation.append(np.stack(compute_11_channels(crop)))
        validation_rows.append({'page_id':page['image_id'],'annotation_index':index,'bbox_xyxy':box,'split':'validation'})
        if len(validation)==800:
            break
    validation = np.stack(validation)
    calibration_teacher_count=len(validation)
    if teacher_models:
        scores=np.max(np.stack([m.decision_function(np.stack([hog(v[0]) for v in validation])) for m in teacher_models]),axis=0)
        keep=scores>=args.positive_hog_min
        np.savez(args.output/'calibration_teacher_scores.npz',scores=scores,keep=keep)
        validation=validation[keep]
        validation_rows=[row for row,k in zip(validation_rows,keep) if k]
    np.save(args.output/'base_indices.npy',selected)
    np.save(args.output/'mined_indices.npy',np.asarray(mined_indices))
    (args.output/'calibration.json').write_text(json.dumps(validation_rows)+'\n')
    provenance = {'seed':args.seed,'positive':int(y.sum()),'negative':int((y==0).sum()),
                  'quarantined_negative_pages':sorted(quarantined),
                  'negative_quarantine_sha256':sha(args.negative_quarantine) if args.negative_quarantine else None,
                  'split_backend':'histogram511' if args.histogram_splits else 'stable_sort',
                  'split_backend_sha256':sha(ROOT/'scripts/benchmark_cascade_split.py') if args.histogram_splits else sha(ROOT/'src/weak_tree_training.py'),
                  'fresh_mined_negative':len(mined_indices),'fully_reviewed':False,'stages':args.stages,'trees':args.trees,
                  'root_candidates':args.root_candidates,'child_candidates':args.child_candidates,'target_stage_recall':args.stage_recall,
                  'base_negative_count':args.base_negative_count,'positive_hog_min':args.positive_hog_min,
                  'positive_teacher_shas':{name:sha(ROOT/f'models/{name}') for name in ('candidate_hog_rbf_step3_v1.joblib','candidate_hog_rbf_tight_v1.joblib')} if teacher_models else {},
                  'guard_provenance_sha256':sha(args.mined_guard/'provenance.json') if args.mined_guard else None,
                  'guard_indices_sha256':sha(args.mined_guard/'accepted_indices.npy') if args.mined_guard else None,
                  'localization_negative_count':len(localization_indices),
                  'proposal_positive_count':len(proposal_indices),
                  'nms_competitor_count':len(competitor_indices),
                  'nms_competitor_manifest_sha256':sha(args.nms_competitors/'candidates.jsonl') if args.nms_competitors else None,
                  'proposal_positive_manifest_sha256':sha(args.proposal_positives/'candidates.jsonl') if args.proposal_positives else None,
                  'localization_manifest_sha256':sha(args.localization_negatives/'candidates.jsonl') if args.localization_negatives else None,
                  'calibration_before_teacher_selection':calibration_teacher_count,
                  'negative_type_initial_weighting':'equal category mass within negative class' if args.balance_negative_types else 'uniform negative samples',
                  'negative_type_trainer_sha256':sha(ROOT/'scripts/fit_group_balanced_cascade.py') if args.balance_negative_types else None,
                  'manifest_sha256':sha(manifest),'base_manifest_sha256':sha(args.dataset/'samples.jsonl'),
                  'mined_manifest_sha256':sha(args.mined/'candidates.jsonl'),'script_sha256':sha(Path(__file__)),
                  'base_indices_sha256':sha(args.output/'base_indices.npy'),'mined_indices_sha256':sha(args.output/'mined_indices.npy'),
                  'calibration_sha256':sha(args.output/'calibration.json'),'calibration_count':len(validation),
                  'calibration_geometry':'margin1.0 square, INTER_AREA, validation annotations, no augmentation'}
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance),flush=True)
    trainer=fit_cascade
    extra={}
    if args.balance_negative_types:
        from fit_group_balanced_cascade import fit_group_balanced_cascade
        trainer=fit_group_balanced_cascade
        extra={'negative_groups':negative_groups}
    model = trainer(x,y,validation,np.ones(len(validation),dtype=np.uint8),seed=args.seed,
                        num_stages=args.stages,num_trees=args.trees,root_candidates=args.root_candidates,child_candidates=args.child_candidates,target_recall=args.stage_recall,**extra)
    payload = cascade_to_dict(model)
    payload['training']=provenance
    (args.output/'model.json').write_text(json.dumps(payload,indent=2)+'\n')
    old = load_reviewed_data(ROOT/'datasets/derived/step3_detection_v1')['validation']
    stats={'train':stage_statistics(model,x,y),'calibration':stage_statistics(model,validation,np.ones(len(validation),dtype=np.uint8)),
           'historical_validation_crops':stage_statistics(model,old[0],old[1]),'seconds':perf_counter()-started}
    (args.output/'stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_candidate_verifier.py'),'--cascade',str(args.output/'model.json'),
                    '--verifier',str(ROOT/'models/candidate_hog_rbf_step3_v1.joblib'),'--output',str(args.output/'validation.json'),
                    '--cache-dir',str(args.output/'validation_cache')],check=True)


if __name__=='__main__':
    main()
