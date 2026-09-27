"""Trace all validation GT through cached NMS proposals, HOG, and size gate."""
import argparse
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
import sys
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_box_refiner import ious


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'datasets/derived/cascade_auto_curated_localization_v1')
    parser.add_argument('--source', type=Path, default=ROOT/'results/cascade_auto_curated_localization_v1_validation20_combined.json')
    parser.add_argument('--verifier', type=Path, default=ROOT/'datasets/derived/step8_tight_crop_verifier/model.joblib')
    parser.add_argument('--threshold', type=float, default=1.75)
    parser.add_argument('--prediction-dir', type=Path, help='Defaults to directory/combined20')
    parser.add_argument('--output', type=Path, default=ROOT/'results/cascade_auto_cached_face_diagnosis20.json')
    args = parser.parse_args()
    directory = args.directory.resolve()
    source = args.source.resolve()
    report = json.loads(source.read_text())
    prediction_path = (args.prediction_dir.resolve() if args.prediction_dir else directory/'combined20')/'predictions.json'
    predictions = json.loads(prediction_path.read_text())
    assert [r['image_id'] for r in report['pages']]==[r['image_id'] for r in predictions]
    state = {p.name:sha(p) for p in (ROOT/'src').glob('*.py')}
    assert json.loads((directory/'validation_cache/source_state.json').read_text())==state
    cache = {sha(p):p for p in (directory/'validation_cache').glob('*.npz')}
    model_path = args.verifier.resolve()
    model = joblib.load(model_path)
    assert report['sha256'][str(model_path.relative_to(ROOT))]==sha(model_path)
    details = []
    for page,pred in zip(report['pages'],predictions):
        with np.load(cache[page['cache_sha256']]) as data:
            boxes,x = data['boxes'],data['features']
        gt = np.asarray(pred['gt']).reshape(-1,4)
        overlap = ious(boxes,gt)
        confidence = model.decision_function(x)
        hog_gate = confidence>=args.threshold
        side_gate = boxes[:,2]-boxes[:,0]>=36
        for index,face in enumerate(gt):
            widths = face[2:]-face[:2]
            side = float(max(widths));ratio = float(min(widths)/max(widths))
            available = overlap[:,index]>=.5
            if not np.any(available):
                reason = 'no_IoU50_postNMS_candidate'
            elif not np.any(available & hog_gate):
                reason = 'HOG_filtered'
            elif not np.any(available & hog_gate & side_gate):
                reason = 'min_side_filtered'
            else:
                reason = 'candidate_available'
            size = 'le36' if side<=36 else '37to64' if side<=64 else '65to128' if side<=128 else 'gt128'
            details.append({'image_id':page['image_id'],'annotation_index':index,'gt':face.tolist(),
                'size_group':size,'reason':reason,'gt_max_side':side,'gt_aspect_ratio':ratio,
                'ideal_square_IoU_upper_bound':float(ratio/(2*np.sqrt(ratio)-ratio)),
                'best_postNMS_IoU':float(overlap[:,index].max()) if len(boxes) else 0,
                'best_HOG_margin_of_IoU50_candidates':float(confidence[available].max()) if np.any(available) else None})
    by_size=defaultdict(Counter)
    for row in details:
        by_size[row['size_group']][row['reason']]+=1
    summary={'faces':len(details),'reasons':dict(Counter(r['reason'] for r in details)),
             'size_groups':{k:dict(v) for k,v in by_size.items()},
             'square_geometry_cannot_reach_IoU50':sum(r['ideal_square_IoU_upper_bound']<.5 for r in details)}
    output={'summary':summary,'faces':details,'threshold':args.threshold,'minimum_proposal_side':36,
            'source_sha256':sha(source),'script_sha256':sha(Path(__file__)),
            'prediction_sha256':sha(prediction_path),'HOG_sha256':sha(model_path),
            'scope':'20 explored validation pages, 252 GT; candidate availability after NMS, not one-to-one TP or pre-NMS Cascade recall',
            'square_geometry_bound':'Ideal aligned square maximum IoU for GT ratio r=min(w,h)/max(w,h): r/(2sqrt(r)-r); no image or classifier effects'}
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':
    main()
