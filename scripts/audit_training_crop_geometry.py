"""Measure reviewed positive crop geometry without changing annotations."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / 'datasets/derived/step3_detection_v1/usable_samples.jsonl'
    rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
    rows = [r for r in rows if r['split'] == 'train' and r['label'] == 1 and not r['augmentation'] and r['review_decision'] == 'keep']
    boxes = np.asarray([r['crop_bbox_xyxy'] for r in rows])
    gt = np.asarray([r['source_bbox_xyxy'] for r in rows])
    size, gt_size = boxes[:,2:] - boxes[:,:2], gt[:,2:] - gt[:,:2]
    overlap = np.maximum(0,np.minimum(boxes[:,2:],gt[:,2:])-np.maximum(boxes[:,:2],gt[:,:2]))
    intersection = np.prod(overlap,axis=1)
    iou = intersection / (np.prod(size,axis=1)+np.prod(gt_size,axis=1)-intersection)
    levels = [0,.1,.5,.9,1]
    report = {'split':'train', 'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'unaugmented_reviewed_positive_count':len(rows), 'square_crop_count':int((size[:,0]==size[:,1]).sum()),
              'crop_aspect_ratio_quantiles':np.quantile(size[:,0]/size[:,1],levels).tolist(),
              'crop_gt_iou_quantiles':np.quantile(iou,levels).tolist(), 'iou_below_05_count':int((iou<.5).sum()),
              'crop_to_gt_max_side_quantiles':np.quantile(size[:,0]/gt_size.max(axis=1),levels).tolist(), 'quantiles':levels}
    (ROOT / 'results/step8_train_crop_geometry.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__ == '__main__':
    main()
