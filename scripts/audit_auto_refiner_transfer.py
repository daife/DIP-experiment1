"""Inspect actual match changes caused by the old proposal box refiner."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def closest(boxes, gt):
    boxes = np.asarray(boxes).reshape(-1,4)
    if not len(boxes):
        return None
    overlap = np.maximum(0,np.minimum(boxes[:,2:],gt[2:])-np.maximum(boxes[:,:2],gt[:2]))
    inter = np.prod(overlap,axis=1)
    iou = inter/(np.prod(boxes[:,2:]-boxes[:,:2],axis=1)+np.prod(np.asarray(gt[2:])-gt[:2])-inter)
    index = int(np.argmax(iou))
    return {'bbox':boxes[index].tolist(),'iou':float(iou[index])}


def main():
    source = ROOT/'datasets/derived/cascade_auto_curated_localization_v1/combined20/predictions.json'
    pages = json.loads(source.read_text())
    transitions = []
    for threshold in ('1.5','1.75'):
        for page in pages:
            raw = page['variants']['tight_postnms_'+threshold]
            refined = page['variants']['tight_refined_'+threshold]
            before = {m['ground_truth_index'] for m in raw['matches']}
            after = {m['ground_truth_index'] for m in refined['matches']}
            for category, indices in [('lost',before-after),('gained',after-before)]:
                for index in sorted(indices):
                    gt = page['gt'][index]
                    transitions.append({'image_id':page['image_id'],'image_path':page['image_path'],
                        'threshold':threshold,'category':category,'annotation_index':index,'gt':gt,
                        'raw_best':closest(raw['boxes'],gt),'refined_best':closest(refined['boxes'],gt)})
    cases = [r for r in transitions if r['threshold']=='1.75' and r['category']=='lost'][:6]
    sheet = Image.new('RGB',(1080,780),'white')
    labels = ImageDraw.Draw(sheet)
    for cell,case in enumerate(cases):
        image = Image.open(ROOT/'datasets'/case['image_path']).convert('RGB')
        gt = case['gt']; side = max(gt[2]-gt[0],gt[3]-gt[1])
        bounds = [max(0,int(gt[0]-side)),max(0,int(gt[1]-side)),min(image.width,int(gt[2]+side)),min(image.height,int(gt[3]+side))]
        crop = image.crop(bounds); draw = ImageDraw.Draw(crop)
        for box,color in [(gt,'red'),(case['raw_best']['bbox'],'green'),(case['refined_best']['bbox'],'blue')]:
            draw.rectangle([box[0]-bounds[0],box[1]-bounds[1],box[2]-bounds[0],box[3]-bounds[1]],outline=color,width=3)
        scale = 350/max(crop.size)
        crop = crop.resize((round(crop.width*scale),round(crop.height*scale)))
        x,y = cell%3*360,cell//3*390
        sheet.paste(crop,(x,y))
        labels.text((x,y+352),case['image_id'].split(':',1)[1]+' #'+str(case['annotation_index']),fill='black')
        labels.text((x,y+368),'raw %.3f -> refined %.3f'%(case['raw_best']['iou'],case['refined_best']['iou']),fill='black')
    destination = source.parent/'refiner_losses.png'
    sheet.save(destination)
    report = {'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope':'fixed 20 explored validation pages; one-to-one match transitions, not independent face availability',
              'counts':{t:{c:sum(r['threshold']==t and r['category']==c for r in transitions) for c in ('lost','gained')} for t in ('1.5','1.75')},
              'transitions':transitions,'sheet':str(destination.relative_to(ROOT)),
              'sheet_cases':[{k:r[k] for k in ('image_id','annotation_index','threshold')} for r in cases],
              'colors':'red GT; green highest-IoU original post-NMS box; blue highest-IoU refined box',
              'visual_review':'pending; image is a diagnostic of old refiner transfer, not all-dataset review'}
    (ROOT/'results/cascade_auto_refiner_transfer20.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['counts'])


if __name__=='__main__':
    main()
