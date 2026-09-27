"""Render six diagnosed train NMS losses; source artwork stays ignored."""
import json
from pathlib import Path
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]


def main():
    report=json.loads((ROOT/'results/cascade_auto_mined_v1_train_proposals.json').read_text())
    cases=[(p,f) for p in report['pages'] for f in p['faces'] if f['reason']=='nms_suppressed'][:6]
    sheet=Image.new('RGB',(1080,780),'white')
    text=ImageDraw.Draw(sheet)
    for i,(page,face) in enumerate(cases):
        im=Image.open(ROOT/'datasets'/page['image_path']).convert('RGB')
        gt=face['gt_bbox']; side=max(gt[2]-gt[0],gt[3]-gt[1])
        bounds=[max(0,int(gt[0]-side)),max(0,int(gt[1]-side)),min(im.width,int(gt[2]+side)),min(im.height,int(gt[3]+side))]
        crop=im.crop(bounds); draw=ImageDraw.Draw(crop)
        for box,color in [(gt,'red'),(face['pre_nms_best']['bbox'],'blue'),(face['post_nms_best']['bbox'],'green')]:
            draw.rectangle([box[0]-bounds[0],box[1]-bounds[1],box[2]-bounds[0],box[3]-bounds[1]],outline=color,width=2)
        scale=350/max(crop.size)
        crop=crop.resize((round(crop.width*scale),round(crop.height*scale)))
        x=i%3*360; y=i//3*390
        sheet.paste(crop,(x,y))
        text.text((x,y+354),page['source_group']+' #'+str(face['annotation_index']),fill='black')
        text.text((x,y+370),'raw %.2f -> NMS %.2f'%(face['pre_nms_best']['iou'],face['post_nms_best']['iou']),fill='black')
    sheet.save(ROOT/'datasets/derived/cascade_auto_mined_v1/train_nms_failures.png')


if __name__=='__main__':
    main()
