"""Render saved before/after detections around each annotated face."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--before',required=True);parser.add_argument('--after',required=True)
    parser.add_argument('--page-contains',action='append',required=True)
    args=parser.parse_args()
    rows=json.loads(args.input.read_text())
    for page in rows:
        if not any(s in page['image_id'] for s in args.page_contains):continue
        image=cv2.imread(str(ROOT/'datasets'/page['image_path']))
        cells=[]
        for gi,gt in enumerate(page['gt']):
            target=np.asarray(gt);center=(target[:2]+target[2:])/2;side=max(target[2:]-target[:2])*2.4
            lo=np.maximum(0,np.floor(center-side/2)).astype(int)
            hi=np.minimum(image.shape[1::-1],np.ceil(center+side/2)).astype(int)
            crop=image[lo[1]:hi[1],lo[0]:hi[0]].copy()
            for name,color in ((args.before,(255,0,0)),(args.after,(0,0,255))):
                for b in page['variants'][name]['boxes']:
                    b=np.asarray(b)
                    if np.all(np.minimum(b[2:],hi)>np.maximum(b[:2],lo)):
                        cv2.rectangle(crop,tuple(b[:2]-lo),tuple(b[2:]-lo),color,2)
            cv2.rectangle(crop,tuple(target[:2]-lo),tuple(target[2:]-lo),(0,180,0),2)
            cell=np.full((275,240,3),255,np.uint8);cell[35:]=cv2.resize(crop,(240,240))
            matched=[any(m['ground_truth_index']==gi for m in page['variants'][k]['matches']) for k in (args.before,args.after)]
            cv2.putText(cell,f'GT{gi} matched {int(matched[0])}->{int(matched[1])}',(5,23),0,.5,(0,0,0),1)
            cells.append(cell)
        sheet=np.full(((len(cells)+3)//4*275,960,3),255,np.uint8)
        for i,c in enumerate(cells):sheet[i//4*275:(i//4+1)*275,i%4*240:(i%4+1)*240]=c
        path=args.input.parent/(page['image_id'].replace(':','_')+'_audit.jpg')
        if not cv2.imwrite(str(path),sheet):raise IOError(path)
        print(path)


if __name__=='__main__':main()
