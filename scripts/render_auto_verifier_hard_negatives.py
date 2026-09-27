"""Render the highest-scored twelve windows of each mined negative category."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'datasets/derived/cascade_auto_proposal_verifier_v2'
    sheet=Image.new('RGB',(1200,540),'white');draw=ImageDraw.Draw(sheet)
    cases=[]
    for group,name in enumerate(('cascade_auto_background_merged_v1','cascade_localization_negatives_v1')):
        source=ROOT/'datasets/derived'/name
        rows=list(map(json.loads,(source/'candidates.jsonl').read_text().splitlines()))
        cache=np.load(source/'channels11.npy',mmap_mode='r')
        with np.load(output/(name+'_mining.npz')) as mining:
            indices=mining['selected_indices'][:12]
            scores=dict(zip(mining['eligible_indices'].tolist(),mining['scores'].tolist()))
        for cell,index in enumerate(indices):
            x,y=cell%6*200,(group*2+cell//6)*135
            sheet.paste(Image.fromarray(cache[index,0]).resize((96,96)).convert('RGB'),(x+52,y))
            identifier=rows[index].get('source_candidate_id',rows[index]['id'])
            draw.text((x+5,y+99),('BG ' if group==0 else 'LOC ')+identifier,fill='black')
            draw.text((x+5,y+116),'margin %.3f'%scores[int(index)],fill='black')
            cases.append({'dataset':name,'source_index':int(index),'id':rows[index]['id'],
                          'page_id':rows[index]['page_id'],'margin':scores[int(index)],'label_basis':rows[index]['label_basis']})
    sheet.save(output/'hard_negatives24.png')
    report={'cases':cases,'sheet':str((output/'hard_negatives24.png').relative_to(ROOT)),
        'selection':'top twelve mining margins per category; BG source/teacher-excluded background, LOC geometrically nonmatching windows, may contain partial face',
        'visual_review':'pending; not human approved'}
    (ROOT/'results/cascade_auto_proposal_verifier_v2_hard_negative_sheet.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
