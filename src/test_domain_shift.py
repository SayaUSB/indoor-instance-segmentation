"""Regresi scope evaluasi: keyboard tidak memengaruhi AP laptop/pillow."""
from experiment import COMMON,score_predictions
from prepare_data import DATA,HERE,read,dump
import numpy as np
from pycocotools import mask as maskutil

assert COMMON==[30,40]
personal=read(DATA/'personal.json')
assert len(personal['images'])==64 and len(personal['annotations'])==64
assert {a['category_id'] for a in personal['annotations']}=={30,40}
assert len(read(DATA/'personal_all_classes.json')['images'])==113
assert len(read(DATA/'categories.json'))==74
assert {c['name'] for c in read(DATA/'train.json')['categories']} >= {'keyboard','laptop','pillow'}

mask=np.zeros((16,16),dtype=np.uint8);mask[2:10,2:10]=1
rle=maskutil.encode(np.asfortranarray(mask));rle['counts']=rle['counts'].decode()
categories=[{'id':c,'name':str(c)} for c in [28,30,40]]
gt={'images':[{'id':1,'width':16,'height':16,'file_name':'synthetic.png'}],'categories':categories,'annotations':[{'id':i+1,'image_id':1,'category_id':c,'segmentation':rle,'area':64,'bbox':[2,2,8,8],'iscrowd':0} for i,c in enumerate([28,30,40])]}
path=HERE/'domain_scope_selfcheck_gt.json';dump(path,gt)
predictions=[{'image_id':1,'category_id':c,'segmentation':rle,'score':1.} for c in [30,40]]
matched=score_predictions(path,predictions,COMMON)
old=score_predictions(path,predictions,[28,30,40])
assert matched['AP']>.999 and .66<old['AP']<.67
assert set(matched['per_class_AP'])=={'laptop','pillow'}
print('DOMAIN SHIFT SCOPE CHECK PASS: 64 photos; two evaluation classes; all 74 training classes preserved')
