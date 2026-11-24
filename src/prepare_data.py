"""Validasi COCO, remap kategori berdasarkan nama, dan buat input persegi bersama."""
import runtime_setup
from pathlib import Path
import json, hashlib, collections, shutil
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as maskutil
PROJECT = Path(__file__).resolve().parents[1]
HERE = PROJECT / 'work'
HERE.mkdir(exist_ok=True)
ROOT = PROJECT / 'data'
DATA=HERE/'prepared'
SIZES=(320,640,960)

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2),encoding='utf8')

def read(path): return json.loads(path.read_text(encoding='utf8'))

def mask_for(ann,h,w):
    seg=ann['segmentation']
    if isinstance(seg,list): return maskutil.decode(maskutil.merge(maskutil.frPyObjects(seg,h,w)))
    if isinstance(seg.get('counts'),list): seg=maskutil.frPyObjects(seg,h,w)
    return maskutil.decode(seg)

def square(image,size):
    w,h=image.size
    nw,nh=round(w*size/max(w,h)),round(h*size/max(w,h))
    left,top=(size-nw)//2,(size-nh)//2
    canvas=Image.new('RGB',(size,size),(114,114,114))
    canvas.paste(image.resize((nw,nh),Image.Resampling.BILINEAR),(left,top))
    return canvas,(nw/w,nh/h,left,top,nw,nh)

def normalize(folder,categories):
    src=folder/'_annotations.coco.json'; j=read(src)
    names={c['id']:c['name'] for c in j['categories']}
    ids={c['name']:c['id'] for c in categories}
    images={i['id']:i for i in j['images']}
    assert len(images)==len(j['images']),f'duplicate image ids: {src}'
    assert len({a['id'] for a in j['annotations']})==len(j['annotations'])
    anns=[]; seen=set(); duplicate_ids=[]
    for im in images.values():
        path=folder/im['file_name']; assert path.is_file(),path
        with Image.open(path) as pil:
            if pil.size!=(im['width'],im['height']):
                reviewed=('182169107_','182169121_','182169109_','182169108_')
                assert path.name.startswith(reviewed),(path,pil.size,im)
                fixed=DATA/'orientation_corrected'/path.name
                fixed.parent.mkdir(parents=True,exist_ok=True)
                rotated=pil.transpose(Image.Transpose.ROTATE_270)
                assert rotated.size==(im['width'],im['height'])
                rotated.convert('RGB').save(fixed,quality=100,subsampling=0)
                path=fixed
        im['file_name']=str(path.resolve())
    for a in j['annotations']:
        assert a['image_id'] in images and names[a['category_id']] in ids,a
        assert not a.get('iscrowd',0),'crowd annotations need an explicit policy'
        key=json.dumps([a['image_id'],a['category_id'],a['segmentation']])
        if key in seen:
            duplicate_ids.append(a['id']); continue
        seen.add(key)
        a['category_id']=ids[names[a['category_id']]]
        im=images[a['image_id']]; h,w=im['height'],im['width']
        for poly in a['segmentation']:
            assert len(poly)>=6 and len(poly)%2==0 and np.isfinite(poly).all(),a['id']
        m=mask_for(a,h,w)
        assert m.any(),f'empty raster mask: {src} {a["id"]}'
        a['area']=int(m.sum())  # COCO export area was often bbox area, not mask area.
        ys,xs=np.where(m); a['bbox']=[int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)]
        a['iscrowd']=0; anns.append(a)
    return {'info':{'source':str(src),'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'removed_exact_duplicate_annotation_ids':duplicate_ids},'images':list(images.values()),'annotations':anns,'categories':categories}

def main():
    source=read(ROOT/'train/_annotations.coco.json')
    categories=[c for c in source['categories'] if c['id']!=0]
    assert [c['id'] for c in categories]==list(range(1,75))
    DATA.mkdir(exist_ok=True)
    datasets={}
    for split in ('train','valid','test'):
        datasets[split]=normalize(ROOT/split,categories)
        dump(DATA/f'{split}.json',datasets[split])
    personal={'info':{'source':'DatasetMandiriSeg, all splits pooled ONLY for external testing'},'categories':categories,'images':[],'annotations':[]}
    sources=set()
    for split in ('train','valid','test'):
        j=normalize(ROOT/'DatasetMandiriSeg'/split,categories)
        image_map={}
        for im in j['images']:
            source_name=Path(im['file_name']).name.split('_jpg.rf.')[0]
            assert source_name not in sources,'duplicate/augmented personal source'
            sources.add(source_name)
            image_map[im['id']]=len(personal['images'])+1
            im['id']=image_map[im['id']]; im['source_split']=split; im['source_name']=source_name
            personal['images'].append(im)
        for a in j['annotations']:
            a['id']=len(personal['annotations'])+1; a['image_id']=image_map[a['image_id']]; personal['annotations'].append(a)
    assert len(sources)>=40
    dump(DATA/'personal_all_classes.json',personal)
    personal['annotations']=[a for a in personal['annotations'] if a['category_id'] in (30,40)]
    selected_ids={a['image_id'] for a in personal['annotations']}
    personal['images']=[im for im in personal['images'] if im['id'] in selected_ids]
    personal['info']['evaluation_scope']='laptop and pillow only; keyboard excluded from domain shift'
    personal['info']['domain_shift_category_ids']=[30,40]
    assert len(personal['images'])>=40
    dump(DATA/'personal.json',personal)
    datasets['personal']=personal
    stats={key:{'images':len(j['images']),'instances':len(j['annotations']),'class_instances':dict(collections.Counter(a['category_id'] for a in j['annotations']))} for key,j in datasets.items()}
    train_ids=set(a['category_id'] for a in datasets['train']['annotations'])
    stats['classes_without_train_examples']=[c['name'] for c in categories if c['id'] not in train_ids]
    stats['personal_annotation_scope']=['laptop','pillow']
    stats['warnings']=['Keyboard has no annotated training examples in the supplied original train split.','Personal images were stretched to 960x960 at export.','Personal annotation completeness for the other 71 classes is not verified.','Ten difficult captures still require visual review.']
    dump(HERE/'audit_coco.json',stats)
    dump(DATA/'categories.json',categories)
    # One shared, precomputed image/polygon transform for both model families.
    for size in SIZES:
        for split in ('train','valid'):
            j=read(DATA/f'{split}.json'); grouped=collections.defaultdict(list); kept=[]; vanished=[]
            for a in j['annotations']: grouped[a['image_id']].append(a)
            target=DATA/str(size)
            for im in j['images']:
                dest=target/'images'/split/Path(im['file_name']).name
                dest.parent.mkdir(parents=True,exist_ok=True)
                with Image.open(im['file_name']) as pil: canvas,tr=square(pil.convert('RGB'),size)
                canvas.save(dest,quality=95,subsampling=0)
                sx,sy,left,top,_,_=tr; lines=[]
                for a in grouped[im['id']]:
                    assert len(a['segmentation'])==1,'multipart masks need explicit YOLO bridging'
                    coords=np.asarray(a['segmentation'][0]).reshape(-1,2)
                    coords[:,0]=coords[:,0]*sx+left; coords[:,1]=coords[:,1]*sy+top
                    coords=np.clip(coords,0,size)
                    a['segmentation']=[coords.reshape(-1).tolist()]
                    m=mask_for(a,size,size)
                    if not m.any():
                        vanished.append({'image_id':im['id'],'annotation_id':a['id'],'category_id':a['category_id']}); continue
                    ys,xs=np.where(m); kept.append(a)
                    a['area']=int(m.sum()); a['bbox']=[int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)]
                    lines.append(str(a['category_id']-1)+' '+' '.join(f'{v/size:.8f}' for v in coords.reshape(-1)))
                label=target/'labels'/split/(dest.stem+'.txt'); label.parent.mkdir(parents=True,exist_ok=True)
                label.write_text('\n'.join(lines),encoding='utf8')
                im.update(file_name=str(dest),height=size,width=size)
            j['annotations']=kept
            dump(target/f'{split}.json',j)
            dump(target/f'{split}_subpixel_instances.json',vanished)
        # JSON is valid YAML and preserves exact names without hand-built escaping.
        dump(DATA/str(size)/'data.yaml',{'path':str(DATA/str(size)),'train':'images/train','val':'images/valid','names':[c['name'] for c in categories]})
        print('prepared input size',size,flush=True)
    print(json.dumps(stats,indent=2))

if __name__=='__main__': main()
