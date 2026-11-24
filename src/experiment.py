"""Tiga model, tiga resolusi, evaluator COCO mask yang sama. Semua output di folder ini."""
import runtime_setup
from pathlib import Path
import argparse, contextlib, csv, io, json, math, random, time, traceback, sys, platform, importlib.metadata, gc, copy
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms.functional import pil_to_tensor
from torchvision.models.detection import maskrcnn_resnet50_fpn,MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskutil
from ultralytics import YOLO
from prepare_data import DATA,HERE,ROOT,SIZES,read,dump,mask_for,square

torch.set_num_threads(4)
MODELS=('yolov8n-seg','yolov8s-seg','maskrcnn_resnet50_fpn')
COMMON=[30,40]
CATEGORIES=read(DATA/'categories.json') if (DATA/'categories.json').exists() else []
NAMES={c['id']:c['name'] for c in CATEGORIES}

def seed():
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False

def append_csv(path,row):
    exists=path.exists()
    with path.open('a',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        if not exists: writer.writeheader()
        writer.writerow(row)

def environment():
    return {'python':sys.version,'platform':platform.platform(),'gpu':torch.cuda.get_device_name(0),'vram_bytes':torch.cuda.get_device_properties(0).total_memory,'cuda':torch.version.cuda,'packages':{p:importlib.metadata.version(p) for p in ['torch','torchvision','ultralytics','numpy','Pillow','pycocotools']}}

class SegDataset(Dataset):
    def __init__(self,path,augment=False):
        self.j=read(path); self.images=self.j['images']; self.augment=augment
        self.by={im['id']:[] for im in self.images}
        for a in self.j['annotations']: self.by[a['image_id']].append(a)
    def __len__(self): return len(self.images)
    def __getitem__(self,index):
        im=self.images[index]; image=Image.open(im['file_name']).convert('RGB'); w,h=image.size
        anns=self.by[im['id']]
        masks=np.stack([mask_for(a,h,w) for a in anns]) if anns else np.empty((0,h,w),dtype=np.uint8)
        boxes=np.array([[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in anns],dtype=np.float32).reshape(-1,4)
        if self.augment and random.random()<.5:
            image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT); masks=masks[:,:,::-1].copy()
            boxes[:,[0,2]]=w-boxes[:,[2,0]]
        target={'boxes':torch.tensor(boxes),'labels':torch.tensor([a['category_id'] for a in anns],dtype=torch.int64),'masks':torch.tensor(masks.copy(),dtype=torch.uint8),'image_id':torch.tensor(im['id'])}
        return pil_to_tensor(image).float()/255,target

def make_maskrcnn(size,pretrained=False):
    model=maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,weights_backbone=None,min_size=size,max_size=size,box_score_thresh=.001,box_nms_thresh=.7,box_detections_per_img=100)
    model.roi_heads.box_predictor=FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features,75)
    model.roi_heads.mask_predictor=MaskRCNNPredictor(model.roi_heads.mask_predictor.conv5_mask.in_channels,256,75)
    return model.cuda()

def model_load(name,size,checkpoint):
    if name.startswith('yolo'): return YOLO(str(checkpoint))
    model=make_maskrcnn(size)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'])
    return model.eval()

def prediction(model,name,size,im):
    with Image.open(im['file_name']) as pil: canvas,tr=square(pil.convert('RGB'),size)
    sx,sy,left,top,nw,nh=tr; w,h=im['width'],im['height']
    torch.cuda.synchronize(); start=time.perf_counter()
    with torch.inference_mode():
        if name.startswith('yolo'):
            r=model.predict(np.asarray(canvas)[:,:,::-1].copy(),imgsz=size,conf=.001,iou=.7,max_det=100,retina_masks=True,device=0,verbose=False)[0]
            labels=r.boxes.cls.cpu().numpy().astype(int)+1; scores=r.boxes.conf.cpu().numpy()
            masks=r.masks.data.cpu().numpy() if r.masks is not None else np.empty((0,size,size))
        else:
            tensor=(pil_to_tensor(canvas).float()/255).cuda()
            with torch.autocast('cuda',enabled=True): r=model([tensor])[0]
            labels=r['labels'].cpu().numpy(); scores=r['scores'].cpu().numpy(); masks=r['masks'][:,0].float().cpu().numpy()
    torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    preds=[]
    for label,score,mask in zip(labels,scores,masks):
        mask=np.asarray(Image.fromarray((mask>=.5).astype(np.uint8)*255).crop((left,top,left+nw,top+nh)).resize((w,h),Image.Resampling.NEAREST))>0
        if not mask.any(): continue
        rle=maskutil.encode(np.asfortranarray(mask.astype(np.uint8)));rle['counts']=rle['counts'].decode('ascii')
        preds.append({'image_id':im['id'],'category_id':int(label),'score':float(score),'segmentation':rle})
    return preds,elapsed

def score_predictions(gt,preds,cat_ids):
    with contextlib.redirect_stdout(io.StringIO()) as capture:
        coco=COCO(str(gt))
        if preds: dt=coco.loadRes(copy.deepcopy(preds))
        else:
            dt=COCO(); dt.dataset={'images':coco.dataset['images'],'categories':coco.dataset['categories'],'annotations':[]};dt.createIndex()
        ev=COCOeval(coco,dt,'segm');ev.params.catIds=cat_ids;ev.params.maxDets=[1,10,100]
        ev.evaluate();ev.accumulate();ev.summarize()
    per_class={}
    for k,c in enumerate(cat_ids):
        values=ev.eval['precision'][:,:,k,0,-1]; values=values[values>=0]
        per_class[NAMES.get(c,str(c))]=float(values.mean()) if values.size else None
    return {'AP':float(ev.stats[0]),'AP50':float(ev.stats[1]),'AP75':float(ev.stats[2]),'per_class_AP':per_class,'evaluator':'pycocotools COCOeval segm','summary':capture.getvalue()}

def evaluate(model,name,size,split,dest=None,limit=None):
    gt=DATA/f'{split}.json'; j=read(gt); preds=[];times=[]
    images=j['images'][:limit] if limit else j['images']
    for n,im in enumerate(images):
        p,t=prediction(model,name,size,im);preds.extend(p)
        if n>0: times.append(t)  # warm first frame excluded
        if n%100==0:print(name,size,split,n,'/',len(images),flush=True)
    cats=COMMON if split=='personal' else [c['id'] for c in CATEGORIES]
    # Only full-set inference is used for AP, never smoke subsets.
    if limit: return preds
    metrics={'all_annotated':score_predictions(gt,preds,cats)}
    if split!='personal': metrics['common_2']=score_predictions(gt,preds,COMMON)
    metrics.update(images=len(images),latency_ms=float(np.mean(times)*1000) if times else None,latency_scope='square input + GPU forward/postprocess + transfer to CPU; excludes image decode, letterbox, final mask upsampling and RLE encoding; first image excluded')
    if dest:
        dump(dest/f'{split}_predictions.json',preds);dump(dest/f'{split}_metrics.json',metrics)
    return metrics

def train_yolo(name,size,epochs,dest,smoke=False):
    seed();torch.cuda.reset_peak_memory_stats()
    model=YOLO(name+'.pt')
    start=time.perf_counter()
    model.train(data=str(DATA/str(size)/'data.yaml'),epochs=epochs,imgsz=size,batch=2,nbs=4,device=0,workers=0,seed=42,deterministic=True,optimizer='SGD',lr0=.01,lrf=.01,momentum=.937,weight_decay=.0005,warmup_epochs=0,patience=0,amp=True,project=str(dest.parent),name=dest.name,exist_ok=True,plots=False,save=True,save_period=-1,val=True,verbose=False,cache=False,rect=False,mask_ratio=4,overlap_mask=False,mosaic=0,close_mosaic=0,mixup=0,copy_paste=0,hsv_h=0,hsv_s=0,hsv_v=0,degrees=0,translate=0,scale=0,shear=0,perspective=0,flipud=0,fliplr=.5,fraction=.02 if smoke else 1.)
    torch.cuda.synchronize()
    dump(dest/'training_time.json',{'elapsed_seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'epochs':epochs,'checkpoint_selection':'Ultralytics built-in combined box and mask validation fitness; evaluated every epoch','environment':environment()})
    return dest/'weights/best.pt'

def train_mask(size,epochs,dest,smoke=False):
    seed();torch.cuda.reset_peak_memory_stats()
    model=make_maskrcnn(size,True)
    ds=SegDataset(DATA/str(size)/'train.json',augment=True)
    loader=DataLoader(ds,batch_size=1,shuffle=True,num_workers=0,collate_fn=lambda b:tuple(zip(*b)))
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.005,momentum=.9,weight_decay=.0005)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[max(1,round(epochs*.6)),max(2,round(epochs*.8))],gamma=.1)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024);best=-1.;start=time.perf_counter();train_seconds=0
    for epoch in range(1,epochs+1):
        model.train();before=model.roi_heads.box_predictor.cls_score.weight.detach().clone() if smoke else None;loss_sum=0.;steps=0;optimizer.zero_grad();torch.cuda.synchronize();t0=time.perf_counter()
        total=min(2,len(loader)) if smoke else len(loader)
        for i,(images,targets) in enumerate(loader):
            if i>=total:break
            images=[x.cuda() for x in images];targets=[{k:v.cuda() for k,v in t.items()} for t in targets]
            # Final incomplete accumulation group uses its actual size.
            group_size=min(4,total-(i//4)*4)
            with torch.autocast('cuda'):
                losses=model(images,targets);loss=sum(losses.values());scaled=loss/group_size
            assert torch.isfinite(loss).item(),f'nonfinite loss: {losses}'
            scaler.scale(scaled).backward()
            if (i+1)%4==0 or i+1==total:
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),10)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad()
            loss_sum+=float(loss.detach());steps+=1
        torch.cuda.synchronize();train_seconds+=time.perf_counter()-t0
        if smoke: assert not torch.equal(before,model.roi_heads.box_predictor.cls_score.weight.detach()),'Smoke optimizer did not update weights'
        ap=None
        if epoch%5==0 or epoch==epochs:
            model.eval()
            if not smoke:
                met=evaluate(model,'maskrcnn_resnet50_fpn',size,'valid');ap=met['all_annotated']['AP']
            else:ap=0.
            if ap>best:
                best=ap;torch.save({'model':model.state_dict(),'epoch':epoch,'val_mask_AP':ap,'size':size},dest/'best.pt')
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'size':size},dest/'last.pt')
        append_csv(dest/'results.csv',{'epoch':epoch,'loss':loss_sum/steps,'val_mask_AP':ap,'lr':optimizer.param_groups[0]['lr'],'train_seconds_cumulative':train_seconds})
        scheduler.step();print('Mask R-CNN',size,'epoch',epoch,'loss',loss_sum/steps,'valAP',ap,flush=True)
    dump(dest/'training_time.json',{'elapsed_seconds':time.perf_counter()-start,'train_loop_seconds':train_seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'epochs':epochs,'checkpoint_selection':'COCO mask AP on original validation; every five epochs and final epoch','environment':environment()})
    return dest/'best.pt'

def self_check():
    # Perfect synthetic masks must score AP=1.0; validates the shared evaluator.
    mask=np.zeros((16,16),np.uint8);mask[2:9,3:11]=1
    rle=maskutil.encode(np.asfortranarray(mask));rle['counts']=rle['counts'].decode()
    j={'images':[{'id':1,'width':16,'height':16,'file_name':'synthetic.png'}],'categories':[{'id':1,'name':'synthetic'}],'annotations':[{'id':1,'image_id':1,'category_id':1,'area':56,'bbox':[3,2,8,7],'iscrowd':0,'segmentation':rle}]}
    dump(HERE/'selfcheck_gt.json',j)
    preds=[{'image_id':1,'category_id':1,'score':1.,'segmentation':rle}]
    for _ in range(2):
        m=score_predictions(HERE/'selfcheck_gt.json',preds,[1])
        assert m['AP']>.999 and 'bbox' not in preds[0],m
    image=Image.new('RGB',(640,480));canvas,tr=square(image,320)
    assert canvas.size==(320,320) and tr==(0.5,0.5,0,40,320,240)
    ds=SegDataset(DATA/'320/train.json');x,t=ds[0]
    assert x.shape==(3,320,320) and t['masks'].shape[1:]==(320,320) and len(t['labels'])==len(t['boxes'])
    print('SELF CHECK PASS',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--epochs',type=int,default=50);parser.add_argument('--models',nargs='+',choices=MODELS,default=MODELS);parser.add_argument('--sizes',nargs='+',type=int,choices=SIZES,default=SIZES);parser.add_argument('--self-check',action='store_true');args=parser.parse_args()
    assert torch.cuda.is_available(),'CUDA required for full experiment; CPU fallback disabled to avoid accidental long runs'
    self_check()
    if args.self_check:return
    dump(HERE/'environment.json',environment())
    runroot=HERE/('smoke_runs' if args.smoke else 'runs');runroot.mkdir(exist_ok=True)
    for size in args.sizes:
        for name in args.models:
            dest=runroot/f'{name}_{size}';dest.mkdir(exist_ok=True)
            if (dest/'DONE.json').exists():print('SKIP completed',dest.name,flush=True);continue
            cfg={'model':name,'input_size':size,'epochs':1 if args.smoke else args.epochs,'seed':42,'microbatch':2 if name.startswith('yolo') else 1,'effective_batch':4,'train':'original train only','personal':'64 external test images, laptop/pillow only','domain_shift_categories':[30,40],'shared_augmentation':'horizontal flip p=0.5 only','mask_threshold':.5,'confidence_threshold':.001,'max_detections':100,'nms_iou':.7}
            dump(dest/'config.json',cfg)
            dump(HERE/'STATUS.json',{'state':'RUNNING','current':dest.name,'smoke':args.smoke,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
            try:
                if (dest/'TRAINED.json').exists():
                    checkpoint=Path(read(dest/'TRAINED.json')['checkpoint'])
                else:
                    checkpoint=train_yolo(name,size,cfg['epochs'],dest,args.smoke) if name.startswith('yolo') else train_mask(size,cfg['epochs'],dest,args.smoke)
                    dump(dest/'TRAINED.json',{'checkpoint':str(checkpoint)})
                gc.collect();torch.cuda.empty_cache();model=model_load(name,size,checkpoint)
                if args.smoke:
                    evaluate(model,name,size,'personal',limit=1)
                else:
                    evaluate(model,name,size,'test',dest);evaluate(model,name,size,'personal',dest)
                dump(dest/'DONE.json',{'checkpoint':str(checkpoint),'smoke':args.smoke,'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
                del model;gc.collect();torch.cuda.empty_cache()
            except Exception as e:
                dump(dest/'ERROR.json',{'error':str(e),'traceback':traceback.format_exc()})
                dump(HERE/'STATUS.json',{'state':'FAILED','current':dest.name,'error':str(e)})
                raise
    dump(HERE/'STATUS.json',{'state':'SMOKE_PASSED' if args.smoke else 'TRAINING_AND_EVALUATION_FINISHED'})
    if not args.smoke:
        import build_report
        build_report.main()

if __name__=='__main__': main()
