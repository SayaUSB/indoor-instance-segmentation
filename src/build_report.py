"""Bangun tabel hasil aktual dan bukti lima kegagalan sesudah seluruh run selesai."""
import runtime_setup
from pathlib import Path
import csv,json,collections
import numpy as np
from PIL import Image,ImageDraw
from pycocotools import mask as maskutil
from prepare_data import HERE,DATA,read,dump,mask_for

def rle(ann,h,w):
    value=maskutil.encode(np.asfortranarray(mask_for(ann,h,w).astype(np.uint8)))
    value['counts']=value['counts'].decode('ascii');return value

def iou(a,b): return float(maskutil.iou([a],[b],[0])[0,0])

def best_match(gt,preds):
    matches=[p for p in preds if p['category_id']==gt['category_id'] and p['score']>=.25]
    return max([(iou(p['segmentation'],gt['segmentation']),p) for p in matches],key=lambda x:x[0],default=(0.,None))

def panel(image,masks,title):
    image=image.copy().convert('RGB'); image.thumbnail((600,600));arr=np.array(image,dtype=np.float32)
    palette=[(0,200,70),(230,60,40),(40,120,240),(255,170,0)]
    for i,mask in enumerate(masks):
        m=Image.fromarray(mask.astype(np.uint8)*255).resize(image.size,Image.Resampling.NEAREST)
        selected=np.asarray(m)>0;arr[selected]=arr[selected]*.55+np.array(palette[i%4])*.45
    out=Image.new('RGB',(600,650),'white');out.paste(Image.fromarray(arr.astype(np.uint8)),(0,40));ImageDraw.Draw(out).text((10,10),title,fill='black');return out

def failures():
    from experiment import model_load,prediction
    base=HERE/'runs/yolov8n-seg_640'
    if not (base/'DONE.json').exists():return []
    j=read(DATA/'personal.json');by=collections.defaultdict(list)
    for p in read(base/'personal_predictions.json'):
        if p['score']>=.25:by[p['image_id']].append(p)
    gts=collections.defaultdict(list)
    for a in j['annotations']:
        im=next(x for x in j['images'] if x['id']==a['image_id'])
        a['segmentation']=rle(a,im['height'],im['width']);gts[a['image_id']].append(a)
    ranked=[]
    for im in j['images']:
        scores=[(best_match(a,by[im['id']])[0],a) for a in gts[im['id']]]
        if scores:
            score,a=min(scores,key=lambda x:x[0])
            if score<.5:ranked.append((score,im,a))
    selected=sorted(ranked,key=lambda x:(x[0],x[1]['id']))[:5]
    target=HERE/'failure_analysis';target.mkdir(exist_ok=True)
    model=model_load('yolov8n-seg',640,base/'weights/best.pt');rows=[]
    for index,(score,im,a) in enumerate(selected,1):
        ps=by[im['id']];best,p=best_match(a,ps)
        if p is None:
            overlap=max([iou(x['segmentation'],a['segmentation']) for x in ps],default=0.)
            kind='salah kelas' if overlap>=.5 else 'tidak ada prediksi kelas benar pada skor >=0.25'
        else:kind='mask tidak cocok (IoU < 0.5)'
        measurements={}
        for size in (320,960):
            predictions,_=prediction(model,'yolov8n-seg',size,im)
            measurements[size]=best_match(a,predictions)[0]
        with Image.open(im['file_name']) as pil:
            gt_masks=[maskutil.decode(x['segmentation']) for x in gts[im['id']]]
            pred_masks=[maskutil.decode(x['segmentation']) for x in ps]
            parts=[panel(pil,[],'Original'),panel(pil,gt_masks,'Ground truth'),panel(pil,pred_masks,'Prediction, confidence >= 0.25')]
        canvas=Image.new('RGB',(1800,650),'white')
        for k,part in enumerate(parts):canvas.paste(part,(600*k,0))
        canvas.save(target/f'case_{index}.jpg',quality=95)
        # Restrict the mechanism claim to what the controlled intervention measures.
        delta=measurements[960]-measurements[320]
        finding='IoU meningkat dengan input lebih besar pada kasus ini' if delta>.05 else ('IoU menurun dengan input lebih besar pada kasus ini' if delta<-.05 else 'Tidak ada peningkatan IoU > 0.05 ketika resolusi dinaikkan')
        rows.append({'case':index,'file':Path(im['file_name']).name,'source':im.get('source_name'),'category_id':a['category_id'],'failure':kind,'IoU_640':score,'IoU_320_same_checkpoint':measurements[320],'IoU_960_same_checkpoint':measurements[960],'finding':finding})
    dump(target/'measurements.json',rows)
    if rows:
        with (target/'measurements.csv').open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return rows

def main():
    from experiment import score_predictions, COMMON
    personal_ids={im['id'] for im in read(DATA/'personal.json')['images']}
    rows=[]
    for d in sorted((HERE/'runs').iterdir()):
        if not (d/'DONE.json').exists():continue
        cfg=read(d/'config.json');a=read(d/'test_metrics.json');b=read(d/'personal_metrics.json');t=read(d/'training_time.json')
        # Rescore saved masks so an already-running training process adopts the new scope.
        original_predictions=read(d/'test_predictions.json')
        personal_predictions=[p for p in read(d/'personal_predictions.json') if p['image_id'] in personal_ids]
        a.pop('common_3',None)
        a['common_2']=score_predictions(DATA/'test.json',original_predictions,COMMON)
        b['all_annotated']=score_predictions(DATA/'personal.json',personal_predictions,COMMON)
        if b['images']!=len(personal_ids):
            b['latency_ms']=None
            b['latency_scope']='Previous timing covered a different image subset; omitted for the 64-photo evaluation.'
        b['images']=len(personal_ids)
        a['domain_shift_categories']=b['domain_shift_categories']=['laptop','pillow']
        dump(d/'test_metrics.json',a);dump(d/'personal_metrics.json',b)
        dump(d/'personal_predictions.json',personal_predictions)
        common_a=a['common_2']['AP'];common_b=b['all_annotated']['AP']
        rows.append({'model':cfg['model'],'input_size':cfg['input_size'],'original_all_AP':a['all_annotated']['AP'],'original_shared_AP':common_a,'personal_shared_AP':common_b,'domain_shift_AP_points':100*(common_b-common_a),'elapsed_minutes':t['elapsed_seconds']/60,'peak_vram_GiB':t['peak_allocated_bytes']/1024**3,'latency_original_ms':a['latency_ms'],'latency_personal_ms':b['latency_ms']})
    with (HERE/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    cases=failures()
    audit=read(HERE/'audit_coco.json')
    lines=['# Evaluasi CPMK 3 — NRP 5054241001','','## Data dan metode','',f"Data latih asli: {audit['train']['images']} gambar; validasi: {audit['valid']['images']}; test asli: {audit['test']['images']}; test mandiri: {audit['personal']['images']} foto tanpa augmentasi.",'','YOLOv8n-seg dan YOLOv8s-seg adalah one-stage CNN; Mask R-CNN ResNet50-FPN adalah two-stage CNN. Setiap model dilatih terpisah pada 320, 640, dan 960. Test mandiri tidak pernah dipakai melatih atau memilih checkpoint.','','Input kedua keluarga menggunakan letterbox persegi yang sama, horizontal flip p=0.5 saat training, seed 42 dan effective batch 4. Resep optimizer dan mekanisme arsitektur berbeda; hasil merupakan perbandingan model dengan resep yang dinyatakan. Konfigurasi lengkap dan versi ada pada masing-masing run serta environment.json.','','YOLO memilih checkpoint menggunakan fitness gabungan box dan mask validasi bawaan; Mask R-CNN menggunakan COCO mask AP setiap lima epoch dan epoch terakhir. Frekuensi/rumus seleksi checkpoint ini tidak identik dan merupakan keterbatasan perbandingan.','','Evaluasi akhir memakai pycocotools COCOeval segm untuk kedua keluarga. Prediksi dikembalikan ke koordinat gambar ekspor sebelum dievaluasi. Nilai AP di tabel dikali 100. Domain shift adalah AP mandiri dikurangi AP asli, pada dua kelas bersama: laptop dan pillow.','','## Hasil aktual','','| Model | Input | AP asli semua | AP asli 2 kelas | AP mandiri 2 kelas | Delta poin AP | Menit run | Peak VRAM GiB |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['model']} | {r['input_size']} | {r['original_all_AP']*100:.2f} | {r['original_shared_AP']*100:.2f} | {r['personal_shared_AP']*100:.2f} | {r['domain_shift_AP_points']:+.2f} | {r['elapsed_minutes']:.1f} | {r['peak_vram_GiB']:.2f} |")
    lines+=['','## Pengaruh ukuran input','']
    for model in sorted({r['model'] for r in rows}):
        group=sorted([r for r in rows if r['model']==model],key=lambda r:r['input_size'])
        if len(group)==3:
            change=100*(group[-1]['personal_shared_AP']-group[0]['personal_shared_AP'])
            lines.append(f"- {model}: perubahan AP mandiri 320 ke 960 = {change:+.2f} poin. AP, biaya waktu, dan VRAM harus dipertimbangkan bersama; satu seed tidak mengukur variansi antar training.")
    lines+=['','## Keterbatasan data yang memengaruhi interpretasi','','- Keyboard dikeluarkan dari evaluasi domain shift karena tidak mempunyai contoh train. Training dan evaluasi semua kelas pada test asli tetap mempertahankan 74 kelas. Kelas tanpa contoh train: '+', '.join(audit['classes_without_train_examples'])+'.','- Arsip anotasi mandiri memiliki tiga kelas, tetapi evaluasi ini hanya menggunakan 64 foto laptop/pillow. Evaluasi domain shift memakai laptop dan pillow saja; kelengkapan anotasi kedua kelas tetap diperlukan.','- Foto mandiri menunjukkan objek yang sama dari banyak sudut pada lingkungan terbatas. Banyak foto berbeda tidak sama dengan keragaman 64 ruangan independen.','- Ekspor mandiri sudah distretch 960x960. Ini menambah perubahan geometri dan dapat menjadi salah satu sumber domain shift.','- Empat gambar asli train/valid diputar 90 derajat searah jarum jam pada salinan kerja setelah overlay diperiksa. Tidak ada gambar test asli yang diubah orientasinya.','- Area ground truth dihitung ulang dari mask; nilai area ekspor sebagian merupakan luas bbox.','- Satu anotasi validasi dan tiga anotasi test yang identik persis dideduplikasi pada COCO kerja; ID dicatat di exact_duplicate_annotations.json. File sumber tetap utuh.','- Instance yang hilang saat rasterisasi pada resolusi rendah dicatat dalam prepared/<size>/*_subpixel_instances.json dan tidak dipakai sebagai target training pada kedua model. Ground truth evaluasi tetap pada resolusi asal.','','## Analisis lima kegagalan dan eksperimen kecil','','Baseline kasus dipilih lebih dahulu: YOLOv8n-seg checkpoint 640. Pilihan kasus sesudah inferensi berdasarkan IoU terendah pada lima foto berbeda; bukan sampel acak. Visualisasi memakai confidence 0.25, sedangkan evaluasi AP memakai 0.001.','','Hipotesis: hilangnya detail karena sampling input/prototype mask berkontribusi pada kegagalan YOLO. Intervensi: checkpoint yang sama diinferensi pada 320 dan 960, semua threshold tetap. Ini menguji sensitivitas resolusi input, tidak mengisolasi penyebab internal prototype mask dari backbone, confidence atau NMS.','','| Kasus | Kelas | Jenis kegagalan | IoU 640 | IoU 320 | IoU 960 | Temuan |','|---|---|---|---:|---:|---:|---|']
    names={c['id']:c['name'] for c in read(DATA/'categories.json')}
    for r in cases:lines.append(f"| {r['case']} | {names[r['category_id']]} | {r['failure']} | {r['IoU_640']:.3f} | {r['IoU_320_same_checkpoint']:.3f} | {r['IoU_960_same_checkpoint']:.3f} | {r['finding']} |")
    for r in cases:
        mechanism='Kelas ini tidak memiliki contoh train; head klasifikasi/mask tidak memperoleh supervisi kelas keyboard pada fine-tuning. Ini merupakan penjelasan berbasis audit data yang perlu dibedakan dari resolusi.' if r['category_id']==28 else 'Hipotesis arsitektur: sampling fitur backbone dan mask prototype YOLO dapat mengurangi ketelitian batas pada sudut miring. Hasil intervensi resolusi di tabel menentukan apakah ada dukungan pada foto ini; penyebab spesifik belum terisolasi.'
        lines+=['',f"### Kasus {r['case']}: {r['file']}",'',mechanism,'',f"![Original, ground truth, prediksi](failure_analysis/case_{r['case']}.jpg)"]
    if len(cases)<5:lines+=['',f'Hanya {len(cases)} foto memenuhi definisi kegagalan IoU <0.5; tidak menambahkan kasus fiktif.']
    lines+=['','## Bukti proses','','Log training asli: runs/*/results.csv dan training.log. Checkpoint best/last disimpan per run. Waktu dan peak CUDA allocated memory berasal dari training_time.json; ini tidak sama dengan seluruh konsumsi VRAM proses. Screenshot Bukti Anotasi.png dan metadata_foto_asli.csv melengkapi bukti pengambilan/anotasi. Merek/model HP berdasarkan keterangan pengguna; tag model kamera tidak tersedia pada foto yang diperiksa.','','## Referensi','','- https://docs.ultralytics.com/tasks/segment/','- https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html','- https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools']
    (HERE/'LAPORAN_HASIL.md').write_text('\n'.join(lines),encoding='utf8')
    dump(HERE/'STATUS.json',{'state':'COMPLETE','runs':len(rows),'failure_cases':len(cases),'report':'LAPORAN_HASIL.md'})
    print('REPORT READY',HERE/'LAPORAN_HASIL.md',flush=True)

if __name__=='__main__':main()
