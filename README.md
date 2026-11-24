# Segmentasi instance pada foto indoor

Ustu Bina Syahdiba · 5054241001 · Computer Vision, Week 11

Saya membandingkan YOLOv8n-seg, YOLOv8s-seg, dan Mask R-CNN pada ukuran input 320, 640, dan 960 piksel. Ketiganya dilatih menggunakan 397 gambar dari Panoptic Indoor Segmentation v5. Untuk melihat perubahan performa pada foto HP, saya membandingkan AP kelas laptop dan pillow pada test set asli dan 64 foto mandiri.

Mask R-CNN pada input 640 mendapat mask AP tertinggi pada foto mandiri, **32,99%**. Pada dua kelas yang sama di test set asli, skornya **12,10%**. Beberapa foto mandiri memperlihatkan objek yang sama dari sudut berbeda, sehingga hasil ini belum menggambarkan performa di banyak ruangan.

## Hasil dan bukti

- [Notebook analisis](analysis.ipynb) memuat tabel sembilan eksperimen, grafik loss, dan lima contoh kegagalan. Grafik tersimpan dan bisa dibaca langsung di GitHub.
- [Laporan PDF](docs/Laporan_Hasil_5054241001.pdf) dan [versi Word](docs/Laporan_Hasil_5054241001.docx).
- [Perbandingan model](results/comparison.csv) dan [log per run](results/runs), termasuk loss per epoch, metrik COCO, waktu, dan penggunaan memori GPU.
- [Bukti anotasi Roboflow](evidence/roboflow.png), [metadata foto](evidence/metadata_foto_asli.csv), dan [10 foto dengan sudut sulit](evidence/kondisi_sulit.csv).
- [Logbook](docs/logbook.md) dan [penggunaan AI](docs/penggunaan-ai.md).

## Menjalankan proyek

Untuk menjalankan ulang **analisis hasil**, pasang pandas, matplotlib, dan JupyterLab. Buka `analysis.ipynb` dari direktori utama. Notebook membaca tabel dan gambar yang sudah tersimpan.

Untuk **mengulang training**, gunakan Python 3.12 dan GPU CUDA. Versi awal memakai PyTorch 2.7.1 dan torchvision 0.22.1 dengan CUDA 11.8. Pasang kedua paket tersebut dengan build CUDA yang sesuai, lalu jalankan `python -m pip install -r requirements.txt`.

Simpan ekspor COCO [Panoptic Indoor Segmentation v5](https://universe.roboflow.com/panopticindoor/panoptic-indoor-segmentation/dataset/5) di `data/train`, `data/valid`, dan `data/test`. Salin ekspor mandiri dari folder lokal `DatasetMandiriSeg` ke `data/DatasetMandiriSeg` dengan struktur split aslinya. Setiap split berisi gambar dan `_annotations.coco.json`. Foto sumber dan checkpoint disimpan secara lokal.

Jalankan dari direktori utama:

```sh
python src/prepare_data.py
python src/experiment.py --self-check
python src/test_domain_shift.py
python src/experiment.py --epochs 50
```

Hasil run baru ditulis ke `work/`. Angka dari eksperimen yang dilaporkan tersimpan di `results/`.
