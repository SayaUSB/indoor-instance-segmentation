# Logbook aktual

- 17 November 2026: membaca instruksi tugas; NRP berakhir 1 -> ukuran input 320/640/960.
- Membaca dataset lokal, README, YAML, seluruh label; menemukan ekspor OBB pada kedua dataset.
- Memeriksa screenshot: anotasi poligon terlihat pada editor Roboflow.
- Menghitung 113 foto asli dan 273 ekspor mandiri; 80 sumber train menjadi 240 gambar augmentasi.
- Mengekstrak metadata dan SHA-256 113 JPEG; waktu pengambilan tersedia, model kamera tidak ditemukan pada tag standar yang diperiksa.
- Windows mendeteksi RTX 4060 Laptop; nvidia-smi belum berhasil dipanggil, CUDA belum diperiksa. Python 3.14.8 yang tersedia belum berisi paket training.
- Keputusan: mempertahankan dataset asli; tidak melatih mask kotak dari OBB; menunggu ekspor segmentasi tanpa augmentasi sebelum training.
- Belum ada training, evaluasi, angka performa, waktu GPU, atau 5 kegagalan model yang dapat dilaporkan.

- Ekspor COCO terbaru diperiksa: 11.845 instance asli dan 139 instance mandiri; seluruh sumber mandiri digabung sebagai test eksternal.
- Python 3.12 lain ditemukan dengan PyTorch 2.7.1+cu118, torchvision 0.22.1+cu118, Ultralytics 8.4.183. CUDA benar-benar berhasil menghitung tensor pada RTX 4060 Laptop 8 GB.
- Membuat .venv lokal dengan paket sistem yang sudah ada; memasang pycocotools, psutil, six, dan python-dateutil yang kompatibel secara lokal. NVML nvidia-smi tetap gagal, tetapi akses CUDA tersedia.
- Menemukan dan mengoreksi orientasi empat gambar train/valid pada salinan kerja setelah overlay ditinjau; tidak mengubah sumber.
- COCO area dihitung ulang dari mask. Satu instance subpiksel pada target training 320 dicatat dan dilewati untuk kedua keluarga.
- Evaluator COCO diuji dengan mask sintetis sempurna: AP=1; transform letterbox dan bentuk target diperiksa dengan assert.
- Menemukan keyboard tidak punya contoh train; mempertahankan split asli dan mencatat keterbatasan interpretasi.
- Menjalankan smoke test model 960 sebelum eksperimen penuh. Nilai smoke bukan hasil final.

- Menemukan satu mask validasi dan tiga mask test asli yang identik persis (image, kelas, poligon sama). Deduplikasi pada COCO kerja untuk kedua keluarga dan evaluator; berkas sumber tetap utuh. ID dicatat dalam exact_duplicate_annotations.json. Total mask kerja: train 4479, valid 2021, test 5341, mandiri 139.

- Smoke test ketiga model pada 960 berhasil. Uji tambahan memverifikasi optimizer Mask R-CNN memperbarui bobot.
- Evaluator COCO dijalankan pada seluruh 166 gambar validasi; metrik semua kelas dan 3 kelas berhasil. Memperbaiki efek mutasi internal COCO.loadRes dengan deepcopy dan menambahkan regresi dua evaluasi berulang ke self-check.
- Memulai 9 eksperimen penuh, masing-masing target 50 epoch.

- Instruksi pengguna: keluarkan keyboard dari domain shift, pertahankan semua kelas training. Membuat subset personal aktif 64 foto/64 mask: 30 laptop, 34 pillow; 49 foto keyboard-only tidak dipakai. Menyimpan arsip 113 foto dalam prepared/personal_all_classes.json.
- Mengubah kelas bersama menjadi COCO [30,40]. Pelaporan akhir menghitung ulang COCO AP dua kelas dari prediksi tersimpan, sehingga proses training aktif tidak perlu dihentikan. Analisis lima kegagalan juga memakai subset 64 foto tersebut.
