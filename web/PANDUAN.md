# Web lokal segmentasi indoor

Ekstrak isi ZIP ke folder repo. Struktur utamanya adalah web/, results/, dan requirements.txt. Jalankan dari folder yang memuat ketiganya. Web juga bisa dijalankan dari hasil ekstraksi terpisah.

Gunakan environment Python 3.12 yang dipakai untuk training. Jika membuat environment baru, pasang PyTorch dan Torchvision sesuai perangkat, lalu jalankan `python -m pip install -r requirements.txt`.

~~~powershell
python web/server.py
~~~

Jika Python bawaan memunculkan `No module named 'PIL'`, buat environment Python 3.12 dengan uv dari PowerShell. Langkah pemasangan memerlukan internet.

~~~powershell
uv python install 3.12
uv venv --python 3.12 .venv-web
uv pip install --python .venv-web/Scripts/python.exe -r requirements.txt
.\.venv-web\Scripts\python.exe web/server.py
~~~

Halaman dan statistik bisa dibuka sebelum paket inferensi terpasang. Untuk memprediksi gambar, jalankan server dengan environment di atas.

Buka http://127.0.0.1:8765 di browser. Server hanya mendengarkan koneksi dari komputer sendiri. Foto dan frame webcam diproses di memori. Browser perlu izin kamera saat tombol webcam ditekan.

Server mencari bobot eksperimen dalam work/runs atau folder Week11/Submission_5054241001/runs di komputer proyek. Lokasi lain bisa diberikan lewat argumen:

~~~powershell
python web/server.py --runs "E:\Kuliah\Semester5\CV\Week11\Submission_5054241001\runs"
~~~

Untuk YOLO, struktur checkpoint adalah `MODEL_UKURAN/weights/best.pt`. Untuk Mask R-CNN, gunakan `maskrcnn_resnet50_fpn_UKURAN/best.pt`. Paket unduhan berisi kode dan statistik, sementara sembilan checkpoint dibaca dari folder training.

Pilih satu atau beberapa model, ukuran 320/640/960, lalu unggah foto atau nyalakan webcam. Klik **Jalankan perbandingan** untuk memproses satu gambar. Centang **Inferensi webcam berulang** untuk terus memproses frame baru. Setiap putaran memakai frame yang sama untuk semua model terpilih. Gunakan **Bekukan frame** untuk mengganti threshold pada gambar yang sama.

Tabel AP memakai hasil evaluasi tersimpan untuk laptop dan bantal. Nilainya mengikuti model dan ukuran input. Threshold di halaman berlaku untuk mask yang ditampilkan saat inferensi. Waktu di bawah gambar adalah waktu prediksi per model, di luar waktu memuat bobot dan menggambar mask. VRAM dan waktu training di tabel berasal dari log eksperimen.
