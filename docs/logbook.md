## 17 November 2026

* Memeriksa dataset dan menemukan bahwa ekspor awal memakai OBB. Setelah ekspor COCO segmentation tersedia, saya memakai mask poligon. Catatan EXIF dari 113 foto memuat waktu pengambilan, tetapi kolom model kamera kosong.
* Menyiapkan Python 3.12 dengan PyTorch CUDA di RTX 4060 Laptop. Saya memeriksa orientasi dan anotasi ganda, lalu menguji evaluator dengan mask sintetis. Uji coba awal juga memastikan optimizer Mask R-CNN mengubah bobot.
* Menjalankan sembilan kombinasi dari tiga model dan ukuran input 320, 640, serta 960, masing-masing selama 50 epoch. Berkas DONE.json mencatat seluruh run selesai pada tanggal ini antara pukul 19.40 dan 23.58 WIB.
* Memakai 64 foto laptop dan bantal untuk membandingkan AP pada kedua data uji. Kelas keyboard tidak punya contoh pada split latih. Saya keluarkan kelas itu dari perbandingan AP dan menghitung ulang metrik dari prediksi tersimpan.

## 24 November 2026

* Merapikan repo dan notebook berbahasa Indonesia. Saya menambahkan tabel lima kasus gagal yang menunjukkan label prediksi, confidence, dan IoU mask. Pada salah satu foto, mask mengikuti bentuk bantal dengan IoU 0,969, tetapi model memberi label dompet.
* Menyimpan kode, hasil, dan analisis dalam commit lokal cc5cf21. Riwayat ini mencatat perapian repo setelah eksperimen selesai.

## 25 November 2026

* Menambahkan skrip inferensi webcam dengan threshold yang bisa diubah saat demo. Perubahan tercatat pada commit lokal 277d798.
* Membuat web lokal untuk membandingkan tiga model pada frame yang sama. Saya menambahkan statistik sembilan run, jumlah foto tiap split, dan daftar 74 kelas. Saya juga memperbaiki server agar halaman dan statistik tetap bisa dibuka saat Pillow belum tersedia. Pemeriksaan mencakup server dan data evaluasi. Inferensi dan webcam fisik belum teruji di sesi pengerjaan.