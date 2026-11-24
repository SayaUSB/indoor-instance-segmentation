# Penggunaan AI

Saya mengambil foto ruangan dengan HP dan memberi anotasi di Roboflow. Saya menggunakan ChatGPT/Codex untuk membantu menulis kode persiapan data, menjalankan perbandingan model, memeriksa evaluasi COCO, dan menyusun notebook serta laporan.

Salah satu instruksi yang saya berikan selama pengerjaan:

> Sudah saya perbaiki format datasetnya. Keduanya menggunakan format coco segmentation

Salah satu masalah muncul saat kode evaluasi memakai daftar prediksi yang sama untuk dua pemanggilan COCO. Pemanggilan pertama mengubah isi daftar tersebut sehingga evaluasi berikutnya gagal. Perbaikannya ada di fungsi `score_predictions` pada `src/experiment.py`. Setiap evaluasi menerima salinan prediksi melalui `copy.deepcopy`. Self-check mengevaluasi mask sintetis yang sama dua kali untuk memeriksa perbaikannya.
