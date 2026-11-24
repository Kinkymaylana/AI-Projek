# Log Kerja Anggota C — Evaluasi, Demo & Integrasi (Proyek 07)

## Ringkasan

Pekerjaan per 24 November 2026: modul evaluasi bersama (`src/evaluate.py`) selesai dan teruji, interface inference Anggota B sudah diverifikasi dan dipakai di demo Gradio 4 tab (`app.py`), audit anti-leakage sudah dijalankan (0 FAIL, 1 WARN, 1 PENDING), dan analisis stabilitas yang tidak memerlukan label sudah selesai. Temuan utamanya: subset SA hanya sedikit lebih stabil daripada subset acak berukuran sama (indeks Nogueira 0,106).

Pada 24 November 2026 Anggota B menyerahkan tiga skrip eksperimen dan empat model final (commit `1a1b51b`). Audit diulang (0 FAIL), dan demo kini membandingkan model final 90 fitur dan 52 fitur secara berdampingan.

Evaluasi test sekali, uji noise, error analysis, dan matriks transfer belum dijalankan. Model final SVM/RF sudah ada, tetapi pembanding hill climbing, random subset, sweep λ, dan ensemble dari B serta mutual information dari A belum ada. Evaluasi test ditunda sampai grup memastikan pembanding mana yang akan dibuat, karena setiap metode hanya boleh dievaluasi sekali di test.

| Langkah (Rencana Kerja) | Status | Hasil utama |
| --- | --- | --- |
| 1. Knowledge map (Minggu 2) | Belum tercatat di repo | Perlu dikonfirmasi: sudah dibuat di luar repo atau belum |
| 2. Kerangka repo | Selesai | `requirements.txt` terkunci dan teruji, `set_seed`, `README.md` |
| 3. Modul evaluasi bersama | Selesai | `src/evaluate.py` v1.1; 10 tes lolos |
| 4. Mini-eksperimen graph search (Minggu 5) | Belum tercatat di repo | Perlu dikonfirmasi |
| 5. Proposal 1 halaman (Minggu 7) | Belum tercatat di repo | Perlu dikonfirmasi |
| 6. Audit anti-leakage | Selesai, diulang setelah file baru masuk | `src/audit_leakage.py`, `docs/audit_anti_leakage.md` |
| 7. Evaluasi test sekali | Menunggu | Model final SVM/RF (90 dan 52 fitur) sudah ada; menunggu kepastian pembanding B dan MI dari A |
| 8. Gambar hasil | Sebagian | Gambar stabilitas selesai; trade-off λ dan confusion matrix menunggu |
| 9. Analisis stabilitas | Sebagian | Heatmap, Jaccard, Nogueira, kelompok redundan selesai; matriks transfer dikunci sampai test tercatat |
| 10. Uji noise | Menunggu langkah 7 | Tabel test bernoise dari A sudah siap |
| 11. Error analysis | Menunggu langkah 7 | 2 file test tanpa nada (aktor 21) sudah dicatat A |
| 12. Demo Gradio | Alpha berjalan | 4 tab di `app.py` v0.2: kurva SA interaktif dari history, probabilitas model final 90 vs 52 fitur berdampingan |
| 13. Integrasi paper dan slide | Belum | — |

**Jaminan anti-leakage:** data test (aktor 21–24) belum pernah dievaluasi oleh C. Semua pengujian demo memakai audio sintetis, dan pengecekan model memakai validation (aktor 17–20).

---

## Langkah 2 — Kerangka repo

### Yang dikerjakan

1. `requirements.txt` dikunci dan diuji pada lingkungan bersih Python 3.13.18 (`uv venv`, lalu `uv pip install -r requirements.txt`): semua paket terpasang tanpa konflik.
2. `scikit-learn==1.6.1` dikunci karena itu versi yang dipakai saat `best_model_p0.joblib` disimpan; model dimuat tanpa `InconsistentVersionWarning`.
3. `set_seed()` ada di `src/evaluate.py` (seed `random`, `numpy`, dan `PYTHONHASHSEED`).
4. `README.md` utama berisi ringkasan proyek, pembagian jalur, instalasi, perintah, struktur folder, aturan anti-leakage, dan status.

### Keputusan desain

| Keputusan | Alasan |
| --- | --- |
| Environment di `.venv` dalam repo (sudah diabaikan Git) | VS Code otomatis menemukannya; anggota lain cukup menjalankan dua perintah di README |
| Tidak ada seed global untuk scikit-learn | scikit-learn memang tidak punya seed global; setiap estimator tetap harus diberi `random_state=42` |

### Catatan

- Instalasi lewat pip biasa (`py -3.13 -m venv`) ditulis di README tetapi belum diuji, karena laptop C tidak punya Python 3.13 terpasang di luar uv.

---

## Langkah 3 — Modul evaluasi bersama (`src/evaluate.py`)

### Isi modul (versi 1.1)

| Fungsi | Kegunaan |
| --- | --- |
| `LABELS` | Urutan 8 emosi untuk semua confusion matrix dan tabel (urutan kode RAVDESS 01–08, sama dengan `EMOTION_ORDER` Anggota B) |
| `set_seed()` | Seed `random` dan `numpy` |
| `evaluate()` | Accuracy, macro F1, UAR, F1 per kelas, confusion matrix |
| `bootstrap_ci()` | Interval kepercayaan bootstrap 95% (1.000 kali) untuk satu metrik |
| `paired_bootstrap()` | Interval selisih dua model pada sampel yang sama, untuk klaim "setara" |
| `mcnemar_test()` | Uji McNemar eksak untuk full vs selected |
| `plot_confusion()` | Gambar confusion matrix |
| `result_row()`, `append_result()` | Menulis satu baris ke `results/main_table.csv` |

### Keputusan desain

| Keputusan | Alasan |
| --- | --- |
| Label yang tidak dikenal ditolak (`ValueError`) | Menangkap typo seperti `fear` dan kode angka seperti `5` sebelum masuk tabel |
| Peringatan jika ada kelas yang tidak muncul di `y_true` | F1 kelas itu dihitung 0 dan menurunkan macro F1 |
| McNemar memakai uji binomial eksak | Jumlah pasangan yang berbeda bisa kecil; uji eksak tidak butuh pendekatan chi-kuadrat |
| 4 kolom tambahan di `main_table.csv`: `run_id`, `seed`, `evaluated_at`, `note` | Rencana Kerja mensyaratkan setiap angka paper bisa ditelusuri ke satu baris dan tanggal evaluasi test dicatat. **Ini perubahan kontrak file dan harus diumumkan ke grup.** |
| Baris split `test*` ditolak tanpa `allow_test=True` | Hanya C yang boleh mencatat hasil test |
| (v1.1) Metode yang sudah tercatat pada split test yang sama ditolak | Mencegah test diulang lalu dipilih hasil terbaik |
| (v1.1) Lokasi tabel memakai path absolut dari root repo | Sebelumnya, skrip yang dijalankan dari folder lain membuat tabel kedua di tempat lain |

### Verifikasi

10 tes di `tests/test_evaluate.py` lolos, termasuk perhitungan tangan (macro F1 7/9, UAR 5/6), kecocokan dengan scikit-learn, urutan confusion matrix, McNemar eksak, determinisme bootstrap, pengaman test, dan lokasi tabel.

### Keterbatasan

- Bootstrap dilakukan per file. File dari aktor yang sama tidak independen, sehingga interval cenderung terlalu sempit. Dengan hanya 4 aktor test, bootstrap per aktor tidak praktis; keterbatasan ini harus disebut di paper.
- Pengaman test bersifat konsekuensi: nama metode yang salah tulis saat evaluasi test tidak bisa diperbaiki dengan mengulang evaluasi. Nama metode harus final sebelum `append_result(..., allow_test=True)` dipanggil.

---

## Langkah 6 — Audit anti-leakage

### Yang dikerjakan

Audit dibuat sebagai skrip (`src/audit_leakage.py`) agar bisa diulang setiap ada file baru. Skrip tidak membuka fitur atau label test untuk evaluasi. Detail lengkap: `docs/audit_anti_leakage.md`; hasil mentah: `results/audit_leakage.json`.

### Hasil audit ulang (24 November 2026, setelah commit `1a1b51b` B): 10 OK, 1 WARN, 1 PENDING, 0 FAIL

| Pemeriksaan | Status |
| --- | --- |
| Aktor tidak tumpang tindih di split utama dan 5 partisi | OK |
| Tabel fitur 1.440 baris tanpa NaN; setiap split punya 2 gender dan 8 emosi | OK |
| Scaler di dalam Pipeline; scaler model P0 melihat 960 sampel = train P0 | OK |
| (baru) Empat model final di `models/`: scaler melihat 1.200 sampel = aktor 1–20, aktor test tidak ada, subset 52 fitur sama dengan SA utama | OK |
| Objective SA memakai scaler per fold GroupKFold di dalam train | OK |
| Kode dan notebook B tidak membaca test (kini 10 file: seluruh `progres_anggota_b/` dan `train_final_models.py`) | OK |
| Tiga skrip eksperimen B tersedia untuk diaudit | OK |
| Run stabilitas P1–P4 tidak memakai aktor test utama | **WARN** |
| Mutual information dihitung di train saja | PENDING |
| Test belum dievaluasi | OK |

### Temuan WARN

Run stabilitas P1–P4 sudah dijalankan B pada 22 November 2026 dengan aktor 21–24 di train (P1–P3) dan di validation (P4), padahal `partitions.json` meminta run ini setelah hasil utama final. Model utama tidak terpengaruh, tetapi angka validation P4 dan rata-rata "Validation Macro F1 0,4734 ± 0,0286" lintas 15 run ikut memuat skor pada aktor test. Angka itu tidak boleh dipakai untuk keputusan apa pun, dan urutan pengerjaan ini dicatat di Experimental Protocol paper.

### Perbaikan skrip audit (v1.1)

- Versi 1.0 hanya memindai `progres_anggota_b/src/` dan `notebooks/`, sehingga skrip baru B di `scripts/` dan `train_final_models.py` di root tidak ikut terperiksa. Pemindaian kini rekursif di `progres_anggota_b/` ditambah daftar `B_ROOT_FILES`.
- Baris yang hanya membaca daftar nomor aktor test (`test_actors = sorted(... splits['test'])`, dipakai B untuk assert tidak tumpang tindih) sempat terbaca sebagai FAIL. Baris seperti itu kini dikecualikan, kecuali bila baris yang sama memfilter tabel data.

---

## Langkah 9 — Analisis stabilitas (sebagian)

### Yang dikerjakan

1. Modul `src/stability.py`: indeks Nogueira et al. (2018), Jaccard subset acak sebagai pembanding, tingkat keterwakilan kelompok redundan, dan matriks transfer. 7 tes di `tests/test_stability.py` lolos.
2. Skrip `src/run_stability_analysis.py` membaca 15 subset dari `progres_anggota_b/results/subsets.json` dan kelompok redundan dari `data/analysis/corr_clusters_peak.csv`. Tidak ada model yang dilatih dan tidak ada label yang dibaca.
3. Output: `results/stability_summary.json`, `results/stability_cluster_rate.csv`, `figures/stability/stab_01_heatmap_seleksi.png`, `figures/stability/stab_02_ringkasan_stabilitas.png`.

### Hasil (15 run: P0–P4 × seed 42, 123, 2026)

| Ukuran | SA | Subset acak berukuran sama |
| --- | --- | --- |
| Jaccard per fitur | 0,395 | 0,344 |
| Jaccard per kelompok redundan | 0,489 | 0,437 |
| Indeks Nogueira (1 = identik, 0 = setara acak) | 0,106 | 0 |

| Ukuran | Partisi sama (beda seed) | Partisi beda |
| --- | --- | --- |
| Jaccard per fitur | 0,420 | 0,391 |
| Jaccard per kelompok redundan | 0,508 | 0,486 |

Pembanding lain: tumpang-tindih aktor train antarpartisi (Jaccard) 0,467. Jumlah fitur k = 46,33 ± 5,79 (rentang 38–58). Hanya `voiced_ratio` yang terpilih di semua 15 run; tidak ada fitur yang tidak pernah terpilih.

### Interpretasi

1. **SA hanya sedikit lebih stabil daripada pemilihan acak.** Jaccard 0,39 tampak sedang, tetapi subset acak berukuran sama sudah mencapai 0,34 karena setiap subset berisi sekitar separuh dari 90 fitur. Nogueira mengoreksi efek ukuran ini dan hanya bernilai 0,106.
2. **Sumber ketidakstabilan terutama dari SA sendiri, bukan dari speaker.** Mengganti seed pada partisi yang sama (0,420) hampir sama berpengaruhnya dengan mengganti aktor (0,391).
3. **Hipotesis Anggota A (langkah 11) didukung.** Kelompok redundan terwakili lebih konsisten daripada anggotanya. Contoh: kelompok `zcr_mean`/`centroid_mean`/`rolloff_mean` terwakili di 87% run, sedangkan tiap anggotanya rata-rata hanya di 44% run; kelompok `mfcc12_std`/`mfcc13_std`/`dmfcc12_std`/`dmfcc13_std` terwakili di 100% run dengan rata-rata anggota 55%.
4. **Bahan research seed:** ketidakstabilan ini mendukung pertanyaan riset di Rencana Kerja, yaitu apakah penalti instabilitas pada objective SA menghasilkan subset yang stabil lintas speaker.

### Keputusan desain

| Keputusan | Alasan |
| --- | --- |
| Indeks Nogueira, bukan Kuncheva | Kuncheva hanya sah untuk subset berukuran sama, sedangkan k hasil SA 38–58 (Rencana Kerja langkah 9; referensi no. 11) |
| Jaccard dibandingkan dengan subset acak berukuran sama | Jaccard naik dengan sendirinya jika subset besar; tanpa titik nol, angka 0,39 mudah disalahartikan |
| Simpangan baku sampel (ddof = 1) | B melaporkan 5,59 (simpangan baku populasi); skrip C menghasilkan 5,79. **Perlu disepakati satu rumus sebelum paper.** |
| Matriks transfer 5 × 5 dikunci sampai hasil test tercatat | Partisi P1–P4 memakai aktor 21–24; `partitions.json` meminta run ini setelah hasil utama final. Skrip menolak `--transfer` sebelum ada baris test di `main_table.csv`. |

### Sisa langkah 9

- Matriks transfer 5 × 5 (`src/run_stability_analysis.py --transfer`) setelah evaluasi test.
- Jika B menambah seed SA di split utama menjadi 5 (sesuai Rencana Kerja), jalankan ulang skrip.

---

## Langkah 12 — Demo Gradio

### Verifikasi interface Anggota B

Interface `progres_anggota_b/src/inference.py` diuji dengan audio sintetis (bukan data RAVDESS) pada Python 3.13.18:

| Input | Hasil |
| --- | --- |
| WAV mono 16 kHz, WAV stereo 44,1 kHz, FLAC | Berhasil; prediksi identik |
| Array int16, stereo int16, float32 | Berhasil; prediksi identik dengan input file |
| Struktur output | Sesuai panduan B: 8 skor, 52 fitur, `score_type = softmax_decision` |
| Model di validation (aktor 17–20) | Macro F1 0,4995, UAR 0,5078, accuracy 0,5125, sama dengan laporan B |

Tiga masalah penanganan input ditemukan lalu diperbaiki di sisi demo (tanpa mengubah kode B):

| Masalah | Perbaikan |
| --- | --- |
| File rusak atau bukan audio memunculkan error mentah | Ditangkap; pengguna diminta mengunggah ulang WAV |
| Audio hening total tetap diberi prediksi | Ditolak jika puncak < 0,001; puncak terkecil di 1.440 file RAVDESS 0,00455, jadi tidak ada rekaman dataset yang tertolak |
| Skor tampil sebagai persen (mis. 65%) sehingga terbaca sebagai probabilitas | Ditampilkan sebagai tabel skor relatif desimal (mis. 0,65) |

File uji sementara `app_uji_inference.py` dihapus setelah fungsinya dipindahkan ke `app.py`.

### Isi `app.py` saat ini

| Tab | Isi | Sesuai Rencana Kerja? |
| --- | --- | --- |
| Optimasi | Ringkasan SA; kurva interaktif per iterasi dari `history` di `sa_main_results.json` (objective, CV macro F1, jumlah fitur, suhu; solusi saat ini vs terbaik); konfigurasi | Ya. Log per run `logs/sa_<run_id>.csv` belum ada, jadi kurva memakai `history` |
| Fitur terpilih | Daftar 90 fitur dengan filter dan pencarian, heatmap fitur × run, sebaran per kelompok | Ya |
| Perbandingan | 90 fitur vs subset SA di validation, F1 per emosi, stabilitas, status test | Sebagian: test dan MI menunggu |
| Coba sendiri | Rekam atau unggah audio → probabilitas 8 emosi dari model final SVM 90 fitur dan SVM 52 fitur, berdampingan | Ya |

**Perubahan v0.2 (setelah model final B tiba).** Tab Coba sendiri memakai `models/svm_all90_final.joblib` dan `models/svm_sa52_final.joblib`. Jika folder `models/` tidak ada, demo kembali ke model P0. Karena model final memakai `probability=True`, skor kini ditampilkan sebagai persen. Label SVM (`predict`, dari decision function) bisa berbeda dari kelas dengan probabilitas tertinggi (Platt scaling): ini terjadi pada 9,0% (90 fitur) dan 5,2% (52 fitur) file train + validation. Demo memberi catatan bila itu terjadi, dan evaluasi test memakai `predict()`.

Semua angka dibaca dari file hasil, tidak diketik manual, dan aplikasi tidak mengevaluasi data test. Hasil test tampil otomatis setelah tercatat di `main_table.csv`.

### Sisa langkah 12

- Uji tab Coba sendiri dengan mikrofon dan file RAVDESS aktor 1–20.
- Pastikan demo jalan di Colab (`share=True`) atau Hugging Face Spaces tanpa mengubah kode.

---

## Catatan untuk paper

1. **Rancangan SA Anggota B berbeda dari Rencana Kerja** dan perlu dijelaskan di Method: objective memakai macro F1 validasi silang 4-fold di train (bukan validation), λ = 0,015 (bukan 0,02), 500 iterasi dengan pendinginan geometrik dan reheating, serta 3 seed (bukan 5).
2. **Subset SA belum terbukti lebih baik di validation.** Macro F1 SVM: 90 fitur 0,5165, subset SA 52 fitur 0,4995 (selisih −0,0170, belum diuji statistik). Pakai frasa "setara dengan X% fitur" hanya jika paired bootstrap atau McNemar di test mendukungnya.
3. **Dampak per emosi berbeda:** F1 neutral turun 0,14 dan surprised turun 0,11, sedangkan fearful naik 0,12 (validation). Bahan error analysis.
4. **Skor demo.** Model P0 memakai skor relatif (softmax `decision_function`). Model final memakai `probability=True` (Platt scaling), sehingga label `predict()` bisa berbeda dari probabilitas tertinggi pada 5–9% sampel. Tabel test memakai `predict()`.
5. **Model final dilatih ulang di train + validation (aktor 1–20)** sesuai langkah 8 Rencana Kerja B. Hyperparameter SVM 90 fitur (`gamma = 0,005`) dan SVM 52 fitur (`gamma = 'scale'`) diambil dari tuning masing-masing; **RF 52 fitur memakai hyperparameter RF 90 fitur tanpa tuning ulang**. Catat di Method.
6. Macro F1 5-fold GroupKFold di train + validation (diagnostik, bukan tuning): SVM 90 fitur 0,4868 ± 0,0448; SVM 52 fitur 0,5034 ± 0,0576; RF 90 fitur 0,4292 ± 0,0358; RF 52 fitur 0,4394 ± 0,0445. B memakai simpangan baku populasi.

---

## Komunikasi dengan anggota lain

| Tanggal | Kepada | Isi | Status |
| --- | --- | --- | --- |
| 24 November 2026 | Grup | Perubahan kontrak `main_table.csv` (4 kolom tambahan) dan aturan test sekali | Disiapkan; tanggal dikirim: … |
| 24 November 2026 | Anggota B | Hasil verifikasi interface, temuan WARN audit, daftar file yang dibutuhkan C | Disiapkan; tanggal dikirim: … |
| 24 November 2026 | Anggota A | Jawaban konfirmasi 4a–4b, konfirmasi pemakaian file langkah 7, 9, 11 | Disiapkan; tanggal dikirim: … |

**Jawaban untuk tabel "Konfirmasi dari Anggota C" di log Anggota A**

| Poin | Isi | Jawaban C |
| --- | --- | --- |
| 4a | Format gambar (PNG 200 dpi, judul bahasa Indonesia) cocok untuk paper | PNG 200 dpi cocok. Judul gambar untuk paper diganti ke **bahasa Inggris** (paper memakai template IEEE); keputusan sementara sampai dikonfirmasi dosen |
| 4b | Demo memakai `src/preprocess.py` | Ya. Demo memanggil `inference.py` milik B, yang memakai `src/preprocess.py` dan `src/features.py` dengan normalisasi `peak` |

**Pesan ke grup**

> Halo semua, ini dari aku (Anggota C).
>
> 1. **Perubahan kontrak `results/main_table.csv`.** Selain 8 kolom di Rencana Kerja, aku menambah 4 kolom: `run_id`, `seed`, `evaluated_at`, `note`, supaya setiap angka di paper bisa ditelusuri dan tanggal evaluasi test tercatat. Kolom lama tidak berubah.
> 2. **Aturan test di `evaluate.py` v1.1:** baris test hanya bisa ditulis dengan `allow_test=True`, dan satu metode hanya bisa dicatat sekali per split test. Jadi tolong pastikan nama metode sudah final sebelum evaluasi akhir.
> 3. Cara pakai dan perintah lengkap ada di `README.md`.
>
> Kalau ada keberatan dengan kolom tambahan, kabari sebelum evaluasi test. Terima kasih!

**Pesan ke Anggota B**

> Halo B, ini dari aku (Anggota C).
>
> 1. **Interface inference sudah aku uji dan jalan.** WAV, FLAC, stereo, dan array numpy semuanya berhasil, dan model menghasilkan macro F1 validation 0,4995, sama dengan laporanmu. Tiga penanganan input (file rusak, audio hening, tampilan skor) aku tangani di sisi demo, jadi `inference.py` tidak perlu diubah.
> 2. **Temuan audit (WARN):** run stabilitas P1–P4 memakai aktor 21–24 di train/validation, padahal `partitions.json` meminta run ini setelah hasil utama final. Model P0 tidak terpengaruh, tapi angka validation P4 dihitung di aktor test. Tolong jangan ubah SA, λ, atau model berdasarkan angka P4. Detailnya di `docs/audit_anti_leakage.md`.
> 3. **Terima kasih untuk 3 skrip dan 4 model final (`1a1b51b`).** Sudah aku audit: skrip tidak membaca test, keempat model di-fit hanya pada aktor 1–20, dan demo sudah memakai model 90 vs 52 fitur berdampingan.
> 4. **Pembanding hill climbing, 30 random subset, sweep λ, dan ensemble: jadi dibuat atau tidak?** Setiap metode hanya bisa dievaluasi sekali di test, jadi aku tunda evaluasi test sampai daftar metode final. Kalau tidak sempat, kita catat sebagai keterbatasan dan aku evaluasi 4 model yang ada.
> 5. RF 52 fitur memakai hyperparameter RF 90 fitur tanpa tuning ulang. Sengaja? Akan aku tulis di Method.
> 6. Log per run `logs/sa_<run_id>.csv` belum ada; untuk sekarang kurva di demo aku ambil dari `history` di `sa_main_results.json`.
> 7. Rencana Kerja meminta SA 5 seed di split utama; sekarang ada 3. Kalau tidak sempat, kita catat sebagai keterbatasan.
> 8. Untuk simpangan baku, punyamu memakai populasi (mis. k 5,59) dan punyaku sampel (5,79). Kita pakai yang mana di paper?
> 9. Gambar di `progres_anggota_b/figures/` yang masuk paper perlu versi bahasa Inggris.
>
> Terima kasih!

**Pesan ke Anggota A**

> Halo A, ini dari aku (Anggota C).
>
> 1. **Jawaban 4b:** demo memakai `src/preprocess.py` dan `src/features.py` lewat `inference.py` B, dengan normalisasi `peak`.
> 2. **Jawaban 4a:** PNG 200 dpi sudah cocok. Karena paper memakai template IEEE, kita pakai **bahasa Inggris** untuk paper dan judul gambarnya (sementara, sampai dikonfirmasi dosen). Tolong buat versi bahasa Inggris untuk gambar EDA, fitur, noise, dan korelasi yang akan masuk paper. Gambar stabilitasku sudah berbahasa Inggris. Demo dan dokumen internal tetap bahasa Indonesia.
> 3. File langkah 7, 9, dan 11 sudah aku pakai. `src/redundancy.py` jalan dan hipotesismu didukung: kelompok redundan terwakili lebih konsisten daripada anggotanya (Jaccard per kelompok 0,489 vs per fitur 0,395). Tapi setelah dibandingkan dengan subset acak berukuran sama, stabilitas SA ternyata rendah (Nogueira 0,106).
> 4. **Yang aku tunggu:** pembanding mutual information (langkah 10) dengan k = 52, dihitung di train saja, dan hasil MLP full vs selected (Minggu 12).
>
> Terima kasih!

---

## Kronologi

| Tanggal | Kegiatan |
| --- | --- |
| 24 November 2026 | `src/evaluate.py` v1.0 dan 8 tes; `requirements.txt` dikunci dan diuji di Python 3.13.18 |
| 24 November 2026 | Interface inference B diverifikasi; model cocok dengan laporan B di validation |
| 24 November 2026 | Tiga masalah penanganan input diperbaiki; `app.py` 4 tab dibuat; `app_uji_inference.py` dihapus |
| 24 November 2026 | `evaluate.py` v1.1: pengaman test sekali per metode dan path tabel tetap; 10 tes lolos |
| 24 November 2026 | Rencana Kerja, Panduan Proyek MK, dan Referensi dicocokkan dengan isi repo |
| 24 November 2026 | Analisis stabilitas tanpa label: `src/stability.py`, `src/run_stability_analysis.py`, 2 gambar; heatmap ditambahkan ke demo; 17 tes lolos |
| 24 November 2026 | Audit anti-leakage: `src/audit_leakage.py`, `docs/audit_anti_leakage.md`; `README.md` utama ditulis |
| 24 November 2026 | Keputusan sementara: paper dan judul gambar paper berbahasa Inggris (menunggu konfirmasi dosen); gambar stabilitas diubah ke bahasa Inggris |
| 24 November 2026 | Pekerjaan C di-push (5 commit, `9cb6627`–`adc91ad`) |
| 24 November 2026 | B push 3 skrip dan 4 model final (`1a1b51b`). Audit v1.1 diulang: 10 OK, 1 WARN, 1 PENDING, 0 FAIL. Demo v0.2: kurva SA interaktif dan model final 90 vs 52 fitur |

---

## Checklist pekerjaan Anggota C (per 24 November 2026)

Centang langsung di dokumen setelah selesai.

**Pekerjaan teknis**

- [ ] Langkah 1: knowledge map (konfirmasi: sudah dibuat di luar repo?)
- [x] Langkah 2: `requirements.txt` terkunci dan teruji, `set_seed`, `README.md`
- [x] Langkah 3: `src/evaluate.py` v1.1 dengan 10 tes
- [ ] Langkah 4: mini-eksperimen graph search (konfirmasi)
- [ ] Langkah 5: proposal 1 halaman (konfirmasi)
- [x] Langkah 6: audit anti-leakage dijalankan dan didokumentasikan
- [x] Langkah 6: audit ulang setelah skrip dan model final B masuk
- [ ] Langkah 6: audit ulang setelah MI dari A masuk
- [ ] Langkah 7: evaluasi test sekali, CI bootstrap, McNemar, `main_table.csv`
- [ ] Langkah 8: trade-off λ dan confusion matrix full vs selected
- [x] Langkah 9: heatmap, Jaccard, Nogueira, kelompok redundan
- [ ] Langkah 9: matriks transfer 5 × 5 (setelah test)
- [ ] Langkah 10: uji noise (test bersih, 20, 10, 0 dB)
- [ ] Langkah 11: error analysis minimal 3 file
- [x] Langkah 12: demo alpha 4 tab berjalan
- [x] Langkah 12: model 90 vs 52 fitur berdampingan di Coba sendiri; kurva Optimasi dari history SA
- [ ] Langkah 12: uji mikrofon dan file RAVDESS aktor 1–20; jalan di Colab/Spaces
- [ ] Langkah 13: integrasi paper IEEE dan slide
- [ ] Bahasa paper (Inggris) dikonfirmasi ke dosen
- [x] Gambar stabilitas berbahasa Inggris untuk paper
- [ ] `reproduce.ipynb` (Definition of done)

**Repo**

- [x] Perubahan Anggota C di-commit dan di-push (5 commit awal)
- [ ] Isi `progres_anggota_b/` dipindah ke struktur utama (setelah sepakat dengan B), lalu path di `inference.py` dan `app.py` disesuaikan

**Komunikasi**

- [ ] Pesan ke grup (kontrak `main_table`, aturan test) dikirim
- [ ] Pesan ke Anggota B dikirim dan jawabannya dicatat
- [ ] Pesan ke Anggota A dikirim; jawaban 4a–4b diisi di log Anggota A

## Langkah berikutnya

1. Kirim ketiga pesan di atas dan catat tanggalnya.
2. Uji tab Coba sendiri dengan mikrofon.
3. Setelah daftar metode final disepakati (pembanding B) dan MI dari A tiba: jalankan audit ulang, lalu evaluasi test sekali (langkah 7), uji noise, error analysis, dan matriks transfer.
4. Mulai kerangka paper IEEE: Introduction dan Failure Analysis bisa ditulis sekarang dari hasil stabilitas.
