# Audit Anti-Leakage — Proyek 07

Penanggung jawab: Anggota C · Langkah 6 Rencana Kerja · Diaudit: 24 November 2026

Audit ini dijalankan dengan skrip, sehingga bisa diulang setiap kali ada file baru dari anggota lain:

```powershell
.venv\Scripts\python.exe src\audit_leakage.py
```

Hasil lengkap tersimpan di `results/audit_leakage.json`. Skrip tidak membuka fitur atau label data test untuk evaluasi; ia hanya membaca daftar aktor, komposisi split, model tersimpan, dan kode sumber.

## Hasil

**Kesimpulan: tidak ditemukan kebocoran pada model utama, 11 model final, maupun pembanding mutual information (11 OK, 1 WARN, 0 FAIL).** Ada 1 temuan yang wajib dicatat di paper.

Audit ulang 24 November 2026 setelah:

- commit `1a1b51b` Anggota B: tiga skrip eksperimen dan empat model final;
- commit `25f93ca` Anggota B: pembanding hill climbing, random subset, sweep λ, ensemble, log per iterasi, dan skrip `run_comparators_and_curves.py`;
- merge PR #1 (`8e4f6fc`) Anggota A: langkah 10 mutual information, langkah 12 MLP, dan mini-eksperimen.

| No | Pemeriksaan | Status | Bukti |
| --- | --- | --- | --- |
| 1 | Split utama: aktor tidak tumpang tindih, 24 aktor terpakai | OK | train 1–16, validation 17–20, test 21–24 |
| 1 | Partisi P0–P4: aktor tidak tumpang tindih | OK | 5 partisi valid; P0 sama dengan split utama |
| 2 | Tabel fitur: 1.440 baris unik, tanpa NaN | OK | 1.440 baris, 0 NaN |
| 2 | Setiap split punya 2 gender dan 8 emosi | OK | train 480 L / 480 P, validation dan test masing-masing 120 L / 120 P |
| 3 | Model P0 (demo awal): StandardScaler di dalam Pipeline, di-fit pada train saja | OK | Pipeline `[StandardScaler, SVC]`; scaler melihat 960 sampel = jumlah file train P0 |
| 3 | Model final `models/**/*.joblib`: di-fit pada train + validation saja | OK | 11 model: SVM/RF × 90/52 fitur, ensemble SVM+RF × 90/52, SVM hill climbing (48 fitur), SVM random subset median (52), MLP × 90/SA-52/MI-52. Semua scaler melihat 1.200 sampel = aktor 1–20; subset SA 52 sama dengan hasil SA utama. Bundle MLP Anggota A tidak mencatat `train_val_actors`, jadi buktinya dari jumlah sampel scaler dan verifikasi di notebook langkah 12 |
| 3 | Objective SA: scaler di-fit per fold GroupKFold di dalam train | OK | `_precompute_scaled_folds` di `sa_feature_selection.py` |
| 4 | Kode dan notebook Anggota B tidak membaca data test | OK | 11 file: semua `.py`/`.ipynb` di `progres_anggota_b/`, `train_final_models.py`, dan `run_comparators_and_curves.py`; bagian test dari `split_frames` dibuang (`_`) |
| 4 | Skrip eksperimen B tersedia untuk diaudit | OK | `run_baseline.py`, `run_sa_main.py`, `run_sa_stability.py` di `progres_anggota_b/scripts/`, serta `run_comparators_and_curves.py` di root |
| 4 | Run stabilitas tidak memakai aktor test utama sebelum evaluasi akhir | **WARN** | lihat di bawah |
| 5 | Mutual information dihitung di train saja | OK | `mutual_info_classif(X_tr, y_tr, …)` di `notebooks/Langkah10_AnggotaA_Mutual_Information.ipynb`; `X_tr` berasal dari aktor 1–16 dan notebook memverifikasi bentuknya (960, 90). Catatan: `data/analysis/mi_topk_peak.json` belum diunggah; daftar 52 fitur sementara diambil dari `mi_scores_peak.csv` (`in_top_k = True`) |
| 6 | Hasil test tercatat hanya oleh C, sekali | OK | test belum dievaluasi |

Catatan pemeriksaan no. 4: baris kode yang hanya membaca **daftar nomor aktor** test, misalnya `test_actors = sorted(int(a) for a in splits['test'])` di `train_final_models.py` untuk memastikan aktor test tidak ikut dilatih, tidak dihitung sebagai membaca data test. Pengecualian ini gugur bila baris yang sama memfilter tabel data (`isin`, `loc`, `read_csv`, dan sejenisnya).

## Temuan WARN: run stabilitas P1–P4

Pada partisi siklik, aktor 21–24 (test utama) berada di train untuk P1–P3 dan di validation untuk P4. Anggota B sudah menjalankan ke-15 run stabilitas pada 22 November 2026, sedangkan `data/partitions.json` meminta run ini dilakukan setelah hasil utama final.

Dampaknya:

- **Model utama dan model final tidak terpengaruh.** `best_model_p0.joblib` dilatih hanya pada aktor 1–16 dan ke-11 model final hanya pada aktor 1–20. `train_final_models.py` mengambil hyperparameter dan subset dari `baseline_results.json` dan `sa_main_results.json` (split P0), bukan dari run P1–P4.
- **Angka validation P4 sebenarnya dihitung pada aktor test.** Kolom `val_macro_f1` untuk P4 di `subsets.json` dan rata-rata "Validation Macro F1 0,4734 ± 0,0286" lintas 15 run ikut memuat skor pada aktor 21–24. Angka ini tidak boleh dipakai untuk keputusan apa pun dan perlu diberi catatan bila dikutip di paper.
- **Analisis stabilitas C tetap sah.** `src/run_stability_analysis.py` hanya memakai daftar fitur terpilih, tanpa label. Matriks transfer 5 × 5 yang melatih model di P1–P4 dikunci sampai hasil test tercatat.

Tindakan: catat urutan pengerjaan ini di bagian Experimental Protocol paper, dan jangan mengubah SA, λ, atau model setelah melihat angka P4.

## Yang harus diaudit ulang

1. Setiap kali Anggota B menambah kode: jalankan ulang skrip audit. Pemindaian otomatis mencakup semua `.py` dan `.ipynb` di `progres_anggota_b/` (termasuk subfolder), `train_final_models.py`, dan `run_comparators_and_curves.py`. Bila B menaruh skrip baru di luar folder itu, tambahkan namanya ke `B_ROOT_FILES` di `src/audit_leakage.py`.
2. Model final ditemukan otomatis di semua subfolder `models/`. Model baru (misalnya SVM MI-52 dari Anggota B) langsung ikut diperiksa.
3. Pemanggilan `mutual_info_classif` dicari di semua `.py` dan `.ipynb` repo; statusnya OK hanya jika argumen pertamanya data train (`X_tr` atau `X_train`).
4. Sebelum evaluasi test: jalankan audit sekali lagi dan simpan `results/audit_leakage.json` sebagai bukti.
