# Audit Anti-Leakage — Proyek 07

Penanggung jawab: Anggota C · Langkah 6 Rencana Kerja · Diaudit: 24 November 2026

Audit ini dijalankan dengan skrip, sehingga bisa diulang setiap kali ada file baru dari anggota lain:

```powershell
.venv\Scripts\python.exe src\audit_leakage.py
```

Hasil lengkap tersimpan di `results/audit_leakage.json`. Skrip tidak membuka fitur atau label data test untuk evaluasi; ia hanya membaca daftar aktor, komposisi split, model tersimpan, dan kode sumber.

## Hasil

**Kesimpulan: tidak ditemukan kebocoran pada model utama (0 FAIL).** Ada 1 temuan yang wajib dicatat di paper dan 2 pemeriksaan yang belum bisa dilakukan.

| No | Pemeriksaan | Status | Bukti |
| --- | --- | --- | --- |
| 1 | Split utama: aktor tidak tumpang tindih, 24 aktor terpakai | OK | train 1–16, validation 17–20, test 21–24 |
| 1 | Partisi P0–P4: aktor tidak tumpang tindih | OK | 5 partisi valid; P0 sama dengan split utama |
| 2 | Tabel fitur: 1.440 baris unik, tanpa NaN | OK | 1.440 baris, 0 NaN |
| 2 | Setiap split punya 2 gender dan 8 emosi | OK | train 480 L / 480 P, validation dan test masing-masing 120 L / 120 P |
| 3 | Model utama: StandardScaler di dalam Pipeline, di-fit pada train saja | OK | Pipeline `[StandardScaler, SVC]`; scaler melihat 960 sampel = jumlah file train P0 |
| 3 | Objective SA: scaler di-fit per fold GroupKFold di dalam train | OK | `_precompute_scaled_folds` di `sa_feature_selection.py` |
| 4 | Kode dan notebook Anggota B tidak membaca data test | OK | 3 modul dan 3 notebook; bagian test dari `split_frames` dibuang (`_`) |
| 4 | Skrip eksperimen B tersedia untuk diaudit | PENDING | `run_baseline.py`, `run_sa_main.py`, `run_sa_stability.py` belum ada di repo |
| 4 | Run stabilitas tidak memakai aktor test utama sebelum evaluasi akhir | **WARN** | lihat di bawah |
| 5 | Mutual information dihitung di train saja | PENDING | langkah 10 Anggota A belum ada |
| 6 | Hasil test tercatat hanya oleh C, sekali | OK | test belum dievaluasi |

## Temuan WARN: run stabilitas P1–P4

Pada partisi siklik, aktor 21–24 (test utama) berada di train untuk P1–P3 dan di validation untuk P4. Anggota B sudah menjalankan ke-15 run stabilitas pada 22 November 2026, sedangkan `data/partitions.json` meminta run ini dilakukan setelah hasil utama final.

Dampaknya:

- **Model utama tidak terpengaruh.** `best_model_p0.joblib` dilatih hanya pada aktor 1–16, dan hasil P1–P4 tidak dipakai untuk memilih normalisasi, hyperparameter, atau subset (dinyatakan di log Anggota B dan konsisten dengan isi model).
- **Angka validation P4 sebenarnya dihitung pada aktor test.** Kolom `val_macro_f1` untuk P4 di `subsets.json` dan rata-rata "Validation Macro F1 0,4734 ± 0,0286" lintas 15 run ikut memuat skor pada aktor 21–24. Angka ini tidak boleh dipakai untuk keputusan apa pun dan perlu diberi catatan bila dikutip di paper.
- **Analisis stabilitas C tetap sah.** `src/run_stability_analysis.py` hanya memakai daftar fitur terpilih, tanpa label. Matriks transfer 5 × 5 yang melatih model di P1–P4 dikunci sampai hasil test tercatat.

Tindakan: catat urutan pengerjaan ini di bagian Experimental Protocol paper, dan jangan mengubah SA, λ, atau model setelah melihat angka P4.

## Yang harus diaudit ulang

1. Setelah Anggota B menyerahkan tiga skrip: jalankan ulang skrip audit; pemeriksaan "kode tidak membaca test" otomatis mencakup skrip baru di folder mana pun dalam repo.
2. Setelah Anggota A menyerahkan pembanding mutual information: pastikan skor MI dihitung pada aktor 1–16 saja.
3. Sebelum evaluasi test: jalankan audit sekali lagi dan simpan `results/audit_leakage.json` sebagai bukti.
