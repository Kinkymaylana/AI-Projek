# Jawaban Resmi Anggota B untuk Anggota C (7 Poin Persiapan Evaluasi Test & Paper)

Dokumen ini menjawab seluruh 7 poin pada tabel **"2. Yang masih ditunggu dari Anggota B"** agar Anggota C dapat langsung menjalankan evaluasi akhir `test` sekali secara serentak dan menyusun paper IEEE.

---

## Poin 1 — Model Final `SVM MI-52` (Selesai & Lolos Audit)
- **File:** `models/svm_mi52_final.joblib` (526,6 KB), juga sudah tercatat di `models/final_models_manifest.json` dan skrip reproduksi `train_final_models.py`.
- **Daftar 52 fitur:** diambil dari `data/analysis/mi_scores_peak.csv` dengan `in_top_k == True` (diurutkan menaik berdasarkan `index` 0–89, format bundle 100% identik dengan `models/svm_sa52_final.joblib`).
- **Protokol pelatihan (zero leakage):**
  - Hyperparameter dituning via 4-fold `GroupKFold` pada **train (aktor 1–16)** saja: `kernel='linear', C=0.05` (CV Macro-F1 train = `0,4718`; Validation Macro-F1 aktor 17–20 = `0,4257`, UAR = `0,4375`).
  - Model final di-fit ulang pada gabungan **train + validation (aktor 1–20, 1.200 file)** dengan `probability=True` dan `class_weight='balanced'`. Diagnostik 5-fold GroupKFold (`train+val`): Macro-F1 = `0,4977 ± 0,0501` (`ddof=1`).
- **Audit anti-leakage (`src/audit_leakage.py`):** sudah dijalankan ulang setelah `svm_mi52_final.joblib` masuk → **`11 OK, 1 WARN, 0 PENDING, 0 FAIL`**.

---

## Poin 2 — Daftar Metode Final untuk Tabel Utama Paper (Evaluasi Test Sekali)

Berikut 12 model final (9 dari B + 3 MLP dari A) yang siap dievaluasi serentak satu kali pada `test` (aktor 21–24, versi `peak`):

| No | Nama Metode (`main_table.csv`) | File Model | Fitur (`k`) | Pemilik |
| :-: | :--- | :--- | :-: | :-: |
| 1 | `SVM_all90` | `models/svm_all90_final.joblib` | 90 | B |
| 2 | `SVM_SA52` | `models/svm_sa52_final.joblib` | 52 | B |
| 3 | `SVM_MI52` | `models/svm_mi52_final.joblib` | 52 | B (+ MI dari A) |
| 4 | `SVM_HC48` | `models/hill_climbing/svm_hc_final.joblib` | 48 | B |
| 5 | `SVM_Random52_median` | `models/random_subset/svm_random52_median_final.joblib` | 52 | B |
| 6 | `RF_all90` | `models/rf_all90_final.joblib` | 90 | B |
| 7 | `RF_SA52` | `models/rf_sa52_final.joblib` | 52 | B |
| 8 | `Ensemble_SVM_RF_all90` | `models/ensemble/ensemble_svm_rf_all90_final.joblib` | 90 | B |
| 9 | `Ensemble_SVM_RF_SA52` | `models/ensemble/ensemble_svm_rf_sa52_final.joblib` | 52 | B |
| 10 | `MLP_full90` | `models/mlp_full90.joblib` | 90 | A |
| 11 | `MLP_SA52` | `models/mlp_sa52.joblib` | 52 | A |
| 12 | `MLP_MI52` | `models/mlp_mi52.joblib` | 52 | A |

---

## Poin 3 — Keputusan Jumlah Seed
- **Keputusan:** Kita **menggunakan jumlah seed yang sudah berjalan** tanpa menambah run baru agar evaluasi `test` tidak tertunda, dan **mencatat perbedaan dari rencana awal secara transparan di bagian *Limitations* paper**:
  - SA utama (`P0`): **3 seed** (`42, 123, 2026`; rencana awal 5) — ditambah 12 run di `P1`–`P4` (total **15 run** stabilitas).
  - Hill Climbing (`HC`): **1 seed** (`42`; rencana awal 5).
  - Sweep $\lambda$: **1 seed** (`42` untuk 6 nilai $\lambda$; rencana awal 3).
  - Random subset pembanding: **30 seed** (sesuai rencana).

---

## Poin 4 — $\lambda$ Final dan Alasan Pemilihan dari Hasil Sweep
- **$\lambda$ final:** **`0,015`**.
- **Ringkasan hasil sweep (`results/lambda_sweep/lambda_sweep.csv`, seed 42, versi `peak`):**

| $\lambda$ | $k$ | Reduksi | CV Macro-F1 (Train) | **Val Macro-F1** | Val UAR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0,000` | 48 | 46,7% | 0,5243 | 0,4940 | 0,5039 |
| `0,005` | 52 | 42,2% | 0,5224 | **0,4995** | **0,5078** |
| **`0,015`** | **52** | **42,2%** | **0,5224** | **0,4995** | **0,5078** |
| `0,030` | 48 | 46,7% | 0,5160 | 0,4860 | 0,5039 |
| `0,050` | 51 | 43,3% | 0,5120 | 0,4892 | 0,4922 |
| `0,080` | 36 | 60,0% | 0,5082 | 0,4493 | 0,4531 |

- **Alasan:** $\lambda = 0{,}015$ menghasilkan **Validation Macro-F1 tertinggi (`0,4995`)** sambil memangkas **42,2%** fitur ($90 \to 52$). Tanpa penalti ($\lambda = 0$), performa validation sedikit lebih rendah (`0,4940`). Di sisi lain, penalti lebih besar ($\lambda \ge 0{,}03$) menurunkan CV maupun validation F1, dan pada $\lambda = 0{,}08$ terjadi penurunan tajam ke `0,4493` akibat penghapusan fitur diskriminatif terlalu agresif. Nilai `0,015` dipilih di atas `0,005` (yang memberi subset & skor identik) karena memberi tekanan parsimoni yang lebih jelas terhadap fitur redundan.

---

## Poin 5 — Lokasi Akhir Model
- **Keputusan:** **Tetap di `models/` root** untuk seluruh model utama (`svm_all90_final.joblib`, `svm_sa52_final.joblib`, `svm_mi52_final.joblib`, `rf_all90_final.joblib`, `rf_sa52_final.joblib`, dan `mlp_*.joblib`), sedangkan model pembanding tetap di subfoldernya masing-masing (`models/hill_climbing/`, `models/random_subset/`, `models/ensemble/`).
- `train_final_models.py` sudah diselaraskan agar selalu menulis ke `models/` root sehingga **`app.py` tidak perlu diubah**.

---

## Poin 6 — Klarifikasi RF-52 & Rumus Standar Deviasi (SD)
- **RF-52 tanpa tuning ulang:** **Ya, disengaja.** `RF_SA52` memakai hyperparameter terbaik `RF_all90` (`n_estimators=400, max_depth=None, max_features=0.3, min_samples_split=5`) agar perbandingan 90 vs 52 fitur pada Random Forest bersifat terkontrol (*controlled comparison*), sementara re-tuning difokuskan pada model utama SVM. Silakan dicatat di bagian *Method*.
- **Rumus SD untuk paper:** **Sepakat memakai sample standard deviation (`ddof=1`)** agar konsisten dengan angka Anggota C dan standar pelaporan statistik sampel kecil (mis. 15 run stabilitas: $k = 46{,}33 \pm 5{,}79$; CV F1 = $0{,}5133 \pm 0{,}0176$; Val F1 = $0{,}4734 \pm 0{,}0296$). Nilai `ddof=1` juga sudah ditambahkan di `final_models_manifest.json` (`cv5_trainval_macro_f1_std_sample`).

---

## Poin 7 — Grafik Bahasa Inggris & Bahan Paper Bagian B (Selesai)
- **5 grafik bahasa Inggris (PNG 200 dpi)** sudah dibuat di `progres_anggota_b/figures/sa_en/`:
  1. `baseline_none_vs_peak_en.png`
  2. `sa_main_convergence_en.png`
  3. `sa_main_feature_groups_en.png`
  4. `sa_stability_frequency_en.png`
  5. `lambda_sweep_tradeoff_en.png`
- **Bahan paper bagian B (bahasa Inggris, format IEEE, memakai `ddof=1`)** sudah disiapkan di `docs/bahan_paper_anggota_B.md`.
