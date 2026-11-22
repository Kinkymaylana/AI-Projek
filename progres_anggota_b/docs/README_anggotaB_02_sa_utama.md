# Tahap 3 Anggota B - Seleksi Fitur Utama dengan Simulated Annealing (P0)

Penanggung jawab: Anggota B · Versi modul: 1.0 · Dijalankan: 2026-11-22

## Tujuan

Mencari subset fitur akustik terbaik dari 90 fitur pada versi normalisasi `peak` dan split utama `P0` menggunakan metaheuristik Simulated Annealing (SA), sekaligus menekan jumlah fitur melalui penalti kompleksitas.

## Input

| File | Asal |
| --- | --- |
| `data/features/features_peak.csv` | Langkah 5 Anggota A (versi terpilih dari Tahap 2) |
| `data/features/feature_names.csv` | Langkah 5 Anggota A (daftar 90 fitur) |
| `data/splits.json`, `src/data_splits.py` | Langkah 7 Anggota A (split utama `P0`) |
| `progres_anggota_b/results/baseline_results.json` | Tahap 2 Anggota B (acuan normalisasi dan baseline SVM) |

## Output

| File | Isi |
| --- | --- |
| `progres_anggota_b/src/sa_feature_selection.py` | Modul SA (`run_simulated_annealing`, `evaluate_subset_cv`, `_propose_neighbor`) |
| `progres_anggota_b/notebooks/Langkah_AnggotaB_02_SA_Utama.ipynb` | Notebook eksekusi SA utama pada `P0` |
| `progres_anggota_b/results/sa_main_results.json` | Indeks & nama 52 fitur terpilih, breakdown per kelompok, dan histori konvergensi |
| `progres_anggota_b/models/best_model_p0.joblib` | Pipeline SVM terbaik yang di-fit pada train `P0` dengan 52 fitur terpilih |
| `progres_anggota_b/figures/sa/sa_main_convergence.png` | Kurva konvergensi fungsi objektif, CV F1, dan jumlah fitur per iterasi |
| `progres_anggota_b/figures/sa/sa_main_feature_groups.png` | Proporsi fitur terpilih per kelompok akustik |

## Desain dan Parameter SA

- Fungsi objektif: `J(s) = mean_GroupKFold_Macro_F1(s) - lambda * (|s| / 90)` dengan `lambda = 0.015`.
- Evaluasi internal: `GroupKFold(n_splits=4)` per aktor pada train `P0` (aktor 1-16); `StandardScaler` di-fit terpisah per fold train (`_precompute_scaled_folds`).
- Klasifier evaluasi: `SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)`.
- Jadwal pendinginan: `n_iter = 500`, `t0 = 0.015`, `t_min = 0.0002`, pendinginan geometrik, `seed = 42`, `min_features = 5`, `stagnation_limit = 80`, `max_reheats = 2`.
- Operator tetangga adaptif: 1-3 bit-flip pada suhu tinggi (`T > 0.30 * t0`), dan 70% 1 bit-flip / 30% swap 1 fitur aktif-nonaktif pada suhu rendah.

## Hasil Utama (P0, seed 42)

| Metrik | Baseline (90 fitur) | Subset SA (52 fitur) | Selisih |
| --- | ---: | ---: | ---: |
| Jumlah fitur (`k`) | 90 | 52 | -38 (-42,22%) |
| CV Macro F1 (4-fold train) | 0,4901 | 0,5224 | +0,0323 |
| Val Macro F1 (SVM, aktor 17-20) | 0,5165 | 0,4995 | -0,0170 |
| Val UAR (SVM) | 0,5352 | 0,5078 | -0,0274 |
| Val Accuracy (SVM) | 0,5167 | 0,5125 | -0,0042 |
| Val Macro F1 (RF pembanding) | 0,4644 | 0,4522 | -0,0122 |

Sebaran 52 fitur terpilih per kelompok: `mfcc` (12/26 = 46,2%), `delta_mfcc` (17/26 = 65,4%), `prosodi` (3/6 = 50,0%), `energi` (2/3 = 66,7%), `zcr` (2/2 = 100,0%), `spektral` (3/8 = 37,5%), `contrast` (6/7 = 85,7%), `chroma` (7/12 = 58,3%).

## Keputusan dan alasannya

| Keputusan | Alasan |
| --- | --- |
| Memakai `gamma='scale'` di dalam SA | Pada kalibrasi awal, `gamma=0.005` tetap membuat SVM underfitting saat `k` menyusut (`val_macro_f1` turun ke 0,4544); `gamma='scale'` menyesuaikan skala RBF terhadap jumlah fitur aktif |
| Menjaga `StandardScaler` di-fit per fold | Kolom-per-kolom scaling bersifat independen, sehingga precompute per fold identik dengan `Pipeline(StandardScaler, SVC)` per subset |
| Menggunakan cache evaluasi (`eval_cached`) | Menghindari pelatihan ulang SVM pada mask biner yang sama (16 cache hit pada run utama) |

## Cara memakai

```python
import sys
sys.path.insert(0, 'progres_anggota_b/src')
import sa_feature_selection as sa

sa_res = sa.run_simulated_annealing(
    X_tr, y_tr, g_tr,
    svm_params={'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
    seed=42, n_iter=500, t0=0.015, t_min=0.0002, lambda_penalty=0.015
)
sel_idx = sa_res['best_indices']
```

## Verifikasi

- [OK] Data test (aktor 21-24) tidak pernah diakses selama optimasi SA maupun evaluasi validation
- [OK] Re-tuning SVM pada subset terpilih (52 fitur) dilakukan hanya pada train `P0`
- [OK] Ukuran subset (`k = 52`) dicatat dan siap dipakai sebagai budget fitur pembanding (misal Mutual Information)

## Keterbatasan

- Terjadi indikasi wrapper overfitting: `CV Macro F1` train naik (+0,0323), tetapi `Val Macro F1` pada aktor baru turun (-0,0170) dibanding baseline 90 fitur.
- `gamma='scale'` membuat skala kernel berbeda dengan baseline 90 fitur yang memakai `gamma=0.005`.

## Riwayat versi

| Versi | Tanggal | Perubahan |
| --- | --- | --- |
| 1.0 | 2026-11-22 | Versi pertama |
