# Tahap 2 Anggota B - Baseline Modeling dan Pemilihan Normalisasi

Penanggung jawab: Anggota B · Versi modul: 1.0 · Dijalankan: 2026-11-22

## Tujuan

Membandingkan performa baseline 90 fitur antara versi normalisasi `none` (tanpa normalisasi) dan `peak` (normalisasi puncak 0,99) menggunakan dua model (SVM dan Random Forest) dengan protokol speaker-independent, untuk menentukan satu versi normalisasi dan satu model baseline utama.

## Input

| File | Asal |
| --- | --- |
| `data/features/features_none.csv` | Langkah 5 Anggota A (1.440 x 95, tanpa normalisasi) |
| `data/features/features_peak.csv` | Langkah 5 Anggota A (1.440 x 95, normalisasi puncak 0,99) |
| `data/features/feature_names.csv` | Langkah 5 Anggota A (daftar 90 nama fitur) |
| `data/splits.json`, `src/data_splits.py` | Langkah 7 Anggota A (split utama per aktor) |

## Output

| File | Isi |
| --- | --- |
| `progres_anggota_b/src/baseline.py` | Modul baseline (`get_svm_search`, `get_rf_search`, `evaluate_model`, `run_baseline_comparison`) |
| `progres_anggota_b/notebooks/Langkah_AnggotaB_01_Baseline_Normalisasi.ipynb` | Notebook eksekusi baseline |
| `progres_anggota_b/results/baseline_results.json` | Hasil lengkap grid search, metrik validation, dan F1 per emosi |
| `progres_anggota_b/results/baseline_summary.csv` | Tabel ringkasan perbandingan 4 kombinasi (`none`/`peak` x `SVM`/`RF`) |
| `progres_anggota_b/figures/sa/baseline_none_vs_peak.png` | Grafik perbandingan macro F1 validation `none` vs `peak` |

## Hasil

Tuning dilakukan dengan `GridSearchCV` + `GroupKFold(n_splits=4)` per aktor pada data train (aktor 1-16, 960 file), lalu dievaluasi sekali pada data validation (aktor 17-20, 240 file). Data test (aktor 21-24) tidak disentuh.

| Versi | Model | Best Params (GroupKFold Train) | CV Macro F1 | Val Macro F1 | Val UAR | Val Accuracy | Waktu (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `none` | SVM | `kernel='rbf', C=5.0, gamma=0.001` | 0,5034 +/- 0,0213 | 0,4846 | 0,5039 | 0,4833 | 4,37 |
| `none` | RF | `n_estimators=100, max_depth=10, max_features=0.3, min_samples_split=2` | 0,4517 +/- 0,0373 | 0,4316 | 0,4414 | 0,4625 | 41,13 |
| `peak` | SVM | `kernel='rbf', C=1.0, gamma=0.005` | 0,4901 +/- 0,0328 | 0,5165 | 0,5352 | 0,5167 | 1,54 |
| `peak` | RF | `n_estimators=400, max_depth=None, max_features=0.3, min_samples_split=5` | 0,4421 +/- 0,0480 | 0,4644 | 0,4688 | 0,4875 | 44,68 |

Keputusan: **normalisasi `peak`** dan **model `SVM`** dipilih sebagai acuan utama untuk tahap Simulated Annealing.

## Keputusan dan alasannya

| Keputusan | Alasan |
| --- | --- |
| Memilih `peak` daripada `none` | Baik SVM (+0,0319) maupun RF (+0,0328) naik konsisten pada aktor validation baru saat memakai `peak`, sejalan dengan temuan EDA bahwa selisih volume antaraktor (18,0 dB) hampir menyamai selisih antaremosi (19,1 dB) |
| Memakai `StandardScaler` di dalam `Pipeline` | Mencegah kebocoran statistik validation/fold ke proses training |
| Memakai `class_weight='balanced'` dan `f1_macro` | Menangani ketidakseimbangan kelas `neutral` (96 file) dibanding 7 emosi lain (masing-masing 192 file) |
| Kriteria seleksi berdasarkan `val_macro_f1` | Mengukur generalisasi ke speaker baru yang belum pernah dilihat saat tuning |

## Cara memakai

```python
import sys
sys.path.insert(0, 'progres_anggota_b/src')
import baseline as bl

results, summary_df, estimators, sel_ver, sel_model = bl.run_baseline_comparison(
    tr_none, va_none, tr_peak, va_peak, feat_names
)
```

## Verifikasi

- [OK] Data test (aktor 21-24) tidak dipakai pada tuning maupun evaluasi baseline
- [OK] StandardScaler selalu berada di dalam Pipeline per fold `GroupKFold(n_splits=4)`
- [OK] Kedua versi (`none`, `peak`) dan kedua model (`SVM`, `RF`) selesai dievaluasi tanpa error

## Keterbatasan

- Dari 90 fitur di `src/features.py`, 87 fitur non-energi sudah dinormalisasi puncak secara internal (`_unit_peak`), sehingga perbedaan `none` vs `peak` berasal dari 3 fitur energi (`rms_mean`, `rms_std`, `rms_max`) dan kombinasi hyperparameter grid yang terpilih.
- Evaluasi validation hanya menggunakan 4 aktor (240 file), dan belum disertai uji signifikansi statistik.

## Riwayat versi

| Versi | Tanggal | Perubahan |
| --- | --- | --- |
| 1.0 | 2026-11-22 | Versi pertama |
