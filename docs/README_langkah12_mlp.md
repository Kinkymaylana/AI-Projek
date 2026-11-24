# Langkah 12 — Mini-eksperimen MLP: Fitur Lengkap vs Terpilih

Penanggung jawab: Anggota A · Versi: 1.0 · Dijalankan: 2026-11-24 · config_hash: `d03ab6d497` (fitur `0c3114fc78`)

## Tujuan

Mini-eksperimen deep learning Minggu 12 (Rencana Kerja; tabel ketergantungan no. 6, A → C): membandingkan MLP 2 hidden layer pada semua fitur vs fitur terpilih (SA dan MI), dengan split dan metrik yang sama.

## Protokol

- Data: `features_peak.csv`. Tuning di train (aktor 1–16) dengan `GridSearchCV` + `GroupKFold(4)` per aktor; grid ukuran layer {(64, 32), (128, 64), (256, 128)} × alpha {0,001; 0,01; 0,1; 1; 3}.
- MLP: ReLU, Adam (learning rate 0,001), early stopping (15% data latih, 20 epoch tanpa perbaikan), maksimum 1.000 epoch.
- Penilaian: macro F1 validation (aktor 17–20), rata-rata ± simpangan baku dari 5 seed. Pembanding: SVM RBF (C = 1, gamma = scale, class_weight = balanced).
- Model final: dilatih ulang di train + validation (aktor 1–20) dengan seed 0. Data test tidak dipakai.

## Hasil (validation)

| Subset | k | Layer, alpha | CV macro F1 train | Val macro F1 MLP (5 seed) | Val UAR MLP | Val macro F1 SVM |
| --- | --- | --- | --- | --- | --- | --- |
| Semua 90 fitur | 90 | (256, 128), 1.0 | 0.4682 | 0.4769 ± 0.0261 | 0.4719 | 0.5144 |
| SA (52 fitur) | 52 | (256, 128), 0.01 | 0.4639 | 0.4771 ± 0.0244 | 0.4766 | 0.4995 |
| MI top-52 | 52 | (128, 64), 0.01 | 0.4634 | 0.4523 ± 0.0284 | 0.4500 | 0.4658 |

## Output

| File | Isi |
| --- | --- |
| `results/mlp_results.json` | Parameter, skor per seed, F1 per emosi, lokasi model |
| `results/mlp_summary.csv` | Tabel ringkas untuk benchmark C |
| `models/mlp_full90.joblib`, `mlp_sa52.joblib`, `mlp_mi52.joblib` | Bundle model final: `pipeline`, `label_classes`, `selected_indices`, `params`, `trained_on` |
| `figures/mlp/` | `mlp_01_mlp_vs_svm.png`, `mlp_02_f1_per_emosi.png`, `mlp_summary.txt` |

## Cara memakai (Anggota C)

```python
import joblib
b = joblib.load(BASE_DIR + '/models/mlp_sa52.joblib')
pred_kode = b['pipeline'].predict(X_test[:, b['selected_indices']])   # X_test dari features_peak.csv, 90 kolom fitur
pred_label = [b['label_classes'][i] for i in pred_kode]
```

## Verifikasi

7/7 pemeriksaan OK.

- [OK] Tuning hanya memakai train (aktor 1-16, 960 baris)
- [OK] Setiap lipatan GroupKFold memisahkan aktor (tidak ada aktor di train dan validasi lipatan sekaligus)
- [OK] Data test (aktor 21-24) tidak dipakai di tuning, penilaian, maupun model final
- [OK] Model final dilatih di train + validation (aktor 1-20, 1.200 baris)
- [OK] Setiap subset dinilai dengan 5 seed
- [OK] 3 model final bisa dimuat ulang dan memprediksi label 0-7
- [OK] SVM pembanding mereproduksi angka langkah 10 dan Anggota B — full90 0.5144, sa52 0.4995

## Keterbatasan

- `MLPClassifier` tidak mendukung pembobotan kelas, sehingga kelas neutral (separuh jumlah kelas lain) tidak diberi bobot tambahan, berbeda dengan SVM (`class_weight='balanced'`).
- Early stopping menyisihkan 15% data latih secara acak (bukan per aktor), sehingga penghentiannya sedikit optimis.
- Dengan 960 data latih, MLP mudah overfitting; grid sengaja kecil dan memakai regularisasi alpha.
- Mini-eksperimen ini latihan pendukung, bukan metode utama proyek.

## Riwayat versi

| Versi | Tanggal | Perubahan |
| --- | --- | --- |
| 1.0 | 2026-11-24 | Versi pertama |
