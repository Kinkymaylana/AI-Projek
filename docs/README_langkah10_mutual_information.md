# Langkah 10 — Pembanding Filter Method (Mutual Information)

Penanggung jawab: Anggota A · Versi: 1.0 · Dijalankan: 2026-11-24 · config_hash: `eccc001f3d` (fitur `0c3114fc78`)

## Tujuan

Menyediakan pembanding *filter method* untuk SA: 52 fitur dengan mutual information tertinggi terhadap label emosi, dihitung hanya di data train. Daftar indeks diserahkan ke Anggota B untuk dievaluasi dengan protokol yang sama seperti SA.

## Parameter

| Parameter | Nilai |
| --- | --- |
| Versi tabel fitur | `peak` (terpilih di baseline Anggota B) |
| k | 52 (SA utama Anggota B (P0, seed 42); median 3 seed SA di P0 = 52 (52, 52, 54)) |
| Data | Train split utama, aktor 1–16 (960 baris) |
| Estimator | `sklearn.feature_selection.mutual_info_classif`, `n_neighbors=3`, rata-rata 10 seed (scikit-learn 1.6.1) |

## Output

| File | Isi |
| --- | --- |
| `data/analysis/mi_topk_peak.json` | Daftar 52 indeks dan nama fitur untuk Anggota B (format kontrak `subsets.json`: `run_id`, `method`, `seed`, `lambda`, `partition`, `indices`) |
| `data/analysis/mi_scores_peak.csv` | Skor MI rata-rata dan simpangan baku, peringkat, status top-k, status di subset SA, nomor kelompok redundan |
| `data/analysis/mi_eval_preview_peak.csv` | Pratinjau evaluasi validation |
| `figures/mutual_info/` | `mi_01_skor_per_fitur.png`, `mi_02_kelompok_mi_vs_sa.png`, `mi_summary.txt` |

## Hasil

- Top-52 identik untuk semua seed estimator (Jaccard minimum 1.000).
- Per kelompok: mfcc 17/26, delta_mfcc 10/26, prosodi 6/6, energi 0/3, zcr 0/2, spektral 6/8, contrast 6/7, chroma 7/12.
- Peringkat fitur energi: rms_mean #79, rms_std #68, rms_max #72.
- Redundansi: top-52 mencakup 39 kelompok redundan, dengan 13 fitur kembaran.

### Pratinjau evaluasi di validation (protokol Anggota B)

| Subset | k | Val macro F1 (SVM tetap: RBF, C=1, gamma=scale) | CV macro F1 train (tuning) | Val macro F1 (tuning) | Parameter terbaik |
| --- | --- | --- | --- | --- | --- |
| Semua 90 fitur | 90 | 0.5144 | 0.4901 | 0.5165 | C=1.0, gamma=0.005, kernel=rbf |
| MI top-52 | 52 | 0.4658 | 0.4818 | 0.4205 | C=0.01, kernel=linear |

Angka resmi untuk paper berasal dari evaluasi Anggota B dan fungsi `evaluate()` Anggota C. Data test tidak dipakai di langkah ini.

## Verifikasi

7/7 pemeriksaan OK.

- [OK] MI dihitung hanya di train (aktor 1-16, 960 baris) — (960, 90)
- [OK] Data test (aktor 21-24) tidak ada di data yang dipakai
- [OK] Skor MI lengkap untuk 90 fitur, tanpa NaN, tidak negatif
- [OK] Top-52 berisi 52 indeks unik di rentang 0-89
- [OK] Top-52 stabil terhadap seed estimator (Jaccard min >= 0,9) — min 1.000
- [OK] File untuk B tersimpan dan isinya sama
- [OK] Format mengikuti kontrak subsets.json (run_id, method, seed, lambda, partition, indices)

## Keterbatasan

- MI menilai setiap fitur sendiri-sendiri, sehingga tidak menangkap fitur yang hanya berguna jika dikombinasikan, dan bisa memilih fitur yang saling redundan.
- Estimator MI berbasis k tetangga terdekat mengandung unsur acak; dirata-ratakan atas 10 seed.
- k mengikuti subset SA milik B. Jika median k berubah, jalankan ulang dengan nilai `K` yang baru.

## Riwayat versi

| Versi | Tanggal | Perubahan |
| --- | --- | --- |
| 1.0 | 2026-11-24 | Versi pertama |
