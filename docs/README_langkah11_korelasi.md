# Langkah 11 — Analisis Korelasi Fitur

Penanggung jawab: Anggota A · Versi: 1.0 · Dijalankan: 2026-11-21 · config_hash: `63bc4a61b4` (fitur `0c3114fc78`)

## Tujuan

Menghitung korelasi Pearson antarfitur di data train dan menandai pasangan |r| > 0.8, sebagai bahan untuk menjelaskan kenapa subset hasil SA dapat tampak tidak stabil tetapi performanya setara. Hanya data train yang dipakai; label emosi tidak dipakai.

## Output

| File | Isi |
| --- | --- |
| `data/analysis/corr_pearson_train_{none,peak}.csv` | Matriks korelasi 90 × 90 |
| `data/analysis/corr_pairs_r08_{none,peak}.csv` | `feature_a, feature_b, r, abs_r, group_a, group_b, jenis` |
| `data/analysis/corr_clusters_{none,peak}.csv` | `index, name, group, cluster, cluster_size, mean_abs_r_in_cluster, representative, is_representative` |
| `src/redundancy.py` | `load_clusters`, `to_clusters`, `jaccard`, `cluster_jaccard`, `mean_pairwise` |
| `figures/correlation/` | `corr_01_matriks_korelasi.png`, `corr_02_pasangan_per_kelompok.png`, `correlation_summary.txt` |

## Hasil

- Pasangan |r| > 0.8: **39** dari 4005 pasangan (versi none); 45 dari 90 fitur punya minimal satu pasangan.
- Kelompok redundan: **66 kelompok** untuk 90 fitur; 17 kelompok berisi ≥ 2 fitur (41 fitur), 49 fitur berdiri sendiri.
- Versi none dan peak hanya berbeda pada korelasi yang melibatkan 3 fitur energi.

### Kelompok redundan berisi ≥ 2 fitur (versi none)

| Kelompok | Ukuran | Wakil | Kelompok fitur | Anggota |
| --- | --- | --- | --- | --- |
| K1 | 4 | `dmfcc13_std` | delta_mfcc, mfcc | mfcc12_std, mfcc13_std, dmfcc12_std, dmfcc13_std |
| K2 | 3 | `f0_mean` | prosodi | f0_mean, f0_p05, f0_p95 |
| K3 | 3 | `rms_std` | energi | rms_mean, rms_std, rms_max |
| K4 | 3 | `centroid_mean` | spektral, zcr | zcr_mean, centroid_mean, rolloff_mean |
| K5 | 3 | `centroid_std` | spektral, zcr | zcr_std, centroid_std, rolloff_std |
| K6 | 3 | `contrast4_mean` | contrast | contrast4_mean, contrast5_mean, contrast6_mean |
| K7 | 2 | `mfcc02_std` | delta_mfcc, mfcc | mfcc02_std, dmfcc02_std |
| K8 | 2 | `mfcc04_std` | delta_mfcc, mfcc | mfcc04_std, dmfcc04_std |
| K9 | 2 | `mfcc10_std` | delta_mfcc, mfcc | mfcc10_std, dmfcc10_std |
| K10 | 2 | `mfcc11_std` | delta_mfcc, mfcc | mfcc11_std, dmfcc11_std |
| K11 | 2 | `f0_std` | prosodi | f0_std, f0_range |
| K12 | 2 | `flatness_mean` | spektral | flatness_mean, flatness_std |
| K13 | 2 | `chroma01_mean` | chroma | chroma01_mean, chroma12_mean |
| K14 | 2 | `chroma02_mean` | chroma | chroma02_mean, chroma03_mean |
| K15 | 2 | `chroma04_mean` | chroma | chroma04_mean, chroma05_mean |
| K16 | 2 | `chroma07_mean` | chroma | chroma07_mean, chroma08_mean |
| K17 | 2 | `chroma10_mean` | chroma | chroma10_mean, chroma11_mean |

### 15 pasangan dengan |r| tertinggi

| Fitur A | Fitur B | r |
| --- | --- | --- |
| `rms_std` | `rms_max` | 0.987 |
| `rms_mean` | `rms_std` | 0.984 |
| `f0_std` | `f0_range` | 0.972 |
| `rms_mean` | `rms_max` | 0.964 |
| `zcr_std` | `centroid_std` | 0.942 |
| `centroid_mean` | `rolloff_mean` | 0.935 |
| `f0_mean` | `f0_p05` | 0.934 |
| `mfcc13_std` | `dmfcc13_std` | 0.926 |
| `mfcc12_std` | `dmfcc12_std` | 0.915 |
| `zcr_mean` | `centroid_mean` | 0.912 |
| `contrast4_mean` | `contrast5_mean` | 0.908 |
| `chroma02_mean` | `chroma03_mean` | 0.907 |
| `flatness_mean` | `flatness_std` | 0.905 |
| `f0_mean` | `f0_p95` | 0.903 |
| `mfcc11_std` | `dmfcc11_std` | 0.889 |

## Cara memakai (Anggota C: analisis stabilitas)

```python
import sys
sys.path.insert(0, BASE_DIR + '/src')
import redundancy as rd
cl = rd.load_clusters(BASE_DIR, 'none')           # atau 'peak', ikut versi pilihan B
subsets = [...]                                    # daftar indeks fitur per run dari subsets.json
j_fitur = rd.mean_pairwise(subsets, rd.jaccard)
j_kelompok = rd.mean_pairwise(subsets, lambda a, b: rd.cluster_jaccard(a, b, cl))
# j_kelompok jauh lebih tinggi daripada j_fitur -> SA memilih fitur berbeda dari kelompok redundan yang sama
```

## Verifikasi

7/8 pemeriksaan OK.

- [OK] Hanya data train (aktor 1-16, 960 baris) — 960 baris
- [OK] Matriks 90 × 90, simetris, diagonal 1, tanpa NaN
- [OK] Setiap fitur masuk tepat satu kelompok redundan
- [CEK] Sebagian besar pasangan |r| > 0,8 berada dalam kelompok yang sama — 69% dari 39 pasangan
- [OK] Rata-rata |r| setiap anggota ke kelompoknya ≥ 0,8 (sesuai batas potong) — minimum 0.807
- [OK] Versi none dan peak hanya berbeda pada korelasi yang melibatkan fitur energi — selisih maks di luar fitur energi 9.88e-05; di fitur energi 0.747
- [OK] Pemeriksaan Spearman: sebagian besar pasangan Pearson juga tinggi di Spearman — Pearson 39, Spearman 37, sama 36
- [OK] Uji modul redundancy.py lolos — 4/4

## Keterbatasan

- Pearson hanya menangkap hubungan linear; pemeriksaan Spearman dipakai sebagai pembanding.
- Batas |r| > 0.8 dan potongan kelompok di jarak 0,2 adalah konvensi, bukan nilai mutlak.
- Korelasi dihitung di train split utama; di partisi lain nilainya bisa sedikit berbeda.

## Riwayat versi

| Versi | Tanggal | Perubahan |
| --- | --- | --- |
| 1.0 | 2026-11-21 | Versi pertama |
