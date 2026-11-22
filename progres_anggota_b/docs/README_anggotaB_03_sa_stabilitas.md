# Tahap 4 Anggota B - Uji Stabilitas Seleksi Fitur SA (5 Partisi x 3 Seed)

Penanggung jawab: Anggota B · Versi modul: 1.0 · Dijalankan: 2026-11-22

## Tujuan

Menguji konsistensi subset fitur yang dipilih Simulated Annealing terhadap perubahan komposisi aktor train (`P0`-`P4`) dan variasi inisialisasi acak (`seed` 42, 123, 2026), serta memeriksa apakah ketidakstabilan di level fitur disebabkan oleh redundansi antarfitur.

## Input

| File | Asal |
| --- | --- |
| `data/features/features_peak.csv` | Langkah 5 Anggota A |
| `data/partitions.json`, `src/data_splits.py` | Langkah 7 Anggota A (5 partisi siklik `P0`-`P4`) |
| `data/analysis/corr_clusters_peak.csv`, `src/redundancy.py` | Langkah 11 Anggota A (kelompok fitur berkorelasi tinggi) |
| `progres_anggota_b/results/sa_main_results.json` | Tahap 3 Anggota B (`P0_seed42` dipakai ulang sebagai salah satu dari 15 run) |

## Output

| File | Isi |
| --- | --- |
| `progres_anggota_b/notebooks/Langkah_AnggotaB_03_SA_Stabilitas.ipynb` | Notebook eksekusi uji stabilitas 15 run |
| `progres_anggota_b/results/subsets.json` | Indeks fitur terpilih dan ringkasan stabilitas dari seluruh 15 run |
| `progres_anggota_b/results/stability_runs_summary.csv` | Tabel `k`, `best_cv_f1`, `val_macro_f1`, dan waktu per run |
| `progres_anggota_b/results/feature_selection_frequency.csv` | Frekuensi terpilihnya masing-masing dari 90 fitur di 15 run |
| `progres_anggota_b/figures/sa/sa_stability_frequency.png` | Visualisasi fitur paling konsisten terpilih lintas 15 run |

## Hasil Lintas 15 Run

| Metrik | Nilai (15 Run) |
| --- | --- |
| Jumlah fitur (`k`) | 46,33 +/- 5,59 (rentang 38-58) |
| CV Macro F1 (train) | 0,5133 +/- 0,0170 |
| Validation Macro F1 | 0,4734 +/- 0,0286 |
| Jaccard per fitur (semua 15 run) | 0,3947 (dalam partisi sama: 0,4200) |
| Jaccard per klaster redundan (semua 15 run) | 0,4892 (dalam partisi sama: 0,5084) |

Fitur paling konsisten terpilih (>= 80% atau >= 12 dari 15 run):
- 15/15 (100,0%): `voiced_ratio`
- 14/15 (93,3%): `mfcc03_std`, `mfcc04_std`, `dmfcc04_std`
- 12/15 (80,0%): `mfcc01_std`, `dmfcc05_std`, `dmfcc06_std`, `f0_mean`, `chroma06_mean`

## Keputusan dan alasannya

| Keputusan | Alasan |
| --- | --- |
| Isolasi hasil `P1`-`P4` | Aktor 21-24 masuk ke train/validation di `P1`-`P4`, sehingga hasilnya hanya untuk analisis stabilitas dan tidak dipakai mengubah model utama `P0` |
| Evaluasi stabilitas berbasis klaster (`src/redundancy.py`) | Jaccard naik dari 0,3947 ke 0,4892 (+0,0945), mendukung hipotesis bahwa sebagian variasi antar-run terjadi karena SA bergantian memilih fitur dari kelompok redundan yang sama |

## Cara memakai

```python
import json, sys
sys.path.insert(0, 'src')
import redundancy as rd

with open('progres_anggota_b/results/subsets.json', encoding='utf-8') as f:
    data = json.load(f)

clusters = rd.load_clusters('.', version='peak')
subsets = [r['indices'] for r in data['runs']]
print('Jaccard fitur  :', rd.mean_pairwise(subsets, rd.jaccard))
print('Jaccard klaster:', rd.mean_pairwise(subsets, lambda a, b: rd.cluster_jaccard(a, b, clusters)))
```

## Verifikasi

- [OK] Run `P0_seed42` identik dengan hasil SA utama pada Tahap 3
- [OK] Seluruh 15 run (`P0`-`P4` x seed 42, 123, 2026) selesai dijalankan dan tercatat di `subsets.json`
- [OK] Hasil `P1`-`P4` tidak dipakai untuk memilih hyperparameter ataupun model utama

## Keterbatasan

- Dalam `data/partitions.json`, 4 aktor (`1, 8, 11, 14`) selalu berada di train pada kelima partisi, sehingga variasi antarpartisi tidak sepenuhnya simetris.
- `voiced_ratio` terpilih 100% pada audio bersih, namun pada `figures/noisy/noisy_summary.txt` termasuk fitur yang paling terdistorsi pada skenario noise `esc50` 0 dB; ketahanan subset ini masih menunggu evaluasi test bernoise oleh Anggota C.

## Riwayat versi

| Versi | Tanggal | Perubahan |
| --- | --- | --- |
| 1.0 | 2026-11-22 | Versi pertama |
