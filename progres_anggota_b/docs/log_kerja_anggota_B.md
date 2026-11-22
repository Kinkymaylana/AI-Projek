# Log Kerja Anggota B — Baseline Modeling, Pemilihan Normalisasi, Simulated Annealing Utama, dan Uji Stabilitas (Proyek 07)

## Ringkasan

Seluruh tanggung jawab **Anggota B** telah selesai dijalankan secara lokal (CPU multi-core; tidak memerlukan Google Colab) pada **22 November 2026**, dengan seluruh output sementara disimpan di `E:\College\UB\S2\AI\Projek akhir\Progres Anggota B` (tanpa mengubah repositori `AI-Projek`).

| Tahap | Status | Hasil Utama |
| --- | --- | --- |
| 1. Konfirmasi preprocessing (2a–2d) & review Dataset Card | Siap dikirim | Draf balasan di `docs/balasan_konfirmasi_ke_A_dan_C.md` |
| 2. Baseline 90 fitur (`none` vs `peak`, SVM & RF) | Selesai | **`peak` terpilih**: SVM val macro F1 `0,5165` vs `0,4846` (`+0,0319`); RF `0,4644` vs `0,4316` (`+0,0328`) |
| 3. Simulated Annealing (SA) utama (`P0`, seed 42) | Selesai | **`k = 52` fitur** (reduksi `42,2%`), CV macro F1 naik ke `0,5224`, val macro F1 `0,4995` |
| 4. Uji stabilitas SA (`P0`–`P4` × 3 seed = 15 run) | Selesai | `results/subsets.json` (15 run); `k = 46,3 ± 5,6`; Jaccard fitur `0,3947` → Jaccard klaster **`0,4892`** |
| 5. Notebook & modul reusable | Selesai | `src/baseline.py`, `src/sa_feature_selection.py`, 3 notebook Colab/lokal di `notebooks/` |

**Jaminan anti-kebocoran data (`zero leakage`):**
- Data `test` split utama (**aktor 21–24**) **tidak pernah disentuh** pada tahap mana pun.
- `StandardScaler` selalu diletakkan di dalam `Pipeline` (atau di-fit per fold train pada evaluasi CV SA).
- `class_weight='balanced'` dan metrik **macro F1** digunakan secara konsisten untuk menangani rasio kelas `neutral` (96 vs 192 file).
- Hasil partisi `P1`–`P4` pada Tahap 4 tidak dipakai untuk mengubah normalisasi, hyperparameter, maupun model utama `P0`.

---

## Tahap 2 — Baseline Modeling & Pemilihan Versi Normalisasi

Dijalankan dengan `scripts/run_baseline.py` (modul `src/baseline.py`). Tuning dilakukan menggunakan `GridSearchCV` + `GroupKFold(n_splits=4)` berdasarkan `actor` pada **train (aktor 1–16, 960 file)**, lalu dievaluasi sekali pada **validation (aktor 17–20, 240 file)**.

| Versi | Model | Best Params (GroupKFold Train) | CV Macro F1 (Train) | Val Macro F1 (Aktor 17–20) | Val UAR | Val Accuracy | Waktu (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `none` | SVM | `kernel='rbf', C=5.0, gamma=0.001` | 0,5034 ± 0,0213 | 0,4846 | 0,5039 | 0,4833 | 4,37 |
| `none` | RF | `n_estimators=100, max_depth=10, max_features=0.3, min_samples_split=2` | 0,4517 ± 0,0373 | 0,4316 | 0,4414 | 0,4625 | 41,13 |
| **`peak`** | **SVM** | **`kernel='rbf', C=1.0, gamma=0.005`** | **0,4901 ± 0,0328** | **0,5165** | **0,5352** | **0,5167** | **1,54** |
| **`peak`** | **RF** | `n_estimators=400, max_depth=None, max_features=0.3, min_samples_split=5` | 0,4421 ± 0,0480 | 0,4644 | 0,4688 | 0,4875 | 44,68 |

**Keputusan & Interpretasi:**
- **Versi normalisasi terpilih: `peak`** (`features_peak.csv`).
- Baik SVM maupun RF mengalami peningkatan konsisten ~`+0,032` macro F1 pada aktor validation yang belum pernah dilihat saat memakai normalisasi puncak.
- Hal ini mengonfirmasi temuan EDA Anggota A bahwa selisih volume antaraktor (`18,0 dB`) hampir sebesar selisih antaremosi (`19,1 dB`): pada versi `none`, model cenderung sedikit lebih tinggi di CV train (`0,5034` vs `0,4901`), tetapi generalisasinya turun saat menghadapi speaker baru di validation (`0,4846` vs `0,5165`).

Output: `results/baseline_results.json`, `results/baseline_summary.csv`, `figures/baseline_none_vs_peak.png`.

---

## Tahap 3 — Seleksi Fitur Utama dengan Simulated Annealing (`P0`)

Dijalankan dengan `scripts/run_sa_main.py` (modul `src/sa_feature_selection.py`) pada versi **`peak`** dan split utama **`P0`**.

### Desain & Kalibrasi SA
- **Fungsi objektif:** $J(s) = \text{CV\_Macro\_F1}(s) - \lambda \cdot \frac{|s|}{90}$ dengan $\lambda = 0{,}015$, dievaluasi via 4-fold `GroupKFold` pada train (aktor 1–16).
- **Klasifier di dalam SA:** `SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')`.
  - *Catatan kalibrasi:* Pada percobaan awal dengan `gamma=0.005` tetap (nilai optimal 90 fitur), SVM mengalami *underfitting* saat jumlah fitur menyusut (val F1 turun ke `0,4544`). Mengganti ke `gamma='scale'` membuat skala kernel RBF otomatis beradaptasi dengan jumlah fitur aktif `k`.
- **Jadwal pendinginan:** `n_iter = 500`, `t0 = 0.015`, `t_min = 0.0002`, pendinginan geometrik, operator tetangga adaptif (1–3 bit-flip pada suhu tinggi, 1 bit-flip / swap pada suhu rendah), minimal 5 fitur aktif, dengan *reheating* bila stagnan 80 iterasi.
- Setelah subset terbaik `S*` diperoleh, SVM di-re-tune sekali via `GroupKFold(n_splits=4)` pada train menggunakan `S*` saja (hasil terbaik tetap `kernel='rbf', C=1.0, gamma='scale'`).

### Hasil SA Utama (`P0`, `seed=42`)
- **Jumlah fitur terpilih (`k`):** **52 dari 90 fitur** (reduksi **42,2%**).
- **CV Macro F1 (4-fold train):** naik dari `0,4901` (baseline 90 fitur) menjadi **`0,5224`** (`+0,0323`).
- **Validation Macro F1 (aktor 17–20):**
  - SVM: **`0,4995`** (UAR `0,5078`, Accuracy `0,5125`) vs baseline 90 fitur `0,5165` (`-0,0170`).
  - RF: **`0,4522`** (UAR `0,4648`) vs baseline 90 fitur `0,4644` (`-0,0122`).
- **Waktu komputasi:** `82,10 detik` lokal (`cache_hits = 16`, `accepted_worse = 67`, `reheats_done = 0`).

**Sebaran fitur terpilih per kelompok (`k = 52`):**

| Kelompok | Total | Terpilih | Persentase |
| --- | ---: | ---: | ---: |
| `mfcc` | 26 | 12 | 46,2% |
| `delta_mfcc` | 26 | 17 | 65,4% |
| `prosodi` | 6 | 3 | 50,0% |
| `energi` | 3 | 2 | 66,7% |
| `zcr` | 2 | 2 | 100,0% |
| `spektral` | 8 | 3 | 37,5% |
| `contrast` | 7 | 6 | 85,7% |
| `chroma` | 12 | 7 | 58,3% |

Output: `results/sa_main_results.json`, `results/best_model_p0.joblib`, `figures/sa_main_convergence.png`, `figures/sa_main_feature_groups.png`.

---

## Tahap 4 — Uji Stabilitas SA (5 Partisi `P0`–`P4` × 3 Seed = 15 Run)

Dijalankan dengan `scripts/run_sa_stability.py` secara paralel pada 12 worker CPU lokal (`245,8 detik` / `4,1 menit` untuk 14 run tambahan; `P0_seed42` menggunakan ulang hasil SA utama secara langsung).

| Run ID | Partisi | Seed | Jumlah Fitur (`k`) | Best CV F1 (Train) | Val Macro F1 | Waktu (s) |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| `P0_seed42` *(utama)* | P0 | 42 | 52 | 0,5224 | 0,4995 | 82,10 |
| `P0_seed123` | P0 | 123 | 52 | 0,5000 | 0,5089 | 162,47 |
| `P0_seed2026` | P0 | 2026 | 54 | 0,5077 | 0,4832 | 166,01 |
| `P1_seed42` | P1 | 42 | 44 | 0,5385 | 0,5064 | 160,06 |
| `P1_seed123` | P1 | 123 | 40 | 0,5153 | 0,4580 | 161,12 |
| `P1_seed2026` | P1 | 2026 | 47 | 0,5409 | 0,5442 | 158,17 |
| `P2_seed42` | P2 | 42 | 45 | 0,4842 | 0,4599 | 160,50 |
| `P2_seed123` | P2 | 123 | 45 | 0,4908 | 0,4568 | 163,68 |
| `P2_seed2026` | P2 | 2026 | 58 | 0,4950 | 0,4660 | 163,94 |
| `P3_seed42` | P3 | 42 | 43 | 0,5124 | 0,4565 | 160,62 |
| `P3_seed123` | P3 | 123 | 39 | 0,5274 | 0,4459 | 150,01 |
| `P3_seed2026` | P3 | 2026 | 38 | 0,5340 | 0,4752 | 146,70 |
| `P4_seed42` | P4 | 42 | 50 | 0,5249 | 0,4361 | 157,48 |
| `P4_seed123` | P4 | 123 | 43 | 0,5034 | 0,4588 | 95,08 |
| `P4_seed2026` | P4 | 2026 | 45 | 0,5026 | 0,4462 | 92,89 |

### Temuan Awal Stabilitas (Preview via `src/redundancy.py`)
- **Ukuran subset (`k`):** rata-rata **`46,3 ± 5,6`** (rentang `38–58` fitur).
- **Performa lintas 15 run:** CV Macro F1 **`0,5133 ± 0,0170`**, Validation Macro F1 **`0,4734 ± 0,0286`**.
- **Jaccard per fitur (15 run):** **`0,3947`** (dalam partisi yang sama: `0,4200`).
- **Jaccard per klaster redundan (`rd.cluster_jaccard`, versi `peak`):** naik menjadi **`0,4892`** (`+0,0945`; dalam partisi yang sama: **`0,5084`**).
  - Temuan ini mendukung hipotesis Anggota A di Langkah 11: sebagian ketidakstabilan di level fitur terjadi karena SA bergantian memilih fitur dari kelompok redundan yang sama, dan Jaccard per klaster (`0,4892`) bahkan sedikit melampaui baseline tumpang-tindih aktor train antarpartisi (`0,46`).
- **Fitur paling konsisten terpilih (≥ 80% / ≥ 12 dari 15 run):**
  - `voiced_ratio` (15/15 — 100%)
  - `mfcc03_std`, `mfcc04_std`, `dmfcc04_std` (14/15 — 93,3%)
  - `mfcc01_std`, `dmfcc05_std`, `dmfcc06_std`, `f0_mean`, `chroma06_mean` (12/15 — 80,0%)

Output: `results/subsets.json`, `results/stability_runs_summary.csv`, `results/feature_selection_frequency.csv`, `figures/sa_stability_frequency.png`.

---

## Panduan Pemindahan ke Repositori `AI-Projek` (Saat Siap)

| File di `Progres Anggota B/` | Tujuan di `AI-Projek/` |
| --- | --- |
| `src/baseline.py`, `src/sa_feature_selection.py` | `src/` |
| `notebooks/Langkah_AnggotaB_01_Baseline_Normalisasi.ipynb` s.d. `03` | `notebooks/` |
| `results/baseline_results.json`, `results/baseline_summary.csv`, `results/sa_main_results.json`, `results/subsets.json`, `results/stability_runs_summary.csv`, `results/feature_selection_frequency.csv` | `results/` (atau `data/results/` sesuai kesepakatan dengan C) |
| `results/best_model_p0.joblib` | `models/` atau Google Drive (sesuai `.gitignore`) |
| `figures/baseline_none_vs_peak.png`, `figures/sa_main_*.png`, `figures/sa_stability_frequency.png` | `figures/modeling/` atau `figures/sa/` |
| `docs/log_kerja_anggota_B.md` | `docs/` |
