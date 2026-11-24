# Proyek 07 — Finding the Best Acoustic Features

**Simulated Annealing-Based Feature Selection for Speech Emotion Classification**

Mini-research mata kuliah Kecerdasan Artifisial. Pertanyaan riset:

> Apakah subset fitur akustik hasil simulated annealing (SA) dapat menyamai performa seluruh fitur, dan apakah subset itu stabil lintas speaker?

| Komponen | Keputusan |
| --- | --- |
| Dataset | RAVDESS bagian speech: 24 aktor, 1.440 file, 8 emosi (CC BY-NC-SA 4.0) |
| Split | Speaker-independent: aktor 1–16 train, 17–20 validation, 21–24 test |
| Fitur | 90 fitur statistik: MFCC, delta MFCC, F0, RMS, ZCR, spektral, contrast, chroma |
| Baseline | Seluruh fitur + SVM (RBF) dan Random Forest |
| Metode utama | SA wrapper feature selection |
| Metrik | Accuracy, macro F1, UAR, confusion matrix, jumlah fitur, waktu |
| Failure experiment | Stabilitas seleksi di 5 partisi speaker; test bernoise SNR 20/10/0 dB |
| Demo | Gradio, 4 tab: Optimasi, Fitur terpilih, Perbandingan, Coba sendiri |

## Pembagian jalur

| Anggota | Jalur | Dokumentasi |
| --- | --- | --- |
| A | Data & Fitur: dataset, EDA, preprocessing, 90 fitur, split, partisi, test bernoise, korelasi | `docs/log_kerja_anggota_A.md`, `docs/dataset_card.md`, `data/README.md` |
| B | Model & Optimasi: baseline, SA, uji stabilitas, model final, inference | `progres_anggota_b/docs/`, `docs/Panduan_Model_Final.md` |
| C | Evaluasi, Demo & Integrasi: `evaluate.py`, analisis stabilitas, audit anti-leakage, evaluasi test, demo | `docs/audit_anti_leakage.md` |

## Instalasi

Diuji dengan **Python 3.13.18** di Windows. Versi pustaka dikunci di `requirements.txt`; scikit-learn harus 1.6.1 agar model tersimpan dapat dimuat tanpa peringatan.

Dengan [uv](https://docs.astral.sh/uv/):

```powershell
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Atau dengan pip biasa (Python 3.13 sudah terpasang):

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Folder `.venv` sudah diabaikan Git. Di VS Code, pilih interpreter `.venv` lewat **Python: Select Interpreter**.

Audio RAVDESS dan ESC-50 tidak disimpan di repo. Tabel fitur sudah tersedia di `data/features/`, jadi audio hanya diperlukan untuk membuat ulang fitur; caranya ada di `data/README.md`.

## Menjalankan

Semua perintah dijalankan dari root repo.

| Tujuan | Perintah | Hasil |
| --- | --- | --- |
| Demo Gradio | `.venv\Scripts\python.exe app.py` | buka http://127.0.0.1:7860 |
| Tes unit | `.venv\Scripts\python.exe -m pytest tests` | 27 tes untuk `evaluate.py`, `stability.py`, `run_test_evaluation.py`, dan `run_noise_evaluation.py` |
| Analisis stabilitas | `.venv\Scripts\python.exe src\run_stability_analysis.py` | `results/stability_*`, `figures/stability/` |
| Audit anti-leakage | `.venv\Scripts\python.exe src\audit_leakage.py` | `results/audit_leakage.json` |
| Uji coba evaluasi test (validation saja, aman) | `.venv\Scripts\python.exe src\run_test_evaluation.py` | `results/test_eval/dryrun_validation.csv` dan status kesiapan |
| Cek kesiapan uji noise (aman) | `.venv\Scripts\python.exe src\run_noise_evaluation.py` | status SIAP / BELUM SIAP |
| Uji noise **sekali** (hanya C, setelah test bersih) | `.venv\Scripts\python.exe src\run_noise_evaluation.py --mode noise --confirm "EVALUASI NOISE SEKALI"` | `results/main_table.csv` (72 baris noise), `results/noise_eval/` |
| Gambar paper dari hasil tersimpan | `.venv\Scripts\python.exe src\plot_test_figures.py` | `figures/test_eval/*_en.png` |
| Evaluasi test **sekali** (hanya C) | `.venv\Scripts\python.exe src\run_test_evaluation.py --mode test --confirm "EVALUASI TEST SEKALI"` | `results/main_table.csv`, `results/test_eval/`, `figures/test_eval/` |
| Membuat ulang fitur dari audio | `.venv\Scripts\python.exe extract_features.py --config config.json` | lihat `docs/README_extract_features.md` |
| Melatih ulang model final | `.venv\Scripts\python.exe train_final_models.py` | `models/*.joblib`, lihat `docs/Panduan_Model_Final.md` |

Tab **Coba sendiri** menerima rekaman mikrofon atau file WAV dan menampilkan probabilitas 8 emosi dari model final SVM 90 fitur dan SVM 52 fitur (subset SA) secara berdampingan.

## Struktur folder

```text
AI-Projek/
├── app.py                    demo Gradio (C)
├── extract_features.py       membuat ulang tabel fitur dari audio (A)
├── config.json               parameter preprocessing dan fitur (A)
├── train_final_models.py     melatih model final di train + validation (B)
├── models/                   model final di train + validation: SVM/RF, ensemble, hill climbing, random subset (B); MLP (A)
├── requirements.txt          versi pustaka terkunci
├── data/                     metadata, fitur, split, partisi, test bernoise, korelasi (A)
├── src/
│   ├── preprocess.py, features.py, data_splits.py, noise.py, redundancy.py   (A)
│   ├── evaluate.py           modul evaluasi bersama: semua angka paper lewat sini (C)
│   ├── stability.py          indeks Nogueira dan pembanding acak (C)
│   ├── run_stability_analysis.py   analisis stabilitas 15 run SA (C)
│   ├── audit_leakage.py      audit anti-leakage (C)
│   ├── run_test_evaluation.py   uji coba di validation, lalu evaluasi test sekali (C)
│   ├── run_noise_evaluation.py  uji noise sekali pada 6 kondisi test bernoise (C)
│   └── plot_test_figures.py  gambar paper dari hasil test tersimpan (C)
├── progres_anggota_b/        baseline, SA, model, inference (B; akan dipindah ke struktur utama)
├── tests/                    tes unit (C)
├── results/                  main_table.csv, ringkasan stabilitas, hasil audit
├── figures/                  gambar EDA, fitur, split, noise, korelasi, stabilitas
├── notebooks/                notebook langkah kerja
└── docs/                     log kerja, dataset card, audit
```

## Aturan anti-leakage

1. **Data test (aktor 21–24) hanya dievaluasi sekali, oleh Anggota C, di akhir.** `evaluate.append_result()` menolak baris test tanpa `allow_test=True` dan menolak metode yang sudah tercatat pada split test yang sama.
2. Tuning, SA, dan pemilihan λ hanya memakai train dan validation.
3. Scaling dilakukan di dalam `Pipeline` dan di-fit pada train saja.
4. Semua angka paper dihitung lewat `src/evaluate.py` dan tercatat di `results/main_table.csv`.
5. Kesimpulan tidak boleh melebihi bukti: jika selisih full vs selected tidak signifikan, tulis "setara dengan X% fitur", bukan "lebih baik".

Status audit terakhir ada di `docs/audit_anti_leakage.md`.

## Status

| Bagian | Status |
| --- | --- |
| Data, fitur, split, partisi, test bernoise, korelasi (A) | Selesai |
| Pembanding mutual information (A, langkah 10) | Selesai |
| Mini-eksperimen MLP, KMP, fuzzy (A) | Selesai |
| Baseline, SA utama, 15 run stabilitas (B) | Selesai |
| Skrip `run_baseline.py`, `run_sa_main.py`, `run_sa_stability.py` (B) | Selesai |
| Model final SVM/RF × 90/52 fitur di train + validation (B) | Selesai |
| Pembanding hill climbing, random subset, sweep λ, ensemble (B) | Selesai (jumlah seed lebih sedikit dari rencana, dicatat sebagai keterbatasan atau ditambah) |
| Model final SVM MI-52 (B) | Selesai |
| `evaluate.py`, audit anti-leakage, analisis stabilitas tanpa label, demo (C) | Selesai; audit 11 OK, 1 WARN, 0 FAIL |
| Skrip evaluasi test (C) | Siap; uji coba validation 12/12 model cocok dengan angka A/B |
| Evaluasi test sekali (C) | Selesai 24 November 2026; `results/main_table.csv`, `results/test_eval/` |
| Uji noise (C) | Skrip siap; belum dijalankan |
| Error analysis, matriks transfer (C) | Berikutnya |
| Mini-paper IEEE dan slide (semua, integrasi C) | Belum |

## Referensi utama

- Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). *PLOS ONE*, 13(5), e0196391.
- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by Simulated Annealing. *Science*, 220(4598), 671–680.
- Nogueira, S., Sechidis, K., & Brown, G. (2018). On the Stability of Feature Selection Algorithms. *JMLR*, 18(174), 1–54.

Daftar lengkap ada di dokumen Referensi dan Dataset Proyek 07.
