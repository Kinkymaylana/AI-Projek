# data/

Folder ini berisi metadata, tabel fitur, pembagian data, dan hasil analisis Proyek 07 (*Finding the Best Acoustic Features*). **Audio tidak disimpan di repo** karena ukurannya besar dan lisensinya tidak mengizinkan distribusi ulang tanpa syarat. Audio dapat diunduh dan diproses ulang dengan langkah di bawah.

Penanggung jawab: Anggota A - Ito (Data & Fitur).

## 1. Dataset yang dipakai

| Dataset | Peran | Lisensi | Sumber |
| --- | --- | --- | --- |
| RAVDESS, bagian speech audio-only (`Audio_Speech_Actors_01-24.zip`, ±208 MB) | Dataset utama: 1.440 rekaman, 24 aktor, 8 emosi | CC BY-NC-SA 4.0 | https://zenodo.org/records/1188976 |
| ESC-50 (240 klip dari 6 kategori) | Sumber noise untuk test bernoise | CC BY-NC 3.0 | https://github.com/karolpiczak/ESC-50 |

Sitasi wajib:

- Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). *PLOS ONE*, 13(5), e0196391. https://doi.org/10.1371/journal.pone.0196391
- Piczak, K. J. (2015). ESC: Dataset for Environmental Sound Classification. *Proc. 23rd ACM International Conference on Multimedia*, 1015–1018. https://doi.org/10.1145/2733373.2806390

Kedua dataset hanya boleh dipakai untuk keperluan non-komersial.

## 2. Cara mendapatkan audio

**RAVDESS:** jalankan `notebooks/Langkah1_AnggotaA_Audit_RAVDESS.ipynb`. Sel 2 mengunduh zip dari Zenodo, memeriksa keutuhannya, lalu mengekstraknya ke `data/raw/ravdess_speech/Actor_01 … Actor_24/`.

Cara manual: unduh `Audio_Speech_Actors_01-24.zip` dari halaman Zenodo di atas, simpan di `data/raw/`, lalu ekstrak isinya ke `data/raw/ravdess_speech/`.

**ESC-50:** jalankan Sel 3 di `notebooks/Langkah9_AnggotaA_Test_Bernoise.ipynb`. Sel ini hanya mengunduh 240 klip yang dipakai (rain, wind, vacuum_cleaner, washing_machine, engine, keyboard_typing) ke `data/raw/esc50/`, tidak seluruh dataset.

## 3. Cara membuat ulang data turunan

| Data | Cara membuat ulang |
| --- | --- |
| `processed/ravdess_16k_trim/` (mono 16 kHz, trim) | `notebooks/Langkah4_AnggotaA_Preprocessing.ipynb` |
| `features/features_none.csv`, `features_peak.csv`, `feature_names.csv` | `python extract_features.py --config config.json --out-dir data/features` (dari folder utama repo), atau notebook langkah 5 |
| `splits.json`, `partitions.json` | `notebooks/Langkah7_AnggotaA_Split_Partisi.ipynb` |
| `features/noisy/` | `notebooks/Langkah9_AnggotaA_Test_Bernoise.ipynb` |
| `analysis/` | `notebooks/Langkah11_AnggotaA_Korelasi_Fitur.ipynb` |

Hasil `extract_features.py` sudah diverifikasi identik bit per bit dengan tabel di repo (bukti: `features/repro_check.json`) pada numpy 2.4.6, pandas 2.2.3, librosa 0.11.0, dan soundfile 0.14.0. Dengan versi library lain, hasilnya bisa berbeda sangat sedikit; periksa dengan opsi `--compare data/features`.

## 4. Isi folder

| File atau folder | Di repo? | Isi |
| --- | --- | --- |
| `metadata.csv` | Ya | 1.440 baris: `file, actor, gender, emotion, emotion_code, intensity, statement, repetition, duration_s, sample_rate, channels` |
| `audit_report.txt` | Ya | Hasil 13 pemeriksaan audit data (langkah 1) |
| `eda_audio_stats.csv` | Ya | Statistik audio per file: puncak, energi, durasi bicara, hening awal dan akhir (langkah 3) |
| `preprocess_log.csv` | Ya | Per file: durasi sebelum dan sesudah trim, puncak, energi, faktor normalisasi (langkah 4) |
| `splits.json` | Ya | Split utama per aktor: train 1–16, validation 17–20, test 21–24 |
| `partitions.json` | Ya | 5 partisi speaker (P0–P4) untuk uji stabilitas; P0 = split utama |
| `unvoiced_check.csv` | Ya | Pemeriksaan 5 file tanpa frame bersuara |
| `features/` | Ya | `features_none.csv`, `features_peak.csv` (1.440 × 95), `feature_names.csv`, `features_meta.json`, `repro_check.json` |
| `features/noisy/` | Ya | 12 tabel `test_noisy_{white,esc50}_{20,10,0}db_{none,peak}.csv`, `noisy_log.csv`, `noisy_meta.json` |
| `analysis/` | Ya | Matriks korelasi, pasangan \|r\| > 0,8, dan kelompok fitur redundan (versi none dan peak) |
| `raw/` | Tidak | Audio asli RAVDESS dan klip ESC-50 |
| `processed/` | Tidak | Audio hasil preprocessing |

## 5. Aturan pemakaian data

- **Data test (aktor 21–24) hanya dipakai sekali oleh Anggota C** untuk evaluasi akhir. Tuning dan simulated annealing memakai train dan validation saja.
- Tabel fitur **belum distandardisasi**. Standardisasi dilakukan di dalam pipeline model dan di-fit pada data train saja.
- Gunakan `src/data_splits.py` untuk memotong data, supaya semua anggota memakai pembagian yang sama.
- Parameter dan `config_hash` setiap tahap tercatat di `config.json`.

Dokumentasi lengkap: `docs/dataset_card.md` dan `docs/README_langkah*.md`.
