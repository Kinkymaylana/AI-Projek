# extract_features.py — Reproduksi Tabel Fitur

Penanggung jawab: Anggota A · Versi skrip: 1.0 · Diverifikasi: 2026-11-21 · config_hash fitur: `0c3114fc78` (preprocessing `d930ff28cd`)

## Tujuan

Membuat ulang tabel fitur Proyek 07 dari audio asli RAVDESS dengan satu perintah, sesuai syarat Definition of done Anggota A. Skrip menjalankan alur langkah 4 (preprocessing) dan langkah 5 (ekstraksi 90 fitur, dua versi normalisasi) dengan parameter dari `config.json`.

## Prasyarat

- Audio asli di `data/raw/ravdess_speech/Actor_XX/*.wav` (cara mengunduh: `data/README.md` atau notebook langkah 1)
- `config.json`, `src/preprocess.py`, `src/features.py`
- Library: numpy, pandas, librosa, soundfile (versi yang dipakai saat verifikasi: Python 3.13.18, numpy 2.4.6, pandas 2.2.3, librosa 0.11.0, soundfile 0.14.0)

## Cara memakai

Jalankan dari folder `project07/`:

```bash
# buat ulang semua tabel fitur (output ke data/features_repro/)
python extract_features.py --config config.json

# buat ulang dan bandingkan dengan tabel resmi
python extract_features.py --config config.json --compare data/features

# uji cepat untuk aktor tertentu, paralel 2 proses
python extract_features.py --config config.json --actors 1 2 --n-jobs 2 --out-dir data/features_repro_test --compare data/features
```

| Argumen | Default | Keterangan |
| --- | --- | --- |
| `--config` | `config.json` | Lokasi config; folder config dianggap folder utama proyek |
| `--raw-dir` | `data/raw/ravdess_speech` | Folder audio asli |
| `--out-dir` | `data/features_repro` | Folder output; tabel resmi tidak ditimpa |
| `--actors` | semua | Hanya aktor tertentu |
| `--compare` | – | Folder tabel pembanding; menulis `compare_report.json` |
| `--n-jobs` | 1 | Jumlah proses paralel; hasil tetap sama |
| `--rtol`, `--atol` | 1e-6, 1e-9 | Toleransi perbandingan |

Exit code: 0 = sukses (dan identik jika `--compare` dipakai), 1 = ada file gagal atau hasil berbeda, 2 = masalah input.

## Output

| File | Isi |
| --- | --- |
| `features_none.csv`, `features_peak.csv` | `file, actor, gender, emotion, intensity` + 90 fitur |
| `feature_names.csv` | `index, name, group, description, scale_dependent` |
| `features_meta.json` | config_hash, jumlah file, versi library, daftar file gagal |
| `compare_report.json` | Hasil perbandingan (hanya jika `--compare` dipakai) |

## Hasil verifikasi

Perintah `python extract_features.py --config config.json --compare data/features` dijalankan dari audio asli pada 2026-11-21.

- [OK] Skrip selesai dengan exit code 0 — exit code 0
- [OK] Tidak ada file gagal — 1440 berhasil, 0 gagal
- [OK] Jumlah baris sama dengan tabel resmi — 1440 vs 1440
- [OK] config_hash hasil = config.json — features 0c3114fc78, preprocessing d930ff28cd
- [OK] features_none.csv: kolom, urutan file, dan kolom info sama
- [OK] features_none.csv: semua nilai fitur identik — 129600/129600 sel sama persis, selisih maks 0
- [OK] features_peak.csv: kolom, urutan file, dan kolom info sama
- [OK] features_peak.csv: semua nilai fitur identik — 129600/129600 sel sama persis, selisih maks 0
- [OK] feature_names.csv sama

Bukti lengkap: `data/features/repro_check.json`.

## Catatan

- Hasil identik bit per bit hanya dijamin dengan versi library yang sama. Versi lain (terutama librosa dan numpy) dapat menghasilkan selisih pembulatan sangat kecil; gunakan `--compare` untuk memeriksanya.
- Kolom info dibaca dari nama file RAVDESS (`modality-channel-emotion-intensity-statement-repetition-actor`), dan urutan baris mengikuti urutan nama file, sama seperti `metadata.csv`.

## Riwayat versi

| Versi | Tanggal | Perubahan |
| --- | --- | --- |
| 1.0 | 2026-11-21 | Versi pertama |
