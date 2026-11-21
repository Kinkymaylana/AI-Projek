# Langkah 5 — Ekstraksi Fitur Akustik

Penanggung jawab: Anggota A · Versi modul: 1.0 · Dijalankan: 2026-11-21 · config_hash: `0c3114fc78` (preprocessing `d930ff28cd`)

## Tujuan

Mengubah 1.440 rekaman hasil preprocessing menjadi tabel 90 fitur akustik dalam dua versi normalisasi, untuk baseline dan simulated annealing.

## Input

| File | Asal |
| --- | --- |
| `data/processed/ravdess_16k_trim/` | Langkah 4 (mono 16 kHz, trim, tanpa normalisasi) |
| `data/preprocess_log.csv`, `data/metadata.csv` | Langkah 4 dan 1 |
| `src/preprocess.py` | Langkah 4 (normalisasi puncak) |

## Output

| File | Isi |
| --- | --- |
| `src/features.py` | Modul ekstraksi fitur |
| `config.json` -> `features` | Parameter, versi, config_hash |
| `data/features/features_none.csv` | 1440 baris: `file, actor, gender, emotion, intensity` + 90 fitur, tanpa normalisasi |
| `data/features/features_peak.csv` | Sama, dengan normalisasi puncak (0,99) |
| `data/features/feature_names.csv` | `index, name, group, description, scale_dependent` |
| `data/features/features_meta.json` | config_hash, waktu pembuatan, jumlah file |
| `figures/features/` | `feat_01_f0_per_emosi_gender.png`, `feat_02_profil_fitur_per_emosi.png`, `features_summary.txt` |

Tabel fitur **belum distandardisasi**. Standardisasi (`StandardScaler`) dilakukan di dalam pipeline model dan di-fit pada data train saja.

## Fitur

| Kelompok | Jumlah | Contoh |
| --- | --- | --- |
| `mfcc` | 26 | `mfcc01_mean`, `mfcc02_mean`, `mfcc03_mean` … |
| `delta_mfcc` | 26 | `dmfcc01_mean`, `dmfcc02_mean`, `dmfcc03_mean` … |
| `prosodi` | 6 | `f0_mean`, `f0_std`, `f0_p05` … |
| `energi` | 3 | `rms_mean`, `rms_std`, `rms_max` |
| `zcr` | 2 | `zcr_mean`, `zcr_std` |
| `spektral` | 8 | `centroid_mean`, `centroid_std`, `bandwidth_mean` … |
| `contrast` | 7 | `contrast1_mean`, `contrast2_mean`, `contrast3_mean` … |
| `chroma` | 12 | `chroma01_mean`, `chroma02_mean`, `chroma03_mean` … |

Frame: `n_fft=512` (32 ms), `hop_length=160` (10 ms). Daftar lengkap ada di `feature_names.csv`.

## Keputusan dan alasannya

| Keputusan | Alasan |
| --- | --- |
| MFCC ke-0 tidak dipakai | Mewakili energi total, sudah tercakup di `rms_*`; perbedaan versi none/peak terkumpul di 3 fitur energi |
| F0 memakai persentil 5 dan 95 | Lebih tahan terhadap kesalahan oktaf daripada nilai minimum dan maksimum |
| File tanpa frame bersuara: statistik F0 = 0 | Menghindari nilai kosong; ditandai oleh `voiced_ratio = 0` |
| F0 dihitung sekali untuk kedua versi | F0 tidak berubah saat volume diubah (diuji di Sel 4) |
| Fitur non-energi dihitung dari salinan sinyal berpuncak 1 | Spectral flatness memakai batas bawah tetap sehingga sedikit bergeser (±7%) pada frame pelan jika volume diubah; normalisasi internal menghilangkan efek ini |
| Dua versi tabel fitur | Keputusan 21 November 2026; Anggota B memilih berdasarkan macro F1 validation |
| Grafik deskriptif hanya aktor 1–16 | Data validation dan test tidak diintip sebelum waktunya |

## Cara memakai (Anggota B dan C)

```python
import pandas as pd
fn = pd.read_csv('.../data/features/features_none.csv')
names = pd.read_csv('.../data/features/feature_names.csv')
X = fn[names['name']].values      # 1440 x 90
y = fn['emotion'].values
groups = fn['actor'].values       # untuk split per aktor / GroupKFold
```

Untuk demo (rekaman baru):

```python
import sys
sys.path.insert(0, '/content/drive/MyDrive/project07/src')
import preprocess as pp, features as ft
y_pre, info = pp.preprocess_signal(y, sr, normalize_method='none')   # atau 'peak'
x = ft.extract_features(y_pre)    # 90 angka, urutan = ft.FEATURE_NAMES
```

## Verifikasi

9/9 pemeriksaan OK.

- [OK ] Kedua tabel berisi semua file — 1440 dan 1440 baris
- [OK ] 90 kolom fitur sesuai feature_names.csv
- [OK ] Tidak ada NaN / tak hingga
- [OK ] Kolom info sama dengan metadata.csv
- [OK ] 87 fitur non-energi identik antara none dan peak — selisih relatif maks 5.6e-04
- [OK ] Rasio energi peak/none = faktor normalisasi — selisih relatif maks 3.4e-07
- [OK ] Ekstraksi ulang 5 file acak = hasil tersimpan — 5 file sama
- [OK ] F0 di 65–500 Hz dan voiced_ratio di 0–1 — F0 mean 78–492 Hz; file tanpa frame bersuara: 5
- [OK ] Tidak ada fitur konstan — 0 fitur konstan

## Keterbatasan

- Chroma dirancang untuk musik; untuk suara bicara informasinya terbatas, dan resolusi frekuensi `n_fft=512` di 16 kHz (31 Hz) kasar untuk nada rendah. Fitur ini tetap disertakan sesuai rencana dan bisa dibuang oleh simulated annealing.
- Estimasi F0 dengan pyin bisa keliru pada suara sangat serak atau berbisik.
- Pada 53 file, segmen hasil trim masih membawa sedikit bunyi bukan-ucapan di tepinya (lihat log langkah 3).

## Riwayat versi

| Versi | Tanggal | Perubahan |
| --- | --- | --- |
| 1.0 | 2026-11-21 | Versi pertama |
