# Langkah 9 — Test Bernoise

Penanggung jawab: Anggota A · Versi: 1.0 · Dijalankan: 2026-11-21 · config_hash: `dcf3a82fad` (fitur `0c3114fc78`)

## Tujuan

Menyediakan versi bernoise dari test split utama (aktor 21, 22, 23, 24) pada SNR 20, 10, dan 0 dB, untuk menguji ketahanan model dan subset fitur hasil SA (failure experiment tambahan). Tidak ada model yang dilatih atau dievaluasi di langkah ini.

## Rancangan

| Komponen | Pilihan |
| --- | --- |
| Jenis noise | `white` (Gaussian) dan `esc50` (240 klip ESC-50: rain, wind, vacuum_cleaner, washing_machine, engine, keyboard_typing) |
| SNR | 20, 10, 0 dB, terhadap daya ucapan hasil trim |
| Titik penambahan | Setelah preprocessing (trim), sebelum ekstraksi fitur; trim tidak diulang |
| Potongan noise | Sama untuk ketiga SNR pada file yang sama; dipilih deterministik dengan seed 2026 |
| Fitur | 90 fitur dengan `src/features.py`, versi `none` dan `peak` |

## Output

| File | Isi |
| --- | --- |
| `data/features/noisy/test_noisy_{white,esc50}_{20,10,0}db_{none,peak}.csv` | 12 tabel, masing-masing 240 baris, kolom sama dengan `features_none.csv` |
| `data/features/noisy/noisy_log.csv` | Per campuran: file, jenis noise, klip sumber, offset, SNR target dan terukur, faktor pengali |
| `data/features/noisy/noisy_meta.json` | config_hash, jumlah file, daftar tabel |
| `src/noise.py` | `white_noise`, `fit_length`, `mix_at_snr`, `measure_snr`, `clean_noise_clip` |
| `figures/noisy/` | `noisy_01_spektrogram.png`, `noisy_02_distorsi_fitur.png`, `noisy_summary.txt` |

## Cara memakai (Anggota C)

```python
import pandas as pd
names = pd.read_csv(BASE_DIR + '/data/features/feature_names.csv')
te_noisy = pd.read_csv(BASE_DIR + '/data/features/noisy/test_noisy_esc50_10db_none.csv')
X_te_noisy = te_noisy[names['name']].values   # urutan baris sama dengan test bersih
y_te = te_noisy['emotion'].values
# model dilatih di train bersih (dan dipilih di validation bersih), lalu dievaluasi di tabel ini
```

Pakai versi (`none` atau `peak`) yang sama dengan pilihan baseline Anggota B.

## Pergeseran fitur akibat noise

Rata-rata |fitur bernoise − fitur bersih| dibagi simpangan baku fitur di data train (versi none):

| Kelompok | white 20 dB | white 10 dB | white 0 dB | esc50 20 dB | esc50 10 dB | esc50 0 dB |
| --- | --- | --- | --- | --- | --- | --- |
| mfcc | 0.94 | 1.48 | 1.98 | 0.63 | 1.05 | 1.53 |
| delta_mfcc | 0.97 | 1.43 | 1.91 | 0.67 | 1.00 | 1.39 |
| prosodi | 0.10 | 0.34 | 1.78 | 0.17 | 0.48 | 1.46 |
| energi | 0.01 | 0.07 | 0.32 | 0.01 | 0.06 | 0.31 |
| zcr | 1.08 | 2.19 | 4.06 | 0.50 | 1.07 | 1.93 |
| spektral | 3.92 | 6.84 | 9.15 | 0.82 | 1.56 | 2.44 |
| contrast | 3.04 | 3.71 | 4.24 | 0.70 | 1.11 | 1.52 |
| chroma | 0.27 | 0.66 | 1.41 | 0.28 | 0.56 | 1.11 |

## Verifikasi

9/9 pemeriksaan OK.

- [OK] 12 tabel ada, masing-masing 240 baris dengan kolom sama seperti tabel bersih — 12 tabel
- [OK] Hanya aktor test (21-24), urutan file dan kolom info sama dengan tabel bersih
- [OK] Tidak ada NaN atau nilai tak hingga
- [OK] SNR terukur = target (selisih < 0,05 dB) untuk semua campuran — 1440 campuran, selisih maks 0.0000 dB
- [OK] Potongan noise sama untuk ketiga SNR pada setiap file
- [OK] 87 fitur non-energi sama antara versi none dan peak (selisih hanya pembulatan float32) — 99.9992% nilai dalam toleransi 0,1%; selisih relatif maks 0.19%
- [OK] Ekstraksi ulang 3 contoh acak = hasil tersimpan
- [OK] Distorsi fitur makin besar saat SNR makin kecil (20 < 10 < 0 dB) — white: 1.20 / 1.90 / 2.67; esc50: 0.56 / 0.95 / 1.48
- [OK] Klip ESC-50 cukup (minimal 200 klip terpakai, 6 kategori) — 240 klip

## Keterbatasan

- Noise ditambahkan secara digital setelah trim. Di rekaman nyata, noise juga memengaruhi trim (hening di awal dan akhir tidak terbuang), sehingga kondisi nyata bisa lebih berat.
- Noise bersifat aditif saja; tidak ada gema ruangan (reverberation) atau distorsi mikrofon.
- ESC-50 berlisensi CC BY-NC 3.0 (Piczak, 2015); klip tidak disimpan di repo, hanya daftar nama klip di `noisy_log.csv`.

## Riwayat versi

| Versi | Tanggal | Perubahan |
| --- | --- | --- |
| 1.0 | 2026-11-21 | Versi pertama |
