# Langkah 7 — Split Data per Aktor dan Partisi Uji Stabilitas

Penanggung jawab: Anggota A · Versi: 1.0 · Dijalankan: 2026-11-21 · config_hash: `04ad9a2f5b` (fitur `0c3114fc78`)

## Tujuan

Membagi data per aktor (speaker-independent) untuk eksperimen utama, membuat 5 partisi speaker untuk uji stabilitas seleksi fitur, dan memeriksa file tanpa frame bersuara.

## Output

| File | Isi |
| --- | --- |
| `data/splits.json` | Split utama: daftar aktor train, validation, test + jumlah file |
| `data/partitions.json` | Kelompok G0–G4, aktor selalu-train, partisi P0–P4 |
| `src/data_splits.py` | `load_splits`, `load_partitions`, `split_frames`, `split_masks`, `check_split` |
| `config.json` -> `splits` | Seed, rancangan, versi, config_hash |
| `data/unvoiced_check.csv` | Pemeriksaan file dengan `voiced_ratio = 0` |
| `figures/splits/` | `split_01_partisi_aktor.png`, `split_02_file_tanpa_nada.png`, `splits_summary.txt` |

## Split utama

| Bagian | Aktor | File | Gender |
| --- | --- | --- | --- |
| train | 1–16 | 960 | 8 L / 8 P |
| validation | 17–20 | 240 | 2 L / 2 P |
| test | 21–24 | 240 | 2 L / 2 P |

Aturan: **test hanya dipakai sekali oleh Anggota C** untuk evaluasi akhir. Tuning (GroupKFold per aktor) dan SA hanya memakai train dan validation.

## Partisi uji stabilitas

Rancangan siklik: kelompok G0 = aktor 21–24, G1 = 17–20, G2–G4 diundi dari aktor 1–16 dengan seed 42 (masing-masing 2 L + 2 P). Partisi Pk memakai test = Gk, validation = G(k+1 mod 5), train = 16 aktor sisanya. Aktor 1, 8, 11, 14 selalu berada di train.

| Partisi | Test | Validation | Aktor train |
| --- | --- | --- | --- |
| P0 | 21, 22, 23, 24 | 17, 18, 19, 20 | 16 |
| P1 | 17, 18, 19, 20 | 2, 6, 7, 9 | 16 |
| P2 | 2, 6, 7, 9 | 4, 5, 15, 16 | 16 |
| P3 | 4, 5, 15, 16 | 3, 10, 12, 13 | 16 |
| P4 | 3, 10, 12, 13 | 21, 22, 23, 24 | 16 |

- P0 identik dengan split utama, sehingga run SA utama dapat dipakai sebagai salah satu partisi.
- Setiap aktor paling banyak sekali menjadi test dan sekali menjadi validation.
- Jaccard aktor train antarpartisi rata-rata 0.46 (0.33–0.60). Kemiripan subset fitur antarpartisi perlu dibaca relatif terhadap angka ini.
- Partisi hanya dipakai untuk eksperimen stabilitas setelah hasil utama final, **bukan** untuk memilih model utama. Aktor 21–24 ikut menjadi train/validation di P1–P4, jadi hasil partisi tidak boleh dipakai untuk mengubah keputusan yang dievaluasi di test utama.

## Cara memakai

```python
import sys
sys.path.insert(0, '/content/drive/MyDrive/project07/src')
import data_splits as ds
import pandas as pd

BASE_DIR = '/content/drive/MyDrive/project07'
fn = pd.read_csv(BASE_DIR + '/data/features/features_none.csv')
tr, va, te = ds.split_frames(fn, ds.load_splits(BASE_DIR))          # split utama
parts = ds.load_partitions(BASE_DIR)
for p, split in parts.items():                                     # P0..P4
    tr_p, va_p, te_p = ds.split_frames(fn, split)
```

## File tanpa frame bersuara

| File | Emosi | Split utama | RMS (dB) | Frame bernada (pyin longgar) |
| --- | --- | --- | --- | --- |
| `Actor_03/03-01-07-02-01-01-03.wav` | disgust strong | train | -40.2 | 126 |
| `Actor_09/03-01-04-02-01-02-09.wav` | sad strong | train | -53.9 | 14 |
| `Actor_09/03-01-05-02-01-02-09.wav` | angry strong | train | -19.5 | 15 |
| `Actor_21/03-01-07-02-01-01-21.wav` | disgust strong | test | -36.8 | 12 |
| `Actor_21/03-01-08-02-02-01-21.wav` | surprised strong | test | -29.5 | 0 |

Keputusan: file tetap dipakai dan tabel fitur tidak diubah; statistik F0 bernilai 0 dan ditandai oleh `voiced_ratio = 0`. Dicatat sebagai keterbatasan estimasi F0.

## Verifikasi

10/10 pemeriksaan OK.

- [OK] Split utama tidak tumpang tindih, 24 aktor terpakai
- [OK] Ukuran split utama 16/4/4 aktor dan 960/240/240 file — aktor {'train': 16, 'validation': 4, 'test': 4}, file {'train': 960, 'validation': 240, 'test': 240}
- [OK] Gender seimbang di split utama (L = P) — train 8L/8P, validation 2L/2P, test 2L/2P
- [OK] Semua 8 emosi ada dengan jumlah sesuai (neutral 4, lainnya 8 per aktor)
- [OK] 5 partisi valid (16/4/4 aktor, tanpa tumpang tindih, test & validation 2L+2P)
- [OK] Setiap aktor paling banyak sekali jadi test dan sekali jadi validation — 20 aktor test berbeda
- [OK] P0 identik dengan split utama
- [OK] Setiap file masuk tepat satu bagian split utama — 1440 dari 1440 file
- [OK] Test split utama = aktor 21-24 (sesuai Rencana Kerja)
- [OK] Pengundian bisa diulang (seed 42 memberi hasil sama)

## Riwayat versi

| Versi | Tanggal | Perubahan |
| --- | --- | --- |
| 1.0 | 2026-11-21 | Versi pertama |
