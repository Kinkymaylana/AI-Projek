# Langkah 4 — Preprocessing Audio RAVDESS

Penanggung jawab: Anggota A · Versi modul: 1.0 · Dijalankan: 2026-11-21 · config_hash: `d930ff28cd`

## Tujuan

Menyiapkan 1.440 rekaman RAVDESS speech agar siap diekstrak fiturnya, dengan aturan yang sama untuk semua file dan bisa diulang persis.

## Urutan proses

1. Muat audio dan ubah ke mono (rata-rata channel; termasuk 5 file stereo yang kedua channel-nya identik)
2. Resample 48 kHz -> 16000 Hz
3. Trim hening awal dan akhir: `top_db=30`, `frame_length=2048`, `hop_length=512`
4. Normalisasi amplitudo (pilihan): `'none'` atau `'peak'` (puncak -> 0.99)

## Input

| File | Asal |
| --- | --- |
| `data/metadata.csv` | Langkah 1 |
| `data/raw/ravdess_speech/` | Audio asli dari Zenodo (tidak diubah) |
| `data/eda_audio_stats.csv` | Langkah 3 (hanya untuk verifikasi) |

## Output

| File | Isi |
| --- | --- |
| `src/preprocess.py` | Modul preprocessing (sumber tunggal aturan preprocessing) |
| `config.json` -> `preprocessing` | Parameter, versi, config_hash |
| `data/processed/ravdess_16k_trim/Actor_XX/*.wav` | 1440 WAV mono 16000 Hz, float32, sudah trim, **tanpa normalisasi** |
| `data/preprocess_log.csv` | Per file: durasi asli, batas trim, durasi hasil, proporsi dibuang, puncak, energi, faktor normalisasi |
| `figures/preprocess/` | `pre_01_sebelum_sesudah.png`, `pre_02_durasi.png`, `pre_03_energi_none_vs_peak.png`, `preprocess_summary.txt` |

## Cara menjalankan ulang

Buka `notebooks/Langkah4_AnggotaA_Preprocessing.ipynb` di Colab dan jalankan semua sel berurutan. Jika parameter di Sel 2 tidak berubah, pemrosesan dilewati dan log dimuat.

## Cara memakai modul (untuk langkah 5 dan demo)

```python
import sys
sys.path.insert(0, '/content/drive/MyDrive/project07/src')
import preprocess as pp
import soundfile as sf

# a) Audio yang sudah diproses (langkah 5): tinggal pilih normalisasi
y, sr = sf.read('.../data/processed/ravdess_16k_trim/Actor_01/03-01-05-02-01-01-01.wav', dtype='float32')
y_peak = pp.normalize(y, 'peak', 0.99)

# b) File mentah atau rekaman baru (demo): seluruh preprocessing sekaligus
y, sr, info = pp.preprocess_file('rekaman.wav', normalize_method='none')
```

## Verifikasi

8/8 pemeriksaan OK.

- [OK ] Jumlah baris log = jumlah file metadata — 1440 / 1440
- [OK ] Semua file hasil ada — 1440 file
- [OK ] Format 16 kHz, mono, float32 (30 file acak)
- [OK ] Tidak ada NaN / tak hingga (30 file acak)
- [OK ] Durasi terpendek setelah trim >= 0,5 s — 1.15 s
- [OK ] Durasi hasil = durasi bicara EDA — selisih maks 0.0000 s
- [OK ] Proses ulang = file tersimpan (10 file acak) — selisih maks 0.00e+00
- [OK ] Normalisasi 'peak' menghasilkan puncak 0,99 (10 file acak)

## Keputusan dan alasannya

| Keputusan | Alasan |
| --- | --- |
| Audio terproses disimpan satu versi (tanpa normalisasi) | Batas trim tidak berubah saat volume diubah (terbukti di Sel 4), jadi normalisasi puncak cukup dilakukan saat ekstraksi fitur |
| Normalisasi dibandingkan, bukan langsung dipilih | EDA: energi membedakan emosi (selisih ±19 dB) tetapi juga berbeda antaraktor (±18 dB). Fitur diekstrak dua versi; Anggota B memilih berdasarkan macro F1 validation |
| Format float32 | Nilai di atas 1,0 pada 18 file clipping tidak terpotong lagi saat disimpan |
| Semua file dipertahankan | File clipping adalah angry/fearful strong (sifat emosi); 53 file dengan hening hampir nol tetap berisi ucapan utuh |
| Parameter trim sama dengan EDA | Hasil preprocessing bisa dicocokkan dengan analisis langkah 3 |

## Keterbatasan

- Pada 53 file (3,7%; tidak ada di data test) trim tidak membuang bunyi kecil bukan-ucapan di tepi file (klik, noise, dengung), sehingga fitur file tersebut sedikit tercampur.
- Ambang trim relatif terhadap puncak file, sehingga pada rekaman sangat pelan noise latar lebih mudah ikut tersisa.
- Resampling ke 16 kHz membuang frekuensi di atas 8 kHz.

## Riwayat versi

| Versi | Tanggal | Perubahan |
| --- | --- | --- |
| 1.0 | 2026-11-21 | Versi pertama |
