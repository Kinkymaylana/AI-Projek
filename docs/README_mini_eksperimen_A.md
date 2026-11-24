# Mini-eksperimen Anggota A — Minggu 6 (KMP) dan Minggu 9–10 (Fuzzy)

Penanggung jawab: Anggota A · Versi: 1.0 · Dijalankan: 2026-11-24

Latihan mingguan sesuai Rencana Kerja ("dokumentasikan sebagai latihan; tidak wajib masuk metode final"). Hanya data train dan validation yang dipakai.

## A. Minggu 6 — Pencarian pola kontur F0 dengan KMP

Kontur F0 (pyin, 65–500 Hz) diubah ke semitone, dipotong per 50 ms, lalu ditokenisasi menjadi N (naik), T (turun), D (datar; ambang ±0,5 semitone), dan U (tidak bersuara). Pola dari pengulangan ke-1 dicari dengan KMP di pengulangan ke-2 (aktor, kalimat, dan emosi sama) dan di emosi lain (aktor sama). Data: aktor 1 dan 2, kalimat 1 (60 file).

| L (token) | Ditemukan persis, sama emosi | Ditemukan persis, beda emosi | Jarak edit relatif, sama emosi | Jarak edit relatif, beda emosi |
| --- | --- | --- | --- | --- |
| 4.0 | 0.40 | 0.28 | 0.17 | 0.20 |
| 6.0 | 0.10 | 0.05 | 0.25 | 0.28 |
| 8.0 | 0.02 | 0.01 | 0.25 | 0.31 |
| 10.0 | 0.00 | 0.00 | 0.25 | 0.33 |
| 12.0 | 0.00 | 0.00 | 0.26 | 0.36 |

1. Bahkan untuk kalimat, aktor, dan emosi yang sama, pola 4 token (200 ms) hanya ditemukan persis pada 40% kasus, dan pola 12 token pada 0% kasus.
2. Penyebabnya: ucapan manusia tidak pernah persis sama. Sedikit perbedaan tempo menggeser batas potongan 50 ms, perubahan nada di dekat ambang 0,5 semitone berpindah antara N/T dan D, dan frame tak bersuara menyisipkan U di posisi berbeda. Satu token yang berbeda saja sudah membuat KMP gagal.
3. Pola pendek yang masih sering ditemukan juga tidak membedakan emosi: 40% (sama emosi) vs 28% (beda emosi), karena pola pendek naik-turun-datar lazim muncul di banyak ujaran, apa pun emosinya.
4. Pencocokan mendekati lebih masuk akal: pada L = 8, jarak edit relatif ke pengulangan emosi yang sama 0.25 vs 0.31 ke emosi lain. Selisihnya kecil, jadi bentuk kontur saja lemah untuk mengenali emosi.
5. Keputusan: tidak dipakai di metode final. Statistik ringkas F0 (rata-rata, simpangan baku, persentil, rentang) di 90 fitur lebih tahan terhadap variasi ujaran dibanding urutan token persis.

## B. Minggu 9–10 — Fuzzy rule arousal

Masukan: F0 mean (batas per gender) dan RMS dalam dB (`features_none.csv`). Fungsi keanggotaan rendah/sedang/tinggi dengan titik P10/P50/P90 data train; 9 aturan (AND = minimum); keluaran singleton 0/0,5/1 dirata-rata berbobot. Pemetaan arousal: rendah = neutral, calm, sad; sedang = disgust; tinggi = happy, angry, fearful, surprised.

| Data | Batas F0 | Akurasi biner | Akurasi 3 kelas |
| --- | --- | --- | --- |
| train | per gender | 0.784 | 0.627 |
| train | digabung | 0.757 | 0.592 |
| validation | per gender | 0.774 | 0.629 |
| validation | digabung | 0.755 | 0.621 |

1. Pada 4 contoh audio, aturan benar pada 4 contoh.
2. Di validation, aturan membedakan arousal tinggi dan rendah dengan akurasi 77% (3 kelas: 63%). Urutan skor per emosi masuk akal: calm 0.05, neutral 0.07, sad 0.47, disgust 0.58, surprised 0.63, happy 0.70, angry 0.73, fearful 0.77.
3. Tanpa informasi gender, akurasi biner menjadi 76%. Batas F0 per gender penting karena nada laki-laki dan perempuan berbeda sekitar 100 Hz.
4. Kelemahan: arousal hanya satu dimensi. Emosi dengan arousal serupa (misalnya angry vs happy, atau calm vs sad) tidak bisa dibedakan, dan skor sad (rata-rata 0.47) cenderung jatuh di tengah, kemungkinan karena sebagian ujaran sad tetap cukup keras atau bernada tinggi.
5. Keputusan: TIDAK dipakai sebagai fitur atau metode klasifikasi, karena F0 dan RMS sudah ada di 90 fitur dan dipelajari langsung oleh model. DIPAKAI (opsional) sebagai indikator arousal yang mudah dijelaskan untuk demo Anggota C, dengan catatan bahwa batas F0 per gender membutuhkan informasi gender; tanpa itu dipakai varian batas gabungan.

## Output

| File | Isi |
| --- | --- |
| `results/mini_eksperimen_A.json` | Parameter, tabel, contoh, evaluasi, kesimpulan |
| `figures/mini/` | `mini_01_kmp_kontur_f0.png`, `mini_02_fuzzy_keanggotaan.png`, `mini_03_fuzzy_skor_per_emosi.png`, `mini_04_fuzzy_sebaran_f0_rms.png`, `mini_summary.txt` |

## Verifikasi

8/8 pemeriksaan OK.

- [OK] KMP identik dengan pencarian naif (2.000 pasangan acak)
- [OK] Bagian A hanya memakai aktor train (1 dan 2) — 60 file
- [OK] Semua 60 rekaman punya token — panjang 15-36
- [OK] Titik keanggotaan hanya dari data train
- [OK] Data test (aktor 21-24) tidak dipakai
- [OK] Fuzzy diuji pada >= 3 contoh audio (Rencana Kerja) — 4 contoh
- [OK] Skor fuzzy selalu di rentang 0-1
- [OK] Skor arousal emosi tinggi > emosi rendah (sanity check) — 0.71 vs 0.06

## Riwayat versi

| Versi | Tanggal | Perubahan |
| --- | --- | --- |
| 1.0 | 2026-11-24 | Versi pertama |
