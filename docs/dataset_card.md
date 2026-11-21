# Dataset Card — RAVDESS Speech (Proyek 07)


## Ringkasan

Proyek ini memakai bagian speech audio-only RAVDESS: 1.440 rekaman WAV dari 24 aktor profesional (12 laki-laki, 12 perempuan), berlabel 8 emosi, total 88,8 menit. Audit pada 19–20 November 2026 menunjukkan data lengkap dan labelnya konsisten. Satu-satunya temuan adalah 5 file stereo yang kedua channel-nya identik, sehingga aman dikonversi ke mono. Audio dipreprocessing menjadi mono 16 kHz dan di-trim, lalu diringkas menjadi 90 fitur akustik dalam dua versi normalisasi. Data dibagi per aktor (speaker-independent), dengan 5 partisi tambahan untuk uji stabilitas dan versi bernoise dari data test untuk uji ketahanan. Diperbarui 22 November 2026 dengan hasil langkah 4–11.

## Identitas dan sumber

| Atribut | Keterangan |
| --- | --- |
| Nama | The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) |
| Pembuat | Steven R. Livingstone dan Frank A. Russo |
| Lokasi resmi | [zenodo.org/records/1188976](https://zenodo.org/records/1188976) |
| Bagian yang dipakai | `Audio_Speech_Actors_01-24.zip` (audio-only, speech); bagian video dan song tidak dipakai |
| Lisensi | CC BY-NC-SA 4.0 (non-komersial, wajib atribusi, turunan dibagikan dengan lisensi sama) |
| Cara memperoleh | Diunduh otomatis lewat notebook Langkah 1 (Sel 2) dari URL file Zenodo, lalu integritas zip diperiksa dengan `zipfile.testzip()` |
| Tanggal akses | 19 November 2026 |
| Lokasi penyimpanan proyek | Google Drive bersama: `project07/data/raw/ravdess_speech/` |

**Sitasi wajib**

Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. *PLOS ONE*, 13(5), e0196391. [doi.org/10.1371/journal.pone.0196391](https://doi.org/10.1371/journal.pone.0196391)

## Struktur data dan label

Label RAVDESS ditetapkan oleh pembuat dataset dan disimpan di nama setiap file sebagai kode 7 bagian. Program proyek hanya membaca dan menerjemahkan kode ini; tidak ada label yang dibuat atau ditebak oleh tim.

Contoh: `03-01-05-02-01-01-12.wav` = audio-only, speech, angry, intensitas strong, kalimat 1, pengulangan 1, aktor 12 (perempuan).

| Posisi | Informasi | Kode |
| --- | --- | --- |
| 1 | Modality | 03 = audio-only (semua file yang dipakai) |
| 2 | Vocal channel | 01 = speech (semua file yang dipakai) |
| 3 | Emotion | 01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust, 08 surprised |
| 4 | Intensity | 01 normal, 02 strong; neutral hanya normal |
| 5 | Statement | 01 "Kids are talking by the door", 02 "Dogs are sitting by the door" |
| 6 | Repetition | 01 atau 02 |
| 7 | Actor | 01–24; ganjil = laki-laki, genap = perempuan |

Setiap aktor merekam 60 file: neutral 1 intensitas × 2 kalimat × 2 pengulangan = 4 file, ditambah 7 emosi lain × 2 intensitas × 2 kalimat × 2 pengulangan = 56 file.

**Asal setiap kolom `metadata.csv`**

| Kolom | Asal | Keterangan |
| --- | --- | --- |
| `emotion`, `emotion_code`, `intensity`, `statement`, `repetition`, `actor` | Label pembuat dataset | Dibaca dari nama file |
| `gender` | Aturan pembuat dataset | Diturunkan dari nomor aktor (ganjil/genap) sesuai dokumentasi resmi |
| `duration_s`, `sample_rate`, `channels` | Diukur dari file audio | Dibaca dari header WAV dengan `soundfile.info()` |
| `file` | Lokasi file | Path relatif terhadap folder `ravdess_speech` |

Label emosi adalah emosi yang diminta untuk diperankan aktor, bukan hasil penilaian pendengar. Pembuat dataset juga menyediakan data validasi persepsi pendengar bersama paper PLOS ONE, tetapi data itu tidak dipakai di proyek ini.

## Statistik

Tujuh emosi masing-masing punya 192 file, sedangkan neutral hanya 96 karena tidak direkam dengan intensitas strong. Gender seimbang sempurna di setiap kelas, dan durasi rekaman relatif seragam (2,94–5,27 detik).

**Jumlah file per emosi, intensitas, dan gender**

| Emosi | Normal | Strong | Perempuan | Laki-laki | Total |
| --- | --- | --- | --- | --- | --- |
| angry | 96 | 96 | 96 | 96 | 192 |
| calm | 96 | 96 | 96 | 96 | 192 |
| disgust | 96 | 96 | 96 | 96 | 192 |
| fearful | 96 | 96 | 96 | 96 | 192 |
| happy | 96 | 96 | 96 | 96 | 192 |
| neutral | 96 | 0 | 48 | 48 | 96 |
| sad | 96 | 96 | 96 | 96 | 192 |
| surprised | 96 | 96 | 96 | 96 | 192 |
| **Total** | **768** | **672** | **720** | **720** | **1.440** |

**Durasi per emosi (detik)**

| Emosi | Jumlah | Rata-rata | Min | Maks |
| --- | --- | --- | --- | --- |
| angry | 192 | 3,87 | 3,24 | 5,11 |
| calm | 192 | 3,80 | 2,94 | 4,77 |
| disgust | 192 | 3,94 | 3,14 | 5,27 |
| fearful | 192 | 3,57 | 3,07 | 5,00 |
| happy | 192 | 3,64 | 3,10 | 4,40 |
| neutral | 96 | 3,50 | 3,07 | 4,14 |
| sad | 192 | 3,69 | 3,10 | 4,74 |
| surprised | 192 | 3,49 | 2,97 | 4,64 |
| **Semua** | **1.440** | **3,70** (median 3,67) | **2,94** | **5,27** |

Rata-rata durasi menurut kelompok: perempuan 3,75 s dan laki-laki 3,65 s; intensitas normal 3,59 s dan strong 3,82 s. Total durasi 88,8 menit.

## Audit kualitas data

Dari 13 pemeriksaan di `audit_report.txt` (beberapa digabung di tabel ini), 12 lolos. Satu pemeriksaan menemukan 5 file stereo; temuan ini tidak memengaruhi label dan ditangani dengan konversi ke mono.

| Pemeriksaan | Hasil |
| --- | --- |
| Total file = 1.440 | Lolos |
| Semua file terbaca, tanpa error | Lolos (0 gagal) |
| 24 aktor, masing-masing 60 file | Lolos |
| Gender 12 laki-laki / 12 perempuan | Lolos |
| Semua file audio-only speech (kode 03-01) | Lolos |
| Nomor aktor di nama file = nomor folder | Lolos |
| Neutral = 96 file, hanya intensitas normal | Lolos |
| 7 emosi lain masing-masing 192 file | Lolos |
| Tidak ada nama file ganda | Lolos |
| Sample rate seragam | Lolos (semua 48.000 Hz) |
| Jumlah channel seragam | **Temuan**: 1.435 mono, 5 stereo |

**File stereo**

| File | Aktor | Emosi, intensitas |
| --- | --- | --- |
| `Actor_01/03-01-02-01-01-02-01.wav` | 1 (L) | calm, normal |
| `Actor_01/03-01-08-01-02-02-01.wav` | 1 (L) | surprised, normal |
| `Actor_05/03-01-02-01-02-02-05.wav` | 5 (L) | calm, normal |
| `Actor_20/03-01-03-01-02-01-20.wav` | 20 (P) | happy, normal |
| `Actor_20/03-01-06-01-01-02-20.wav` | 20 (P) | fearful, normal |

**Penanganan:** kelima file dikonversi ke mono dengan merata-ratakan kedua channel (`librosa.load(..., mono=True)`), sama seperti file lain. Tidak ada file yang dibuang.

**Hasil cek kemiripan channel kiri-kanan (20 November 2026):** pada kelima file, selisih maksimum antara channel kiri dan kanan adalah 0,00000 dan korelasinya 1,0000. Artinya kedua channel identik, sehingga file ini secara isi sama dengan mono. Format stereo berasal dari file asli dataset, karena proses proyek hanya mengekstrak (dengan verifikasi checksum CRC) dan membaca file tanpa mengubahnya.

## Preprocessing

Preprocessing sudah dijalankan pada 21 November 2026 (modul `src/preprocess.py`, `config_hash` `d930ff28cd`, 8/8 verifikasi OK): 1.440 file menjadi mono 16 kHz, total durasi turun dari 88,8 menjadi 45,5 menit karena trim. Semua file diproses dengan langkah yang sama, dan parameternya dicatat di `config.json`.

1. Muat audio dan konversi ke mono (`mono=True`), termasuk 5 file stereo.
2. Resample dari 48 kHz ke 16 kHz; rentang frekuensi suara bicara masih tercakup dan komputasi lebih ringan.
3. Potong hening di awal dan akhir (`librosa.effects.trim`, `top_db=30`).
4. Normalisasi amplitudo: **dibandingkan dalam dua versi.** EDA menunjukkan energi adalah petunjuk emosi yang kuat, sedangkan normalisasi puncak bisa menghapus sebagian informasi itu. Karena itu fitur diekstrak dalam dua versi, tanpa normalisasi (`features_none.csv`) dan dengan normalisasi puncak (`features_peak.csv`), lalu dibandingkan pada baseline berdasarkan macro F1 validation bersama Anggota B. Hasil preprocessing: normalisasi puncak menurunkan rentang energi antaremosi dari 19,3 menjadi 1,2 dB dan rentang antaraktor dari 18,5 menjadi 4,5 dB; yang terpengaruh hanya fitur berbasis level absolut (energi RMS dan MFCC ke-0).
5. Standardisasi fitur (`StandardScaler`) **tidak** dilakukan di tahap ini, tetapi di dalam pipeline model dan di-fit hanya pada data train.

Catatan hasil preprocessing:

- **18 file kemungkinan clipping tetap dipakai.** 17 di antaranya angry strong (aktor berteriak), jadi clipping itu bagian dari ekspresi emosinya.
- **Pada 53 file (3,7%), segmen hasil trim masih membawa sedikit bunyi bukan ucapan** (napas, klik, atau noise) di tepinya, karena hening awal atau akhirnya sangat pendek. Ucapan tetap utuh, dan file tetap dipakai.
- Audio terproses disimpan satu versi (tanpa normalisasi, float32), karena trim tidak berubah jika volume diubah. Normalisasi puncak diterapkan saat ekstraksi fitur.

## Fitur akustik

Setiap rekaman diringkas menjadi 90 fitur dengan `src/features.py` (langkah 5, config\_hash `0c3114fc78`). Frame analisis 32 ms dengan pergeseran 10 ms (`n_fft = 512`, `hop_length = 160` pada 16 kHz).

| Kelompok | Jumlah | Fitur |
| --- | --- | --- |
| MFCC | 26 | MFCC ke-1 sampai 13, rata-rata dan simpangan baku (MFCC ke-0 tidak dipakai karena mewakili energi total) |
| Delta MFCC | 26 | Laju perubahan MFCC ke-1 sampai 13, rata-rata dan simpangan baku |
| Prosodi | 6 | F0 dengan pyin (65–500 Hz): rata-rata, simpangan baku, persentil 5 dan 95, rentang, proporsi frame bersuara |
| Energi | 3 | RMS rata-rata, simpangan baku, maksimum |
| ZCR | 2 | Zero-crossing rate rata-rata dan simpangan baku |
| Spektral | 8 | Centroid, bandwidth, rolloff (85%), flatness; rata-rata dan simpangan baku |
| Contrast | 7 | Spectral contrast 7 pita (rata-rata) |
| Chroma | 12 | Chroma 12 kelas nada (rata-rata) |

- **Dua versi tabel:** `features_none.csv` (tanpa normalisasi) dan `features_peak.csv` (normalisasi puncak 0,99). Keduanya berbeda hanya pada 3 fitur energi; 87 fitur lainnya dihitung dari salinan sinyal yang puncaknya dijadikan 1, sehingga tidak bergantung volume.
- **Tidak ada nilai kosong.** Pada 5 file (semuanya laki-laki dengan intensitas strong: aktor 3, 9, dan 21) pyin tidak menemukan frame bersuara. Statistik F0-nya diisi 0 dan ditandai dengan `voiced_ratio = 0`. Ucapannya jelas, tetapi suaranya serak atau berteriak sehingga nadanya tidak teratur.
- **F0 median di data train:** laki-laki 149 Hz, perempuan 259 Hz. Nada naik seiring keterbangkitan emosi, tetapi selisih antargender (sekitar 100 Hz) hampir sebesar selisih antaremosi.
- **Bisa direproduksi:** `python extract_features.py --config config.json` membuat ulang kedua tabel dari audio asli dengan hasil identik bit per bit (diverifikasi 22 November 2026; numpy 2.4.6, pandas 2.2.3, librosa 0.11.0, soundfile 0.14.0).
- **Redundansi (langkah 11, data train):** 39 dari 4.005 pasangan fitur berkorelasi |r| > 0,8. Fitur-fitur itu membentuk 17 kelompok redundan (41 fitur), misalnya ketiga fitur energi (r = 0,96–0,99), serta ZCR, centroid, dan rolloff.

## Split speaker-independent

Data dibagi per aktor, bukan per file, sehingga setiap aktor hanya muncul di satu bagian. Setiap bagian seimbang gender dan memuat kedelapan emosi. Split disimpan di `data/splits.json` (langkah 7, 10/10 verifikasi OK) dan dibaca dengan `src/data_splits.py`.

| Bagian | Aktor | Laki-laki / Perempuan | File | Neutral | Tiap emosi lain | Fungsi |
| --- | --- | --- | --- | --- | --- | --- |
| Train | 1–16 | 8 / 8 | 960 | 64 | 128 | Melatih model |
| Validation | 17–20 | 2 / 2 | 240 | 16 | 32 | Menilai subset fitur di SA dan tuning |
| Test | 21–24 | 2 / 2 | 240 | 16 | 32 | Evaluasi akhir, dipakai sekali |

- **Alasan:** setiap aktor mengucapkan dua kalimat yang sama berulang kali. Jika file dibagi acak, suara aktor yang sama ada di train dan test, sehingga model bisa mengenali orangnya, bukan emosinya, dan skor menjadi terlalu tinggi.
- **File stereo:** aktor 1 dan 5 berada di train, aktor 20 di validation; tidak ada di test.
- Failure experiment: selain split utama, ada 5 partisi speaker di `data/partitions.json` untuk menguji stabilitas seleksi fitur. Aktor dibagi ke 5 kelompok berisi 4 orang (2 L + 2 P): G0 = 21–24, G1 = 17–20, dan G2–G4 diundi dari aktor 1–16 dengan seed 42. Partisi Pk memakai test = Gk, validation = G(k+1), dan train = 16 aktor sisanya. P0 identik dengan split utama, dan setiap aktor paling banyak sekali menjadi test. Partisi berbagi rata-rata 46% aktor train (Jaccard 0,33–0,60).

## Data turunan: test bernoise

Untuk menguji ketahanan terhadap kondisi rekaman yang tidak bersih, dibuat versi bernoise dari test split utama (aktor 21–24, 240 file) pada langkah 9 (config\_hash `dcf3a82fad`). Train dan validation tetap bersih.

| Komponen | Keterangan |
| --- | --- |
| Jenis noise | White noise Gaussian, dan noise ruangan dari ESC-50 (240 klip: rain, wind, vacuum\_cleaner, washing\_machine, engine, keyboard\_typing) |
| SNR | 20, 10, dan 0 dB, terhadap daya ucapan hasil trim; SNR terukur sama persis dengan target |
| Titik penambahan | Setelah preprocessing (trim), sebelum ekstraksi fitur |
| Hasil | 12 tabel `test_noisy_{white,esc50}_{20,10,0}db_{none,peak}.csv` (masing-masing 240 baris) |
| Sumber ESC-50 | github.com/karolpiczak/ESC-50, lisensi CC BY-NC 3.0; klip tidak disimpan di repo |

Sitasi ESC-50: Piczak, K. J. (2015). ESC: Dataset for Environmental Sound Classification. *Proceedings of the 23rd ACM International Conference on Multimedia*, 1015–1018. doi.org/10.1145/2733373.2806390

Pergeseran fitur akibat noise (tanpa model): white noise jauh lebih merusak daripada noise ruangan pada SNR yang sama. Fitur bentuk spektrum, contrast, dan ZCR paling bergeser, sedangkan energi dan MFCC orde rendah paling tahan. Pada white 0 dB, nada tidak terdeteksi di 134 dari 240 file.

## Keterbatasan dan risiko bias

Hasil proyek berlaku untuk emosi yang diperankan dalam kondisi studio, dan tidak bisa langsung digeneralisasi ke percakapan nyata.

| Keterbatasan | Dampak pada proyek | Cara menangani |
| --- | --- | --- |
| Emosi diperankan aktor profesional, bukan spontan | Ekspresi cenderung lebih jelas daripada emosi sehari-hari, sehingga performa bisa lebih tinggi daripada di dunia nyata | Disebut eksplisit di paper; uji lintas dataset (CREMA-D) sebagai opsi |
| Hanya 24 speaker dengan aksen Amerika Utara netral | Keragaman suara terbatas; fitur terpilih mungkin khas speaker tertentu | Split speaker-independent dan uji stabilitas di 5 partisi |
| Hanya 2 kalimat yang sama | Isi kata tidak bisa membantu menebak emosi (kelebihan), tetapi variasi fonetik sangat sempit | Disebut sebagai batas generalisasi |
| Neutral setengah jumlah kelas lain | Model bisa kurang memperhatikan neutral | Macro F1 dan `class_weight='balanced'` |
| Label = emosi yang diminta, bukan yang dipersepsi pendengar | Sebagian rekaman mungkin terdengar ambigu (mis. calm vs neutral) | Dibahas di error analysis |
| Rekaman studio bersih | Model dilatih di rekaman bersih; fitur bentuk spektrum bergeser jauh jika ada noise | 12 tabel test bernoise (white dan ESC-50, SNR 20/10/0 dB) untuk dievaluasi Anggota C |
| Pengulangan kalimat oleh aktor yang sama | Risiko kebocoran jika split per file | Split per aktor |
| Estimasi F0 kurang andal untuk suara serak atau berteriak | 5 file (laki-laki, intensitas strong) tanpa nada terdeteksi; 2 di antaranya di test | F0 diisi 0 dan ditandai voiced\_ratio = 0; dibahas di error analysis |
| Trim menyisakan bunyi bukan ucapan di 53 file | Sedikit menambah noise pada fitur file yang pelan | File tetap dipakai; dicatat sebagai keterbatasan |

## Riwayat versi

| Tanggal | Perubahan |
| --- | --- |
| 20 November 2026 | Draf pertama: identitas, label, statistik, audit, rencana preprocessing dan split |
| 21 November 2026 | Hasil preprocessing dan keputusan dua versi normalisasi |
| 22 November 2026 | Fitur akustik, split dan partisi final, test bernoise (ESC-50), redundansi fitur, keterbatasan tambahan |

## Catatan untuk Anggota A

Bagian ini membantu Anda memahami dan mempertanggungjawabkan dataset card. Hapus atau pindahkan sebelum dokumen dikumpulkan.

**Fungsi setiap bagian**

| Bagian | Menjawab pertanyaan | Dipakai di |
| --- | --- | --- |
| Identitas dan sumber | Data dari mana, boleh dipakai atau tidak, bagaimana mengutipnya | Method: Dataset; daftar pustaka |
| Struktur data dan label | Siapa yang memberi label dan bagaimana membacanya | Method: Dataset |
| Statistik | Seberapa besar dan seimbang datanya | Method: Dataset (tabel ringkas) |
| Audit kualitas | Apakah data sudah diperiksa dan apa temuannya | Method: Preprocessing |
| Preprocessing | Apa yang dilakukan pada audio sebelum fitur diekstrak | Method: Preprocessing |
| Rencana split | Bagaimana mencegah kebocoran data | Method: Experimental protocol |
| Keterbatasan | Batas kesimpulan yang boleh ditarik | Discussion / Conclusion |

**Pertanyaan yang mungkin diajukan dosen**

- *Siapa yang memberi label emosi?* Pembuat RAVDESS; label tersimpan di nama file dan program kami hanya membacanya.
- *Mengapa split per aktor, bukan acak?* Agar model diuji pada suara orang yang belum pernah didengar; split acak membuat skor terlalu tinggi.
- *Mengapa neutral hanya 96 file?* Neutral tidak direkam dengan intensitas strong, sehingga kami memakai macro F1.
- *Apa yang dilakukan pada file stereo?* Dikonversi ke mono dengan merata-ratakan dua channel; labelnya tidak terpengaruh.
- *Mengapa resample ke 16 kHz?* Informasi suara bicara yang relevan masih tercakup, dan ekstraksi fitur serta SA jadi lebih cepat.
- Mengapa fitur diekstrak dua versi? Energi adalah petunjuk emosi yang kuat, tetapi juga dipengaruhi jarak mikrofon dan kebiasaan bicara setiap aktor. Baseline membandingkan keduanya di validation untuk memilih yang lebih baik.
- Bagaimana jika nada tidak terdeteksi? Statistik F0 diisi 0 dan ditandai voiced\_ratio = 0, sehingga tidak ada nilai kosong. Ini terjadi di 5 file.
- Apakah data test bernoise membocorkan test? Tidak. Saat dibuat, tidak ada model dan label yang dipakai; evaluasinya tetap hanya oleh Anggota C, sekali.

**Yang masih perlu dilengkapi**

- [x] Hasil cek stereo (selisih maksimum dan korelasi kiri-kanan) di bagian Audit kualitas data
- [x] Hasil preprocessing, fitur, split final, partisi, dan test bernoise (diperbarui 22 November 2026)
- [ ] Konfirmasi keputusan preprocessing (poin 2a–2d) dari Anggota B
- [ ] Versi normalisasi terpilih (none atau peak) dari hasil baseline Anggota B
- [ ] Bagikan dokumen ini ke Anggota B dan C
- [ ] Hapus atau pindahkan bagian "Catatan untuk Anggota A" sebelum dikumpulkan
