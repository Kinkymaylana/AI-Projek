# Bahan Paper Anggota A — Dataset, Preprocessing, Fitur, dan Related Work

## Cara memakai dokumen ini

Dokumen ini berisi draf siap tempel untuk bagian mini-paper yang menjadi tanggung jawab Anggota A (Ito) menurut Rencana Kerja, ditambah bahan slide dan jejak sumber setiap angka. Paper memakai template konferensi IEEE di Overleaf (6–8 halaman) dan disatukan oleh Anggota C.

| Bagian paper | Isi dari Anggota A | Perkiraan panjang |
| --- | --- | --- |
| II. Related Work | Paragraf tentang fitur akustik untuk pengenalan emosi suara (3 referensi) | 150–200 kata |
| III-A. Dataset | RAVDESS speech, label, statistik, audit, Tabel I | ±150 kata + tabel |
| III-B. Preprocessing | Mono, 16 kHz, trim, dua versi normalisasi | ±120 kata |
| III-C. Ekstraksi fitur | 90 fitur, Tabel II, keputusan desain, reproduksibilitas | ±200 kata + tabel |
| III-D. Data uji stabilitas dan ketahanan (opsional) | Split, partisi, test bernoise, redundansi | ±150 kata; C boleh memindahkannya ke Protocol atau Failure Analysis |

Aturan:

- **Nomor referensi \[n\] di sini sementara.** Di Overleaf, nomor diatur otomatis oleh BibTeX; entri BibTeX ada di bagian Daftar pustaka.
- **Tanda ⟨B⟩ menandai isi yang menunggu Anggota B.** Semua isian ⟨B⟩ sudah dilengkapi pada 24 November 2026 (versi terpilih: `peak`).
- **Setiap angka bisa dilacak ke file sumbernya** (bagian Jejak angka), sesuai aturan Rencana Kerja.
- Gaya bahasa memakai kalimat pasif dan istilah teknis yang lazim di paper berbahasa Indonesia. Istilah asing ditulis miring saat pertama muncul. C bebas menyunting agar seragam dengan bagian lain.

## II. Related Work — Fitur akustik untuk pengenalan emosi suara

**Draf (siap tempel):**

> *Speech emotion recognition* (SER) umumnya dimulai dengan meringkas sinyal suara menjadi fitur akustik, lalu mengklasifikasikannya. Survei-survei SER mengelompokkan fitur tersebut menjadi fitur prosodi (nada dasar F0 dan energi), fitur spektral seperti *mel-frequency cepstral coefficients* (MFCC), serta fitur kualitas suara \[1\], \[2\]. El Ayadi dkk. \[2\] mencatat bahwa belum jelas fitur mana yang paling kuat membedakan emosi, sehingga banyak sistem memakai himpunan fitur yang besar. Untuk mengatasi hal itu, Eyben dkk. \[3\] mengusulkan *Geneva Minimalistic Acoustic Parameter Set* (GeMAPS), yaitu himpunan kecil parameter akustik yang dipilih berdasarkan pertimbangan teoretis dan bukti empiris, agar hasil penelitian afektif lebih mudah dibandingkan. Pendekatan ini menetapkan himpunan fitur secara manual oleh pakar. Penelitian ini mengambil arah berbeda: dari 90 fitur akustik standar, subset terbaik dicari secara otomatis dengan *simulated annealing*, kemudian diperiksa apakah subset itu tetap sama ketika speaker diganti dan tetap andal ketika rekaman diberi noise.

**Catatan untuk Anggota A:**

- &#91;1\] Akçay dan Oğuz (2020) adalah survei SER yang luas (model emosi, dataset, fitur, preprocessing, classifier). Dipakai untuk kalimat pembuka dan pengelompokan jenis fitur.
- &#91;2\] El Ayadi dkk. (2011) adalah survei klasik yang secara eksplisit membahas fitur dan menyatakan belum jelas fitur mana yang paling kuat. Kalimat ini menjadi alasan langsung bagi proyek: mencari fitur terbaik.
- &#91;3\] Eyben dkk. (2016), GeMAPS, adalah rujukan standar himpunan fitur minimalis untuk afek. Ini menjadi pembanding filosofis: GeMAPS memilih fitur secara manual oleh pakar, sedangkan proyek memilih secara otomatis dengan SA.
- Sebelum mengumpulkan, baca minimal abstrak dan bagian fitur dari ketiga paper. Pastikan setiap klaim di atas sesuai dengan isinya. Ringkasan di sini adalah parafrase.

## III-A. Dataset

**Draf (siap tempel):**

> Penelitian ini memakai bagian *speech* audio-only dari *Ryerson Audio-Visual Database of Emotional Speech and Song* (RAVDESS) \[4\]. Dataset ini berisi 1.440 rekaman WAV 48 kHz dari 24 aktor profesional berbahasa Inggris dengan aksen Amerika Utara (12 laki-laki dan 12 perempuan). Setiap aktor mengucapkan dua kalimat netral dalam delapan emosi: *neutral*, *calm*, *happy*, *sad*, *angry*, *fearful*, *disgust*, dan *surprised*. Tujuh emosi direkam pada dua intensitas (normal dan kuat), sedangkan *neutral* hanya pada intensitas normal. Akibatnya, *neutral* berisi 96 rekaman dan setiap emosi lain 192 rekaman (Tabel I). Label emosi adalah emosi yang diminta untuk diperankan, dan dibaca langsung dari nama file. Durasi rekaman 2,94–5,27 detik (rata-rata 3,70 detik; total 88,8 menit). Audit data menunjukkan seluruh file lengkap, terbaca, dan berlabel konsisten. Lima file tersimpan sebagai stereo, tetapi kedua channel-nya identik, sehingga file tersebut dikonversi ke mono tanpa kehilangan informasi. Dataset ini berlisensi CC BY-NC-SA 4.0.

**Tabel I (usulan): Distribusi data RAVDESS speech**

| Emosi | Normal | Kuat | Laki-laki | Perempuan | Total |
| --- | --- | --- | --- | --- | --- |
| neutral | 96 | 0 | 48 | 48 | 96 |
| calm, happy, sad, angry, fearful, disgust, surprised (masing-masing) | 96 | 96 | 96 | 96 | 192 |
| Total | 768 | 672 | 720 | 720 | 1.440 |

**Catatan untuk Anggota A:**

- Kalimat "label dibaca dari nama file" penting, karena dosen bisa bertanya siapa yang memberi label. Jawabannya: pembuat dataset; tim hanya membaca kode di nama file.
- Jika ruang terbatas, Tabel I bisa diganti satu kalimat: "*neutral* 96 rekaman, tujuh emosi lain masing-masing 192 rekaman, seimbang gender."

## III-B. Preprocessing

**Draf (siap tempel):**

> Setiap rekaman diproses dengan langkah yang sama menggunakan librosa \[5\]. Pertama, sinyal dikonversi ke mono dan di-*resample* dari 48 kHz ke 16 kHz, yang masih mencakup rentang frekuensi penting suara bicara. Kedua, hening di awal dan akhir rekaman dipotong dengan ambang 30 dB di bawah puncak sinyal. EDA menunjukkan median 51% isi setiap rekaman adalah hening, sehingga pemotongan ini mencegah fitur didominasi bagian tanpa ucapan; total durasi turun dari 88,8 menjadi 45,5 menit. Ketiga, normalisasi amplitudo diperlakukan sebagai pilihan desain yang diuji. EDA menunjukkan energi membedakan emosi dengan jelas (median energi *angry* 19,1 dB lebih tinggi daripada *calm*), tetapi level suara antaraktor juga berbeda hingga 18,5 dB. Normalisasi puncak menurunkan rentang energi antaremosi menjadi 1,2 dB dan antaraktor menjadi 4,5 dB. Karena itu fitur diekstrak dalam dua versi, yaitu tanpa normalisasi dan dengan normalisasi puncak (0,99), lalu versi terbaik dipilih berdasarkan macro F1 pada data validasi. Versi dengan normalisasi puncak dipilih karena SVM baseline 90 fitur mencapai macro F1 validasi 0,5165, dibanding 0,4846 tanpa normalisasi; Random Forest menunjukkan arah yang sama (0,4644 vs 0,4316). Delapan belas rekaman yang kemungkinan mengalami *clipping* tetap dipakai karena 17 di antaranya adalah *angry* intensitas kuat, sehingga *clipping* merupakan bagian dari ekspresi emosinya.

**Catatan untuk Anggota A:**

- Isian ⟨B⟩ sudah dilengkapi (24 November 2026) dari hasil baseline Anggota B: versi `peak` terpilih (SVM val macro F1 0,5165 vs 0,4846).
- Alasan 16 kHz yang bisa disampaikan saat ditanya: frekuensi Nyquist 8 kHz masih mencakup informasi utama suara bicara, dan komputasi menjadi tiga kali lebih ringan.
- Standardisasi fitur (StandardScaler) bukan bagian A; itu ada di pipeline model milik B dan di-fit hanya pada data train. Jika C ingin menyebutnya, letaknya di bagian Protocol.

## III-C. Ekstraksi fitur

**Draf (siap tempel):**

> Setiap rekaman diringkas menjadi 90 fitur akustik tingkat ujaran (Tabel II), yang mencakup kelompok fitur yang lazim di SER \[1\]–\[3\]. Fitur tingkat frame dihitung dengan jendela 32 ms dan pergeseran 10 ms (`n_fft` = 512, `hop_length` = 160 pada 16 kHz), lalu diringkas dengan statistik rata-rata dan simpangan baku. MFCC \[6\] dihitung dari 40 pita mel; koefisien ke-0 tidak dipakai karena mewakili energi total yang sudah tercakup oleh fitur RMS. Nada dasar F0 diestimasi dengan pYIN \[7\] pada rentang 65–500 Hz, dan diringkas dengan persentil ke-5 dan ke-95 alih-alih nilai minimum dan maksimum agar tahan terhadap kesalahan oktaf. Jika tidak ada frame bersuara, statistik F0 diisi 0 dan ditandai oleh proporsi frame bersuara (`voiced_ratio`) = 0; kasus ini terjadi pada 5 dari 1.440 rekaman. Seluruh fitur non-energi dihitung dari salinan sinyal yang puncaknya dinormalisasi secara internal, sehingga hanya 3 fitur energi yang berbeda antara versi tanpa normalisasi dan versi normalisasi puncak. Fitur belum distandardisasi pada tahap ini; standardisasi dilakukan di dalam *pipeline* model dan di-*fit* hanya pada data latih. Seluruh tabel fitur dapat dibuat ulang dari audio asli dengan satu skrip, dan hasilnya identik bit per bit.

**Tabel II (usulan): Fitur akustik yang diekstrak**

| Kelompok | Jumlah | Fitur |
| --- | --- | --- |
| MFCC | 26 | MFCC 1–13, rata-rata dan simpangan baku |
| Delta MFCC | 26 | ΔMFCC 1–13, rata-rata dan simpangan baku |
| Prosodi | 6 | F0 rata-rata, simpangan baku, P5, P95, rentang (P95 − P5), proporsi frame bersuara |
| Energi | 3 | RMS rata-rata, simpangan baku, maksimum |
| ZCR | 2 | Rata-rata, simpangan baku |
| Spektral | 8 | Centroid, bandwidth, rolloff (85%), flatness; rata-rata dan simpangan baku |
| Contrast | 7 | Spectral contrast 7 pita, rata-rata |
| Chroma | 12 | 12 kelas nada, rata-rata |
| **Total** | **90** |  |

**Kalimat tambahan (opsional, jika ruang cukup):**

> Pada data latih, median F0 adalah 149 Hz untuk aktor laki-laki dan 259 Hz untuk aktor perempuan. F0 meningkat seiring keterbangkitan emosi, tetapi selisih antargender (sekitar 100 Hz) hampir sebesar selisih antaremosi, sehingga fitur nada tidak dapat digunakan secara terpisah dari fitur lain.

**Catatan untuk Anggota A:**

- Kelima rekaman tanpa frame bersuara semuanya laki-laki dengan intensitas kuat (aktor 3, 9, dan 21). Ucapannya jelas, tetapi suaranya serak atau berteriak. Detail ini lebih cocok di Failure Analysis milik C; di bagian A cukup disebut jumlahnya.
- Kalimat "identik bit per bit" bisa diberi catatan kaki: "Diverifikasi dengan `extract_features.py` pada numpy 2.4.6, pandas 2.2.3, librosa 0.11.0."
- Pertanyaan yang mungkin muncul: "Kenapa chroma dipakai untuk suara bicara?" Jawaban: chroma dirancang untuk musik dan informasinya terbatas untuk suara bicara, tetapi tetap disertakan sebagai kandidat. Justru SA yang menentukan apakah fitur itu berguna.

## III-D. Data untuk uji stabilitas dan ketahanan (opsional)

Bagian ini menjelaskan data yang disiapkan A untuk eksperimen B dan C. C boleh memindahkannya ke bagian Experimental Protocol atau Failure Analysis.

**Draf (siap tempel):**

> *Split data.* Data dibagi per aktor agar model selalu diuji pada suara yang belum pernah didengar: aktor 1–16 untuk pelatihan (960 rekaman), 17–20 untuk validasi (240), dan 21–24 untuk pengujian (240). Setiap bagian seimbang gender dan memuat kedelapan emosi. Pembagian per file dihindari karena setiap aktor mengucapkan kalimat yang sama berulang kali, sehingga identitas speaker dapat bocor ke data uji.
>
> *Partisi stabilitas.* Untuk menguji stabilitas seleksi fitur, aktor dibagi ke lima kelompok berisi empat orang (dua laki-laki dan dua perempuan). Pada partisi ke-*k*, kelompok ke-*k* menjadi data uji, kelompok berikutnya menjadi data validasi, dan 16 aktor sisanya menjadi data latih. Partisi pertama identik dengan split utama, dan setiap aktor paling banyak sekali menjadi data uji. Rata-rata indeks Jaccard aktor latih antarpartisi adalah 0,46, sehingga kemiripan subset fitur antarpartisi perlu dibaca relatif terhadap angka ini.
>
> *Data uji bernoise.* Untuk menguji ketahanan, data uji diberi *white noise* Gaussian atau noise ruangan dari ESC-50 \[8\] (240 klip dari enam kategori: hujan, angin, penyedot debu, mesin cuci, mesin, dan ketikan keyboard) pada SNR 20, 10, dan 0 dB, dihitung terhadap daya ucapan. Noise ditambahkan setelah preprocessing dan sebelum ekstraksi fitur, dengan potongan noise yang sama untuk ketiga SNR.
>
> *Redundansi fitur.* Pada data latih, 39 dari 4.005 pasangan fitur berkorelasi |r| > 0,8 dan membentuk 17 kelompok fitur redundan, misalnya ketiga fitur energi (r = 0,96–0,99) serta ZCR, centroid, dan rolloff. Fitur dalam satu kelompok dapat saling menggantikan, sehingga stabilitas seleksi juga diukur pada tingkat kelompok.

**Temuan pendukung untuk Failure Analysis (bahan bagi C, bukan teks A):**

- Tanpa model, *white noise* menggeser fitur jauh lebih besar daripada noise ESC-50 pada SNR yang sama (rata-rata 2,67 vs 1,48 simpangan baku pada 0 dB).
- Fitur bentuk spektrum (flatness, rolloff, bandwidth), contrast, dan ZCR paling rapuh; energi dan MFCC orde rendah paling tahan.
- Pada *white noise* 0 dB, nada tidak terdeteksi pada 134 dari 240 rekaman uji.
- Hipotesis: subset SA yang banyak memakai fitur spektral akan turun paling tajam di data bernoise; top-k mutual information kemungkinan memilih beberapa anggota dari kelompok redundan yang sama.

## Bahan tambahan untuk Results dan Failure Analysis (dari A, untuk B dan C)

Bagian ini bukan draf siap tempel. Isinya temuan langkah 10, 12, dan mini-eksperimen yang bisa dipakai B (Results) dan C (Failure Analysis). Semua angka dari validation (aktor 17–20); angka resmi tetap dari pipeline B dan `evaluate()` C.

**Filter method vs SA (langkah 10).** Top-52 mutual information (dihitung di train dengan estimator k-NN \[10\], \[11\]) memberi val macro F1 0,4658, sedangkan SA pada k yang sama 0,4995 (SVM RBF, C = 1, gamma = scale). Kalimat tawaran:

> Pada jumlah fitur yang sama (k = 52), subset hasil SA mengungguli 52 fitur dengan *mutual information* tertinggi (macro F1 validasi 0,500 vs 0,466). Filter method \[9\] menilai fitur satu per satu sehingga melewatkan fitur energi dan ZCR, yang lemah sendirian tetapi berguna dalam kombinasi, dan memilih lebih banyak fitur yang saling redundan (13 fitur kembaran dalam 39 kelompok, dibanding 8 dalam 44 kelompok pada SA).

Catatan untuk B: pada SA, CV macro F1 di train naik (0,5224) tetapi macro F1 validation turun (0,4995 vs 0,5165 dengan 90 fitur). Pola ini adalah *wrapper overfitting* yang dibahas Kohavi dan John \[12\]: pencarian subset ikut menyesuaikan diri dengan noise data yang dipakai untuk menilai subset. Sitasi \[9\] dan \[12\] bisa dipakai di paragraf yang membahasnya.

**MLP vs SVM (langkah 12).** Val macro F1 MLP dua hidden layer (rata-rata 5 seed): 90 fitur 0,477; SA 0,477; MI 0,452. SVM: 0,514; 0,499; 0,466. Pada data sekecil ini (960 baris latih), MLP tidak lebih baik daripada SVM, dan subset SA setara dengan semua fitur untuk MLP.

**Pemetaan kelompok fitur ke fenomena (peta pengetahuan).**

| Kelompok fitur | Fenomena akustik | Emosi yang paling terbantu | Bukti dari pekerjaan A |
| --- | --- | --- | --- |
| Prosodi (F0) | Tinggi nada dan variasinya | Arousal tinggi (angry, fearful, happy, surprised) vs rendah | 6/6 fitur F0 di top-52 MI; `f0_mean` peringkat MI #2 |
| Energi (RMS) | Keras-pelan | Arousal; sebagian besar hilang setelah normalisasi puncak | MI rendah (#68–#79), tetapi tetap dipilih SA (2/3) |
| MFCC dan delta MFCC | Bentuk saluran suara dan dinamikanya | Membedakan emosi dengan arousal sama (misalnya angry vs happy) | Fitur MI #1 `dmfcc04_std`; 27 dari 52 top MI |
| Spektral, contrast, ZCR | Kecerahan dan kekasaran suara | Angry, disgust | Paling rapuh terhadap noise (langkah 9) |

**Indikator arousal fuzzy (mini-eksperimen Minggu 9–10).** Sembilan aturan dari F0 dan RMS memisahkan arousal tinggi vs rendah dengan akurasi 77% di validation. Ini cocok sebagai contoh *interpretability* di Discussion, bukan sebagai metode utama.

**KMP (mini-eksperimen Minggu 6).** Pencocokan persis pola kontur F0 hampir tidak pernah berhasil untuk pola ≥ 8 token (2% sama emosi vs 0,5% beda emosi pada 8 token), sehingga tidak dipakai. Ini bahan untuk menjelaskan kenapa statistik ringkas lebih tahan terhadap variasi ujaran.

## Tabel dan gambar usulan

Paper 6–8 halaman hanya muat sedikit gambar. Prioritas 1 sebaiknya masuk; prioritas 2 dipakai jika ada ruang atau dipindah ke slide.

| Prioritas | Elemen | Isi | File sumber |
| --- | --- | --- | --- |
| 1 | Tabel I | Distribusi data per emosi, intensitas, dan gender | `data/metadata.csv`; tabel di bagian III-A |
| 1 | Tabel II | 90 fitur per kelompok | `data/features/feature_names.csv` |
| 2 | Gambar | Waveform dan mel-spectrogram satu contoh per emosi (aktor yang sama) | `figures/eda/eda_04_contoh_aktor01.png` |
| 2 | Gambar | Energi per emosi, tanpa dan dengan normalisasi puncak | `figures/preprocess/pre_03_energi_none_vs_peak.png` |
| 2 | Gambar | F0 per emosi dan gender | `figures/features/feat_01_f0_per_emosi_gender.png` |
| 2 (untuk C) | Gambar | Peran aktor di 5 partisi | `figures/splits/split_01_partisi_aktor.png` |
| 2 (untuk C) | Gambar | Pergeseran fitur akibat noise per kelompok | `figures/noisy/noisy_02_distorsi_fitur.png` |
| 2 (untuk C) | Gambar | Matriks korelasi antarfitur | `figures/correlation/corr_01_matriks_korelasi.png` |

Catatan format: semua grafik berjudul bahasa Indonesia dan disimpan sebagai PNG resolusi tinggi. Untuk paper IEEE dua kolom, gambar selebar satu kolom sebaiknya dibuat ulang lebih ringkas (judul pindah ke caption) jika C memintanya.

## Daftar pustaka bagian A (format IEEE)

&#91;1\] M. B. Akçay and K. Oğuz, "Speech emotion recognition: Emotional models, databases, features, preprocessing methods, supporting modalities, and classifiers," *Speech Communication*, vol. 116, pp. 56–76, 2020, doi: 10.1016/j.specom.2019.12.001.

&#91;2\] M. El Ayadi, M. S. Kamel, and F. Karray, "Survey on speech emotion recognition: Features, classification schemes, and databases," *Pattern Recognition*, vol. 44, no. 3, pp. 572–587, 2011, doi: 10.1016/j.patcog.2010.09.020.

&#91;3\] F. Eyben *et al.*, "The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for voice research and affective computing," *IEEE Transactions on Affective Computing*, vol. 7, no. 2, pp. 190–202, 2016.

&#91;4\] S. R. Livingstone and F. A. Russo, "The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English," *PLOS ONE*, vol. 13, no. 5, e0196391, 2018, doi: 10.1371/journal.pone.0196391.

&#91;5\] B. McFee *et al.*, "librosa: Audio and music signal analysis in Python," in *Proc. 14th Python in Science Conf. (SciPy)*, 2015, pp. 18–24.

&#91;6\] S. B. Davis and P. Mermelstein, "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 28, no. 4, pp. 357–366, 1980, doi: 10.1109/TASSP.1980.1163420.

&#91;7\] M. Mauch and S. Dixon, "pYIN: A fundamental frequency estimator using probabilistic threshold distributions," in *Proc. IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP)*, 2014, pp. 659–663, doi: 10.1109/ICASSP.2014.6853678.

&#91;8\] K. J. Piczak, "ESC: Dataset for environmental sound classification," in *Proc. 23rd ACM Int. Conf. Multimedia*, 2015, pp. 1015–1018, doi: 10.1145/2733373.2806390.

&#91;9\] I. Guyon and A. Elisseeff, "An introduction to variable and feature selection," *J. Mach. Learn. Res.*, vol. 3, pp. 1157–1182, 2003.

&#91;10\] A. Kraskov, H. Stögbauer, and P. Grassberger, "Estimating mutual information," *Phys. Rev. E*, vol. 69, no. 6, p. 066138, 2004, doi: 10.1103/PhysRevE.69.066138.

&#91;11\] B. C. Ross, "Mutual information between discrete and continuous data sets," *PLOS ONE*, vol. 9, no. 2, p. e87357, 2014, doi: 10.1371/journal.pone.0087357.

&#91;12\] R. Kohavi and G. H. John, "Wrappers for feature subset selection," *Artif. Intell.*, vol. 97, no. 1–2, pp. 273–324, 1997, doi: 10.1016/S0004-3702(97)00043-X.

Untuk Overleaf, salin entri berikut ke file `.bib` bersama, lalu sitasi dengan `\cite{akcay2020}` dan seterusnya:

```bibtex
@article{akcay2020,
  author  = {Ak{\c{c}}ay, Mehmet Berkehan and O{\u{g}}uz, Kaya},
  title   = {Speech emotion recognition: Emotional models, databases, features, preprocessing methods, supporting modalities, and classifiers},
  journal = {Speech Communication}, volume = {116}, pages = {56--76}, year = {2020},
  doi     = {10.1016/j.specom.2019.12.001}}

@article{elayadi2011,
  author  = {El Ayadi, Moataz and Kamel, Mohamed S. and Karray, Fakhri},
  title   = {Survey on speech emotion recognition: Features, classification schemes, and databases},
  journal = {Pattern Recognition}, volume = {44}, number = {3}, pages = {572--587}, year = {2011},
  doi     = {10.1016/j.patcog.2010.09.020}}

@article{eyben2016gemaps,
  author  = {Eyben, Florian and Scherer, Klaus R. and Schuller, Bj{\"o}rn W. and others},
  title   = {The {Geneva} Minimalistic Acoustic Parameter Set ({GeMAPS}) for Voice Research and Affective Computing},
  journal = {IEEE Transactions on Affective Computing}, volume = {7}, number = {2}, pages = {190--202}, year = {2016}}

@article{livingstone2018ravdess,
  author  = {Livingstone, Steven R. and Russo, Frank A.},
  title   = {The {Ryerson} Audio-Visual Database of Emotional Speech and Song ({RAVDESS}): A dynamic, multimodal set of facial and vocal expressions in {North American English}},
  journal = {PLOS ONE}, volume = {13}, number = {5}, pages = {e0196391}, year = {2018},
  doi     = {10.1371/journal.pone.0196391}}

@inproceedings{mcfee2015librosa,
  author    = {McFee, Brian and Raffel, Colin and Liang, Dawen and others},
  title     = {librosa: Audio and Music Signal Analysis in {Python}},
  booktitle = {Proc. 14th Python in Science Conference (SciPy)}, pages = {18--24}, year = {2015}}

@article{davis1980mfcc,
  author  = {Davis, Steven B. and Mermelstein, Paul},
  title   = {Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences},
  journal = {IEEE Transactions on Acoustics, Speech, and Signal Processing}, volume = {28}, number = {4}, pages = {357--366}, year = {1980},
  doi     = {10.1109/TASSP.1980.1163420}}

@inproceedings{mauch2014pyin,
  author    = {Mauch, Matthias and Dixon, Simon},
  title     = {{pYIN}: A fundamental frequency estimator using probabilistic threshold distributions},
  booktitle = {Proc. IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP)}, pages = {659--663}, year = {2014},
  doi       = {10.1109/ICASSP.2014.6853678}}

@inproceedings{piczak2015esc,
  author    = {Piczak, Karol J.},
  title     = {{ESC}: Dataset for Environmental Sound Classification},
  booktitle = {Proc. 23rd ACM International Conference on Multimedia}, pages = {1015--1018}, year = {2015},
  doi       = {10.1145/2733373.2806390}}

@article{guyon2003featsel,
  author  = {Guyon, Isabelle and Elisseeff, Andr{\'e}},
  title   = {An Introduction to Variable and Feature Selection},
  journal = {Journal of Machine Learning Research}, volume = {3}, pages = {1157--1182}, year = {2003}}

@article{kraskov2004mi,
  author  = {Kraskov, Alexander and St{\"o}gbauer, Harald and Grassberger, Peter},
  title   = {Estimating mutual information},
  journal = {Physical Review E}, volume = {69}, number = {6}, pages = {066138}, year = {2004},
  doi     = {10.1103/PhysRevE.69.066138}}

@article{ross2014mi,
  author  = {Ross, Brian C.},
  title   = {Mutual Information between Discrete and Continuous Data Sets},
  journal = {PLOS ONE}, volume = {9}, number = {2}, pages = {e87357}, year = {2014},
  doi     = {10.1371/journal.pone.0087357}}

@article{kohavi1997wrappers,
  author  = {Kohavi, Ron and John, George H.},
  title   = {Wrappers for feature subset selection},
  journal = {Artificial Intelligence}, volume = {97}, number = {1--2}, pages = {273--324}, year = {1997},
  doi     = {10.1016/S0004-3702(97)00043-X}}
```

Catatan: \[1\]–\[3\] adalah tiga referensi Related Work yang diminta Rencana Kerja. \[4\]–\[8\] adalah sitasi metode dan data yang wajib ada karena dataset, library, dan algoritmanya dipakai. \[9\]–\[12\] ditambahkan 24 November 2026 untuk pembanding mutual information dan pembahasan wrapper: \[9\] untuk pembedaan filter vs wrapper, \[10\] dan \[11\] untuk estimator MI yang dipakai `mutual_info_classif` di scikit-learn, \[12\] untuk *wrapper overfitting* (CV naik, validation turun). Rujukan ini dipakai di bagian Method/Results milik Anggota B dan C; sampaikan ke penyusun paper. Entri \[3\], \[5\], dan \[9\] belum memuat DOI; tambahkan dari halaman penerbit jika C meminta format yang seragam.

## Bahan slide Anggota A (±1 menit)

Sesuai Rencana Kerja, bagian A di presentasi adalah masalah, dataset, dan fitur. Usulan: 2 slide.

**Slide 1 — Masalah dan dataset**

- Judul: "Fitur akustik mana yang benar-benar dibutuhkan untuk mengenali emosi dari suara?"
- Isi: RAVDESS speech: 24 aktor (12 L, 12 P), 1.440 rekaman, 8 emosi. Split per aktor: 16 latih, 4 validasi, 4 uji.
- Visual: `eda_04_contoh_aktor01.png` (mel-spectrogram per emosi dari satu aktor).

**Slide 2 — Dari audio ke 90 fitur**

- Alur (bisa digambar sebagai panah): audio 48 kHz → mono 16 kHz → potong hening (51% isi rekaman adalah hening) → 90 fitur (MFCC, delta, F0, energi, ZCR, spektral, contrast, chroma).
- Satu temuan untuk dibawa ke slide B: "Banyak fitur saling tumpang tindih: 17 kelompok fitur saling menggantikan. Jadi pertanyaannya bukan fitur mana yang berguna, tetapi kombinasi mana yang cukup."
- Visual: Tabel II versi ringkas, atau `corr_01_matriks_korelasi.png`.

**Naskah bicara (±60 detik):**

> "Kami memakai RAVDESS: 1.440 rekaman dari 24 aktor yang memerankan delapan emosi. Kami membagi data per aktor, supaya model selalu diuji dengan suara orang yang belum pernah didengar. Setiap rekaman kami ubah ke mono 16 kHz dan kami potong heningnya, karena separuh isi rekaman ternyata hening. Lalu setiap rekaman kami ringkas menjadi 90 fitur akustik standar: warna suara (MFCC), nada, keras-pelan, dan bentuk spektrum. Ternyata banyak fitur yang saling tumpang tindih; misalnya tiga fitur energi hampir identik. Jadi kita tidak butuh 90 fitur. Pertanyaannya: kombinasi kecil mana yang cukup? Itulah yang dicari simulated annealing, yang akan dijelaskan selanjutnya."

## Jejak angka

Setiap angka di bagian A dan file sumbernya (semua di `project07/`). Jika data atau parameter berubah, perbarui angka dari file ini.

| Angka | Nilai | Sumber |
| --- | --- | --- |
| Jumlah rekaman, aktor, emosi, distribusi kelas | 1.440; 24 (12 L/12 P); 8; neutral 96, lainnya 192 | `data/metadata.csv`, `data/audit_report.txt` |
| Durasi rekaman | 2,94–5,27 s; rata-rata 3,70 s; total 88,8 menit | `data/metadata.csv` |
| File stereo identik | 5 file, korelasi kiri-kanan 1,0000 | Log Kerja, langkah 1 (cek stereo) |
| Median proporsi hening | 51% | `figures/eda/eda_summary.txt` |
| Selisih energi angry vs calm | 19,1 dB | `figures/eda/eda_summary.txt` |
| Rentang energi antaremosi/antaraktor | none 19,3/18,5 dB; peak 1,2/4,5 dB | `figures/preprocess/preprocess_summary.txt` |
| Durasi setelah trim | 45,5 menit | `figures/preprocess/preprocess_summary.txt` |
| File clipping | 18 (17 angry strong) | Log Kerja, hasil tindak lanjut EDA |
| Jumlah fitur per kelompok | 90 (26/26/6/3/2/8/7/12) | `data/features/feature_names.csv` |
| Rekaman tanpa frame bersuara | 5 | `figures/features/features_summary.txt`, `data/unvoiced_check.csv` |
| Median F0 train | L 149 Hz, P 259 Hz | `figures/features/features_summary.txt` |
| Reproduksi identik | 259.200 sel sama persis | `data/features/repro_check.json` |
| Split | 16/4/4 aktor; 960/240/240 rekaman | `data/splits.json`, `figures/splits/splits_summary.txt` |
| Jaccard aktor latih antarpartisi | rata-rata 0,46 (0,33–0,60) | `figures/splits/splits_summary.txt` |
| Pergeseran fitur akibat noise | white 1,20/1,90/2,67; ESC-50 0,56/0,95/1,48 (20/10/0 dB) | `figures/noisy/noisy_summary.txt` |
| Rekaman uji tanpa nada pada white 0 dB | 134 dari 240 | `figures/noisy/noisy_summary.txt` |
| Pasangan \|r\| > 0,8 dan kelompok redundan | 39 pasangan; 17 kelompok (41 fitur) | `figures/correlation/correlation_summary.txt` |
| Versi normalisasi terpilih dan macro F1 | peak; SVM 0,5165 vs 0,4846 (none); RF 0,4644 vs 0,4316 | Hasil baseline Anggota B (`results/`) |
| Top-52 MI vs SA-52 (val macro F1, SVM tetap) | 0,4658 vs 0,4995 | data/analysis/mi\_eval\_preview\_peak.csv (langkah 10) |
| MLP val macro F1 (90 / SA / MI) | 0,4769 / 0,4771 / 0,4523 | results/mlp\_summary.csv (langkah 12) |
| Akurasi fuzzy arousal tinggi vs rendah | 77% | results/mini\_eksperimen\_A.json |

Dokumen terkait: [Log Kerja Anggota A](https://claude.ai/artifact/NFiHzzfxsnaqjWYY26BJ9R) · [Dataset Card](https://claude.ai/artifact/7oWwaTQeho4ggEUvuDRkcf) · [Referensi dan Dataset](https://claude.ai/artifact/NsPec7Ydjq43i1r8sbB89Q)
