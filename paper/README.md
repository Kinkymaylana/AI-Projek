# Mini-paper IEEE — Proyek 07

Draf paper berbahasa Inggris dengan template konferensi IEEE (`IEEEtran`), target 6–8 halaman. Disatukan oleh Anggota C dari bahan A, bahan B, dan hasil C. Versi 0.1, 24 November 2026.

| File | Isi |
| --- | --- |
| `main.tex` | Naskah lengkap; sisa 8 `\TODO{...}` (merah): nama dan afiliasi, konfirmasi kontribusi, URL repo |
| `references.bib` | 17 referensi: 12 dari bahan A, 5 tambahan C (SA, stabilitas, scikit-learn, uji statistik, bootstrap) |

## Cara kompilasi

LaTeX belum terpasang di komputer C, jadi draf ini **belum pernah dikompilasi**. Cara termudah adalah Overleaf:

1. Buat proyek kosong di Overleaf, lalu unggah `main.tex` dan `references.bib`.
2. Buat folder `figures/` di Overleaf dengan subfolder berikut, lalu unggah gambar dari repo dengan nama yang sama:
   - `figures/lambda/lambda_tradeoff_en.png`
   - `figures/test_eval/cm_SVM_SA52_en.png`
   - `figures/test_eval/noise_macro_f1_svm_en.png`
   - `figures/stability/stab_03_transfer.png`
3. Di `main.tex`, ganti baris `\graphicspath{...}` menjadi `\graphicspath{{figures/}}`.
4. Compile dengan pdfLaTeX + BibTeX (bawaan Overleaf).

Jika dikompilasi langsung dari folder `paper/` di repo, path `../figures/` sudah benar dan langkah 2–3 tidak perlu.

## Sumber setiap bagian

| Bagian | Sumber |
| --- | --- |
| Abstract, Introduction, Conclusion | C, dari hasil di bawah |
| II. Related Work | Bahan A (`docs/Bahan Paper Anggota A …md`, bagian II), ditambah paragraf seleksi fitur dan stabilitas oleh C |
| III. Data and Features | Bahan A (III-A sampai III-C), diringkas dan diterjemahkan |
| IV. Method | Bahan B (`docs/bahan_paper_anggota_B.md`, III-E sampai III-G); Gambar λ dari `src/plot_lambda_tradeoff.py` |
| V. Experimental Protocol | Bahan A (III-D), `docs/audit_anti_leakage.md`, `src/run_test_evaluation.py` |
| VI. Results | Tabel validation: bahan B; tabel test dan uji berpasangan: `results/test_eval/test_summary.json`; confusion matrix: `figures/test_eval/` |
| VII. Failure Analysis | `results/stability_summary.json`, `results/stability_transfer.csv`, `results/noise_eval/noise_summary.json`, `results/error_analysis/`, catatan dengar di `docs/log_kerja_anggota_C.md` |

Semua angka diambil dari file hasil, bukan disalin dari ingatan. Jika ada file hasil yang berubah, angka di paper harus diperbarui dari file tersebut.

## Yang harus dilengkapi atau dikonfirmasi tim

- [ ] Nama dan afiliasi ketiga anggota (`\TODO` di bagian author).
- [x] Paragraf Discussion (bagian VIII) ditulis C: (1) fitur lebih sedikit, akurasi sama, dan tanda *wrapper overfitting*; (2) subset tidak stabil tetapi saling menggantikan, sehingga interpretasi sebaiknya per kelompok fitur, serta kaitannya dengan GeMAPS; (3) batas performa berasal dari speaker dan noise, bukan dari seleksi. Mohon A dan B membacanya.
- [ ] Konfirmasi Author Contributions.
- [ ] URL repo di Code and Data Availability.
- [ ] A: periksa terjemahan bagian III dan Related Work, dan pastikan klaim tiga referensi survei sesuai isi paper aslinya.
- [ ] B: periksa bagian IV dan Tabel II (validation). Ada dua catatan:
  - Di `models/final_models_manifest.json`, simpangan baku CV 5-fold `SVM_MI52` (0,0448 / 0,0501) **identik** dengan `SVM_all90`, padahal rata-ratanya berbeda. Kemungkinan salah salin di `train_final_models.py`. Angka ini tidak dipakai di paper.
  - Klaim "SA substantially outperformed MI" hanya berlaku di validation. Di test, selisih SA − MI +0,047 dengan CI mencakup 0, jadi paper menyebutnya "cenderung", bukan "lebih baik".
- [ ] Semua: periksa 5 referensi tambahan C di `references.bib` terhadap halaman penerbit.
- [ ] Kompilasi di Overleaf, pastikan tabel dan gambar muat di satu kolom dan total ≤ 8 halaman.

## Aturan penulisan

- Kesimpulan tidak boleh melebihi bukti: CI yang mencakup 0 ditulis "no detectable difference" atau "indistinguishable", bukan "equivalent" atau "better".
- Angka test hanya berasal dari evaluasi sekali (24 November 2026). Tidak ada model, subset, atau λ yang boleh diubah setelah itu.
- Hasil MLP SA-52 (0,554) tidak dijadikan klaim utama; alasannya tertulis di bagian VI.
