"""
app.py - Demo final Proyek 07 (Finding the Best Acoustic Features).

Penanggung jawab : Anggota C
Versi            : 0.2 (24 November 2026)

Empat tab:
  Optimasi        kurva SA per iterasi (dari history) dan konfigurasinya
  Fitur terpilih  52 fitur model dan frekuensinya di 15 run stabilitas
  Perbandingan    90 fitur vs subset SA pada validation, serta status hasil test
  Coba sendiri    probabilitas emosi dari model final 90 fitur dan subset SA, berdampingan

Semua angka dibaca dari file hasil, tidak diketik manual. Aplikasi ini TIDAK menghitung
apa pun pada data test. Hasil test baru tampil jika sudah tercatat di results/main_table.csv.

Jalankan dari root repo:
    .venv\\Scripts\\python.exe app.py
lalu buka http://127.0.0.1:7860
"""
import json
import os
import sys

import gradio as gr
import joblib
import pandas as pd
import soundfile as sf
from audioread.exceptions import DecodeError

ROOT = os.path.dirname(os.path.abspath(__file__))
B_DIR = os.path.join(ROOT, 'progres_anggota_b')
B_RESULTS = os.path.join(B_DIR, 'results')
B_FIGURES = os.path.join(B_DIR, 'figures', 'sa')
MAIN_TABLE = os.path.join(ROOT, 'results', 'main_table.csv')
STAB_SUMMARY = os.path.join(ROOT, 'results', 'stability_summary.json')
STAB_FIG_DIR = os.path.join(ROOT, 'figures', 'stability')

sys.path.insert(0, os.path.join(B_DIR, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from evaluate import LABELS  # noqa: E402
from inference import get_predictor  # noqa: E402

# Model final Anggota B (dilatih pada train + validation, aktor 1-20). Dimuat sekali saat mulai.
MODEL_FINAL = {'90 fitur': os.path.join(ROOT, 'models', 'svm_all90_final.joblib'),
               'SA 52 fitur': os.path.join(ROOT, 'models', 'svm_sa52_final.joblib')}
if all(os.path.exists(p) for p in MODEL_FINAL.values()):
    PREDIKTOR = {nama: get_predictor(path) for nama, path in MODEL_FINAL.items()}
    CV5 = {nama: joblib.load(path)['cv5_trainval_macro_f1_mean'] for nama, path in MODEL_FINAL.items()}
else:  # cadangan: model P0 dari handoff awal B
    PREDIKTOR, CV5 = {'SA 52 fitur (P0)': get_predictor()}, {}

# Puncak di bawah ambang ini dianggap hening. Puncak terkecil di 1.440 file RAVDESS = 0,00455
# (data/preprocess_log.csv), jadi ambang ini tidak menolak rekaman dataset.
AMBANG_HENING = 0.001


# ----------------------------------------------------------------------------
# Muat hasil
# ----------------------------------------------------------------------------
def muat_json(nama):
    with open(os.path.join(B_RESULTS, nama), encoding='utf-8') as f:
        return json.load(f)


SA = muat_json('sa_main_results.json')
BASELINE = muat_json('baseline_results.json')['results'][SA['normalization']]
SUBSETS = muat_json('subsets.json')
STABIL = SUBSETS['stability_preview']
VAL = SA['validation_comparison']
FREKUENSI = pd.read_csv(os.path.join(B_RESULTS, 'feature_selection_frequency.csv'))
RUNS = pd.read_csv(os.path.join(B_RESULTS, 'stability_runs_summary.csv'))
TERPILIH = set(SA['selected_indices'])
N_RUN = SUBSETS['n_runs']
SEMUA = 'Semua'


def koma(nilai, desimal=4):
    """Format angka dengan koma desimal."""
    return f'{nilai:.{desimal}f}'.replace('.', ',')


def selisih(nilai):
    return ('+' if nilai > 0 else '') + koma(nilai)


# ----------------------------------------------------------------------------
# Tabel
# ----------------------------------------------------------------------------
def tabel_optimasi():
    m, c = SA['sa_metrics'], SA['config']
    return pd.DataFrame([
        ['Partisi dan seed', f"{SA['partition']}, seed {SA['seed']}"],
        ['Normalisasi', SA['normalization']],
        ['Iterasi', c['n_iter']],
        ['Suhu awal dan akhir', f"{koma(c['t0'], 4)} ke {koma(c['t_min'], 4)}"],
        ['Penalti jumlah fitur (lambda)', koma(c['lambda_penalty'], 3)],
        ['Validasi silang', f"GroupKFold {c['n_splits']} fold per aktor (train)"],
        ['Solusi lebih buruk yang diterima', m['accepted_worse']],
        ['Waktu komputasi', f"{koma(m['elapsed_s'], 1)} detik"],
    ], columns=['Parameter', 'Nilai']).astype(str)


def tabel_kelompok():
    return pd.DataFrame([[g['group'], g['selected'], g['total'], f"{koma(g['pct'], 1)}%"]
                         for g in SA['group_breakdown']],
                        columns=['Kelompok', 'Terpilih', 'Total', 'Persentase']).astype(str)


def saring_fitur(kelompok=SEMUA, status=SEMUA, cari=''):
    d = FREKUENSI.copy()
    d['dipakai'] = d['index'].isin(TERPILIH)
    if kelompok != SEMUA:
        d = d[d['group'] == kelompok]
    if status == 'Dipakai model':
        d = d[d['dipakai']]
    elif status == 'Tidak dipakai':
        d = d[~d['dipakai']]
    cari = (cari or '').strip().lower()
    if cari:
        d = d[d['name'].str.lower().str.contains(cari, regex=False)]
    d = d.sort_values(['count', 'name'], ascending=[False, True])
    return pd.DataFrame({
        'Fitur': d['name'],
        'Kelompok': d['group'],
        'Dipakai model': d['dipakai'].map({True: 'Ya', False: 'Tidak'}),
        f'Terpilih dari {N_RUN} run': d['count'].astype(str),
        'Frekuensi': d['freq_pct'].map(lambda v: f'{koma(v, 1)}%'),
    })


def tabel_perbandingan():
    svm, rf = BASELINE['SVM'], BASELINE['RF']
    k = SA['k_selected']
    n = SA['n_features_total']
    return pd.DataFrame([
        ['SVM', 'Semua fitur', n, koma(svm['val_macro_f1']), koma(svm['val_uar']), koma(svm['val_accuracy'])],
        ['SVM', 'Subset SA', k, koma(VAL['svm_sa_subset']), koma(VAL['svm_sa_uar']), koma(VAL['svm_sa_accuracy'])],
        ['RF', 'Semua fitur', n, koma(rf['val_macro_f1']), koma(rf['val_uar']), koma(rf['val_accuracy'])],
        ['RF', 'Subset SA', k, koma(VAL['rf_sa_subset']), koma(VAL['rf_sa_uar']), '-'],
        ['SVM', 'Top-k Mutual Information', k, 'Menunggu Anggota A', '-', '-'],
    ], columns=['Model', 'Fitur', 'Jumlah', 'Macro F1', 'UAR', 'Accuracy']).astype(str)


def tabel_per_kelas():
    penuh, sa = BASELINE['SVM']['per_class_f1_val'], VAL['svm_sa_per_class_f1']
    return pd.DataFrame([[e, koma(penuh[e]), koma(sa[e]), selisih(sa[e] - penuh[e])] for e in LABELS],
                        columns=['Emosi', f"{SA['n_features_total']} fitur", f"SA {SA['k_selected']} fitur",
                                 'Selisih'])


def tabel_stabilitas():
    """Ringkasan stabilitas. Jika analisis C (src/run_stability_analysis.py) sudah dijalankan,
    Jaccard dibandingkan dengan pemilihan acak dan indeks Nogueira ikut ditampilkan."""
    rows = [
        ['Jumlah fitur (k)', f"{koma(STABIL['k_mean'], 2)} ± {koma(STABIL['k_std'], 2)} "
                             f"(rentang {STABIL['k_min']}-{STABIL['k_max']})", '-'],
        ['CV Macro F1', f"{koma(STABIL['cv_f1_mean'])} ± {koma(STABIL['cv_f1_std'])}", '-'],
        ['Validation Macro F1', f"{koma(STABIL['val_f1_mean'])} ± {koma(STABIL['val_f1_std'])}", '-'],
    ]
    if os.path.exists(STAB_SUMMARY):
        with open(STAB_SUMMARY, encoding='utf-8') as f:
            s = json.load(f)
        rows += [
            ['Jaccard per fitur', koma(s['jaccard_feature_all']), koma(s['jaccard_feature_random'])],
            ['Jaccard per kelompok redundan', koma(s['jaccard_cluster_all']), koma(s['jaccard_cluster_random'])],
            ['Indeks Nogueira (1 = identik, 0 = acak)', koma(s['nogueira_all']), koma(0)],
        ]
    else:
        rows += [['Jaccard per fitur', koma(STABIL['jaccard_feature_all15']), '-'],
                 ['Jaccard per kelompok redundan', koma(STABIL['jaccard_cluster_all15']), '-']]
    return pd.DataFrame(rows, columns=[f'Ukuran ({N_RUN} run)', 'SA', 'Subset acak, ukuran sama'])


def tabel_run():
    d = RUNS[['run_id', 'n_features', 'best_cv_f1', 'val_macro_f1']].copy()
    d['n_features'] = d['n_features'].astype(str)
    d['best_cv_f1'] = d['best_cv_f1'].map(koma)
    d['val_macro_f1'] = d['val_macro_f1'].map(koma)
    d.columns = ['Run', 'Jumlah fitur', 'CV Macro F1', 'Val Macro F1']
    return d


def status_test():
    """Baca hasil test yang sudah tercatat. Fungsi ini tidak mengevaluasi data test."""
    kolom = ['Metode', 'Split', 'Jumlah fitur', 'Accuracy', 'Macro F1', 'CI bawah', 'CI atas']
    kosong = pd.DataFrame(columns=kolom)
    pesan = ('**Hasil test belum tersedia.** Evaluasi test (aktor 21-24) dijalankan sekali setelah '
             'pembanding Mutual Information selesai.')
    if not os.path.exists(MAIN_TABLE):
        return kosong, pesan
    d = pd.read_csv(MAIN_TABLE, dtype=str, keep_default_na=False)
    d = d[d['split'].str.startswith('test')]
    if d.empty:
        return kosong, pesan
    d = d[['method', 'split', 'n_features', 'accuracy', 'macro_f1', 'ci_low', 'ci_high']].copy()
    for c in ['accuracy', 'macro_f1', 'ci_low', 'ci_high']:
        d[c] = d[c].map(lambda v: koma(float(v)) if v else '-')
    d.columns = kolom
    return d, f'**Hasil test dari `results/main_table.csv`** ({len(d)} baris).'


# ----------------------------------------------------------------------------
# Coba sendiri
# ----------------------------------------------------------------------------
KOLOM_SKOR = ['Emosi'] + list(PREDIKTOR)


def nilai_skor(res, emosi):
    """Probabilitas dalam persen; skor relatif (model lama) dalam desimal."""
    v = res['scores'][emosi]
    return f'{koma(v * 100, 1)}%' if res['score_type'] == 'probability' else koma(v, 2)


def format_skor(hasil):
    """Satu baris per emosi, diurutkan menurut model terakhir (subset SA)."""
    acuan = hasil[list(hasil)[-1]]['scores']
    return [[e] + [nilai_skor(r, e) for r in hasil.values()]
            for e in sorted(acuan, key=acuan.get, reverse=True)]


def prediksi(audio):
    kosong = pd.DataFrame(columns=KOLOM_SKOR)
    if audio is None:
        return kosong, 'Belum ada audio. Rekam atau unggah file dulu.'
    try:
        hasil = {nama: p.predict_gradio(audio) for nama, p in PREDIKTOR.items()}
    except ValueError as e:
        return kosong, f'Input tidak valid: {e}'
    except FileNotFoundError:
        return kosong, 'File audio tidak ditemukan. Unggah ulang filenya.'
    except (DecodeError, sf.LibsndfileError, EOFError):
        return kosong, 'File tidak dapat dibaca sebagai audio. Pastikan file tidak rusak, lalu unggah WAV.'
    res = next(iter(hasil.values()))
    if res['preprocess_info']['peak_trim'] < AMBANG_HENING:
        return kosong, 'Audio hening: tidak ada suara yang dapat dinilai. Rekam ulang dengan suara lebih jelas.'

    label = {nama: r['emotion'] for nama, r in hasil.items()}
    baris = [f"**{nama}:** {label[nama]}" for nama in hasil]
    info = '### Prediksi\n' + ' · '.join(baris)
    if len(set(label.values())) > 1:
        info += '\n\nKedua model memberi label berbeda untuk rekaman ini.'
    for nama, r in hasil.items():
        teratas = max(r['scores'], key=r['scores'].get)
        if r['score_type'] == 'probability' and teratas != r['emotion']:
            # SVC: label dari decision function, probabilitas dari Platt scaling; bisa tidak searah.
            info += (f"\n\n*{nama}: label SVM ({r['emotion']}) berbeda dari probabilitas tertinggi "
                     f"({teratas}). Ini wajar pada SVM; label yang dipakai untuk evaluasi adalah {r['emotion']}.*")
    info += (f"\n\nDurasi setelah trim {koma(res['duration_after_trim_s'], 2)} detik, "
             f"frame bersuara {res['voiced_ratio']:.0%}.")
    if res['voiced_ratio'] < 0.2:
        info += ('\n\n**Peringatan:** hampir tidak ada frame bersuara yang terdeteksi, '
                 'sehingga prediksi ini kurang dapat diandalkan.')
    return pd.DataFrame(format_skor(hasil), columns=KOLOM_SKOR), info


# ----------------------------------------------------------------------------
# Teks
# ----------------------------------------------------------------------------
_d_svm = VAL['svm_sa_subset'] - BASELINE['SVM']['val_macro_f1']
RINGKASAN = (
    f"SA memilih **{SA['k_selected']} dari {SA['n_features_total']} fitur** "
    f"(berkurang {koma(SA['feature_reduction_pct'], 1)}%). CV Macro F1 pada train naik dari "
    f"{koma(BASELINE['SVM']['cv_macro_f1_mean'])} ke {koma(SA['sa_metrics']['best_cv_f1'])}, "
    f"tetapi Macro F1 pada aktor validation berubah dari {koma(BASELINE['SVM']['val_macro_f1'])} "
    f"ke {koma(VAL['svm_sa_subset'])} ({selisih(_d_svm)})."
)
CATATAN_PERBANDINGAN = (
    f"Pada validation, subset SA memakai fitur lebih sedikit, dengan Macro F1 SVM "
    f"{selisih(_d_svm)} dibanding semua fitur. Selisih ini belum diuji secara statistik, "
    f"jadi belum bisa disebut lebih baik, lebih buruk, atau setara."
)
if CV5:
    _akurasi = ('Macro F1 validasi silang per aktor (5 fold, aktor 1-20): '
                + ', '.join(f'{nama} {koma(v, 2)}' for nama, v in CV5.items())
                + ' (8 kelas, tebakan acak sekitar 0,125)')
    CATATAN_COBA = f"""
**Cara membaca hasil**
- Kedua kolom adalah model SVM final: satu memakai 90 fitur, satu memakai {SA['k_selected']} fitur hasil SA.
- Angka adalah probabilitas per emosi (Platt scaling); tiap kolom berjumlah 100%.
- {_akurasi}, jadi prediksi yang salah adalah hal biasa.
- Model dilatih pada 20 aktor RAVDESS: bahasa Inggris, rekaman studio, emosi diperankan.
  Rekaman mikrofon biasa, bahasa lain, atau suasana bising berada di luar kondisi tersebut.
"""
else:
    CATATAN_COBA = f"""
**Cara membaca hasil**
- Model final belum ada di `models/`; yang dipakai model P0 dari handoff awal Anggota B.
- Skor adalah **skor relatif** dari SVM, **bukan probabilitas terkalibrasi**.
- Macro F1 model pada aktor validation {koma(VAL['svm_sa_subset'], 2)} (8 kelas, tebakan acak sekitar 0,125).
"""

# ----------------------------------------------------------------------------
# Kurva optimasi (dari history SA, satu baris per iterasi)
# ----------------------------------------------------------------------------
HISTORY = pd.DataFrame(SA['history'])
METRIK = {
    'Objective (CV Macro F1 - penalti)': ('cur_obj', 'best_obj'),
    'CV Macro F1': ('cur_cv_f1', 'best_cv_f1'),
    'Jumlah fitur': ('cur_k', 'best_k'),
    'Suhu': ('temp', None),
}


def kurva(metrik):
    cur, best = METRIK[metrik]
    seri = [('Solusi saat ini', cur), ('Terbaik sejauh ini', best)] if best else [('Suhu', cur)]
    return pd.concat([pd.DataFrame({'Iterasi': HISTORY['iter'], 'Nilai': HISTORY[kol], 'Seri': nama})
                      for nama, kol in seri], ignore_index=True)


# ----------------------------------------------------------------------------
# Antarmuka
# ----------------------------------------------------------------------------
def tabel(value, label=None, **kwargs):
    return gr.Dataframe(value=value, label=label, interactive=False, wrap=True, **kwargs)


def gambar(nama, label, folder=B_FIGURES, height=460):
    # Tinggi eksplisit: tanpa ini gambar di dalam tab bisa tampil kosong.
    path = os.path.join(folder, nama)
    if not os.path.exists(path):
        return gr.Markdown(f'*{label}: gambar belum dibuat (`{os.path.relpath(path, ROOT)}`).*')
    return gr.Image(value=path, label=label, interactive=False, height=height)


with gr.Blocks(title='Proyek 07 - Finding the Best Acoustic Features') as demo:
    gr.Markdown('# Finding the Best Acoustic Features\n'
                'Seleksi fitur akustik dengan Simulated Annealing untuk pengenalan emosi '
                'dari ucapan (RAVDESS, 8 emosi).')

    with gr.Tab('Optimasi'):
        gr.Markdown(RINGKASAN)
        p_metrik = gr.Radio(list(METRIK), value=list(METRIK)[0], label='Tampilkan')
        p_kurva = gr.LinePlot(kurva(list(METRIK)[0]), x='Iterasi', y='Nilai', color='Seri',
                              color_map={'Solusi saat ini': '#b5b3ad', 'Terbaik sejauh ini': '#2a78d6',
                                         'Suhu': '#2a78d6'},
                              title=f"Simulated Annealing, {len(HISTORY) - 1} iterasi", height=380,
                              y_title='', tooltip=['Seri', 'Iterasi', 'Nilai'])
        p_metrik.change(kurva, p_metrik, p_kurva)
        gr.Markdown('Garis abu-abu adalah solusi yang sedang dikunjungi SA; ia boleh turun karena SA '
                    'kadang menerima solusi lebih buruk. Garis biru adalah solusi terbaik yang disimpan.')
        tabel(tabel_optimasi(), 'Konfigurasi SA utama')

    with gr.Tab('Fitur terpilih'):
        gr.Markdown(f"Model memakai **{SA['k_selected']} fitur**. Frekuensi menunjukkan seberapa sering "
                    f"sebuah fitur terpilih di {N_RUN} run uji stabilitas.")
        with gr.Row():
            f_kelompok = gr.Dropdown([SEMUA] + sorted(FREKUENSI['group'].unique()), value=SEMUA, label='Kelompok')
            f_status = gr.Radio([SEMUA, 'Dipakai model', 'Tidak dipakai'], value=SEMUA, label='Status')
            f_cari = gr.Textbox(label='Cari nama fitur', placeholder='mis. mfcc03')
        t_fitur = tabel(saring_fitur(), 'Daftar fitur', max_height=420)
        for komponen in (f_kelompok, f_status, f_cari):
            komponen.change(saring_fitur, [f_kelompok, f_status, f_cari], t_fitur)
        gambar('stab_01_heatmap_seleksi.png', f'Heatmap seleksi fitur di {N_RUN} run',
               folder=STAB_FIG_DIR, height=420)
        tabel(tabel_kelompok(), 'Fitur terpilih per kelompok')
        gambar('sa_main_feature_groups.png', 'Sebaran kelompok fitur')

    with gr.Tab('Perbandingan'):
        gr.Markdown('## Validation (aktor 17-20)\n' + CATATAN_PERBANDINGAN)
        tabel(tabel_perbandingan(), 'Semua fitur vs subset SA')
        tabel(tabel_per_kelas(), 'F1 per emosi (SVM)')
        gr.Markdown('## Stabilitas seleksi fitur')
        tabel(tabel_stabilitas(), 'Ringkasan')
        gambar('stab_02_ringkasan_stabilitas.png', 'SA dibanding pemilihan acak', folder=STAB_FIG_DIR,
               height=320)
        gambar('sa_stability_frequency.png', 'Fitur paling konsisten')
        tabel(tabel_run(), 'Hasil per run', max_height=320)
        gr.Markdown('## Test (aktor 21-24)')
        m_test = gr.Markdown()
        t_test = tabel(None, 'Hasil test')
        demo.load(status_test, outputs=[t_test, m_test])

    with gr.Tab('Coba sendiri'):
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(sources=['upload', 'microphone'], type='filepath',
                                 label='Rekam atau unggah audio')
                tombol = gr.Button('Prediksi', variant='primary')
            with gr.Column():
                ringkasan = gr.Markdown()
                skor = gr.Dataframe(headers=KOLOM_SKOR, datatype=['str'] * len(KOLOM_SKOR), interactive=False,
                                    label='Probabilitas per emosi' if CV5 else 'Skor relatif per emosi')
        gr.Markdown(CATATAN_COBA)
        tombol.click(prediksi, inputs=audio, outputs=[skor, ringkasan])

if __name__ == '__main__':
    demo.launch()
