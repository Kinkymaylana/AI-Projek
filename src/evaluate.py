"""
evaluate.py - Modul evaluasi bersama untuk Proyek 07
(Finding the Best Acoustic Features).

Penanggung jawab : Anggota C
Versi            : 1.1 (24 November 2026)

Aturan proyek: SEMUA angka di paper dihitung lewat modul ini, supaya
perbandingan antar-metode memakai metrik, urutan label, dan format yang sama.

Isi modul:
  LABELS              urutan 8 emosi untuk semua confusion matrix dan tabel
  set_seed()          seed untuk random dan numpy
  evaluate()          accuracy, macro F1, UAR, F1 per kelas, confusion matrix
  bootstrap_ci()      interval kepercayaan bootstrap untuk satu metrik
  paired_bootstrap()  interval kepercayaan selisih dua model pada sampel yang sama
  mcnemar_test()      uji McNemar eksak untuk dua model pada sampel yang sama
  plot_confusion()    gambar confusion matrix
  result_row()        menyusun satu baris main_table dari hasil evaluate()
  append_result()     menambah satu baris ke results/main_table.csv

Contoh:
    import evaluate as ev
    ev.set_seed(42)
    m = ev.evaluate(y_va, model.predict(X_va))
    lo, hi = ev.bootstrap_ci(y_va, model.predict(X_va))[1:]
    row = ev.result_row('SVM_full', 'validation', m, n_features=90,
                        ci=(lo, hi), runtime_s=1.5, run_id='baseline_peak', seed=42)
    ev.append_result(row)

Pembagian data per aktor TIDAK diatur di sini; pakai src/data_splits.py (Anggota A).
"""
import csv
import os
import random
import warnings
from datetime import datetime

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)

EVALUATE_VERSION = '1.1'
DEFAULT_SEED = 42

# Urutan kode emosi RAVDESS (01..08). Sama dengan EMOTION_ORDER di baseline.py (Anggota B).
# Jangan pakai pipeline.classes_ untuk urutan tampilan: urutan itu alfabetis.
LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Delapan kolom pertama = kontrak di Rencana Kerja. Empat kolom terakhir = tambahan
# untuk ketertelusuran. Ini perubahan kontrak: umumkan ke grup sebelum dipakai.
CONTRACT_COLUMNS = ['method', 'split', 'n_features', 'accuracy', 'macro_f1',
                    'ci_low', 'ci_high', 'runtime_s']
TRACE_COLUMNS = ['run_id', 'seed', 'evaluated_at', 'note']
MAIN_TABLE_COLUMNS = CONTRACT_COLUMNS + TRACE_COLUMNS
# Path absolut dari root repo, supaya tabel yang sama dipakai dari folder mana pun skrip dijalankan.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TABLE_PATH = os.path.join(REPO_ROOT, 'results', 'main_table.csv')


# ----------------------------------------------------------------------------
# Seed
# ----------------------------------------------------------------------------
def set_seed(seed=DEFAULT_SEED):
    """Set seed untuk modul `random` dan `numpy` (generator global).

    Catatan: scikit-learn TIDAK punya seed global. Setiap estimator dan splitter
    yang acak harus tetap diberi `random_state=seed` secara eksplisit.
    PYTHONHASHSEED juga diisi agar proses turunan (joblib) mewarisinya; proses
    Python yang sedang berjalan tidak terpengaruh.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    return seed


# ----------------------------------------------------------------------------
# Metrik
# ----------------------------------------------------------------------------
def _as_label_arrays(y_true, y_pred, labels):
    """Ubah ke array string dan pastikan semua label dikenal (menangkap typo/kode angka)."""
    yt = np.asarray(y_true).astype(str)
    yp = np.asarray(y_pred).astype(str)
    if yt.shape != yp.shape or yt.ndim != 1:
        raise ValueError(f'y_true dan y_pred harus 1 dimensi dan sama panjang: {yt.shape} vs {yp.shape}')
    if yt.size == 0:
        raise ValueError('y_true kosong')
    known = set(labels)
    unknown = (set(yt) | set(yp)) - known
    if unknown:
        raise ValueError(f'Label tidak dikenal {sorted(unknown)}; label yang sah: {list(labels)}')
    return yt, yp


def _macro_f1(yt, yp, labels):
    return float(f1_score(yt, yp, labels=labels, average='macro', zero_division=0))


METRICS = {
    'macro_f1': _macro_f1,
    'accuracy': lambda yt, yp, labels: float(accuracy_score(yt, yp)),
    'uar': lambda yt, yp, labels: float(balanced_accuracy_score(yt, yp)),
}


def evaluate(y_true, y_pred, labels=None):
    """Hitung metrik standar proyek.

    Returns
    -------
    dict berisi:
      n                 jumlah sampel
      accuracy          proporsi prediksi benar
      macro_f1          rata-rata F1 delapan kelas (bobot sama; kelas neutral yang lebih kecil
                        tetap dihitung setara)
      uar               unweighted average recall (= balanced accuracy)
      per_class_f1      {label: F1}
      confusion_matrix  np.ndarray int (baris = label sebenarnya, kolom = prediksi),
                        urutan baris/kolom = labels
      labels            urutan label yang dipakai
    """
    labels = list(LABELS if labels is None else labels)
    yt, yp = _as_label_arrays(y_true, y_pred, labels)
    absent = [lab for lab in labels if lab not in set(yt)]
    if absent:
        warnings.warn(f'Label tidak muncul di y_true: {absent}. F1-nya dihitung 0 dan '
                      f'menurunkan macro F1.', stacklevel=2)
    per_class = f1_score(yt, yp, labels=labels, average=None, zero_division=0)
    return {
        'n': int(yt.size),
        'accuracy': METRICS['accuracy'](yt, yp, labels),
        'macro_f1': METRICS['macro_f1'](yt, yp, labels),
        'uar': METRICS['uar'](yt, yp, labels),
        'per_class_f1': {lab: float(v) for lab, v in zip(labels, per_class)},
        'confusion_matrix': confusion_matrix(yt, yp, labels=labels),
        'labels': labels,
    }


# ----------------------------------------------------------------------------
# Ketidakpastian dan uji statistik
# ----------------------------------------------------------------------------
def bootstrap_ci(y_true, y_pred, metric='macro_f1', n_boot=1000, seed=DEFAULT_SEED,
                 ci=0.95, labels=None):
    """Interval kepercayaan bootstrap persentil untuk satu metrik.

    Resampling dilakukan per FILE (dengan pengembalian). Keterbatasan: file dari
    aktor yang sama tidak independen, sehingga interval ini cenderung terlalu sempit.
    Dengan hanya 4 aktor test, bootstrap per aktor tidak praktis; sebutkan
    keterbatasan ini di paper.

    Returns
    -------
    (nilai_titik, batas_bawah, batas_atas)
    """
    labels = list(LABELS if labels is None else labels)
    yt, yp = _as_label_arrays(y_true, y_pred, labels)
    fn = METRICS[metric]
    rng = np.random.default_rng(seed)
    n = yt.size
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = fn(yt[idx], yp[idx], labels)
    a = (1.0 - ci) / 2.0
    lo, hi = np.percentile(stats, [100 * a, 100 * (1 - a)])
    return fn(yt, yp, labels), float(lo), float(hi)


def paired_bootstrap(y_true, pred_a, pred_b, metric='macro_f1', n_boot=1000,
                     seed=DEFAULT_SEED, ci=0.95, labels=None):
    """Interval kepercayaan untuk selisih metrik (A - B) pada sampel yang SAMA.

    Indeks resample yang sama dipakai untuk kedua model, sehingga variasi antarsampel
    saling meniadakan. Pakai ini untuk klaim "setara": jika interval selisih mencakup 0
    dan sempit, kedua model tidak dapat dibedakan pada data ini.

    Returns
    -------
    dict: diff (titik), ci_low, ci_high, metric_a, metric_b
    """
    labels = list(LABELS if labels is None else labels)
    yt, ya = _as_label_arrays(y_true, pred_a, labels)
    _, yb = _as_label_arrays(y_true, pred_b, labels)
    fn = METRICS[metric]
    rng = np.random.default_rng(seed)
    n = yt.size
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = fn(yt[idx], ya[idx], labels) - fn(yt[idx], yb[idx], labels)
    a = (1.0 - ci) / 2.0
    lo, hi = np.percentile(diffs, [100 * a, 100 * (1 - a)])
    ma, mb = fn(yt, ya, labels), fn(yt, yb, labels)
    return {'metric': metric, 'metric_a': ma, 'metric_b': mb, 'diff': ma - mb,
            'ci_low': float(lo), 'ci_high': float(hi)}


def mcnemar_test(y_true, pred_a, pred_b, labels=None):
    """Uji McNemar eksak (binomial dua sisi) untuk dua model pada sampel yang sama.

    b = A benar & B salah, c = A salah & B benar. H0: b dan c sama mungkin.
    Uji ini membandingkan ACCURACY (benar/salah per file), bukan macro F1.
    p besar tidak membuktikan kedua model setara; itu hanya berarti tidak ada bukti beda.
    """
    labels = list(LABELS if labels is None else labels)
    yt, ya = _as_label_arrays(y_true, pred_a, labels)
    _, yb = _as_label_arrays(y_true, pred_b, labels)
    ca, cb = ya == yt, yb == yt
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5, alternative='two-sided').pvalue)
    return {'a_only_correct': b, 'b_only_correct': c, 'n_discordant': b + c,
            'p_value': p, 'method': 'exact binomial'}


# ----------------------------------------------------------------------------
# Gambar
# ----------------------------------------------------------------------------
def plot_confusion(cm, labels=None, title=None, normalize=True, ax=None, save_path=None, lang='id'):
    """Gambar confusion matrix. Warna = proporsi per baris (recall); angka = jumlah file.

    Parameters
    ----------
    cm : np.ndarray dari evaluate()['confusion_matrix'] (urutan sesuai labels)
    normalize : bool, warnai berdasarkan proporsi per baris (default True)
    save_path : str, opsional; jika diisi, gambar disimpan (dpi 200)
    lang : 'id' (demo, dokumen) atau 'en' (gambar paper)

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    labels = list(LABELS if labels is None else labels)
    cm = np.asarray(cm)
    if cm.shape != (len(labels), len(labels)):
        raise ValueError(f'Ukuran cm {cm.shape} tidak cocok dengan {len(labels)} label')
    row_sum = cm.sum(axis=1, keepdims=True)
    shown = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0) if normalize else cm

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
    else:
        fig = ax.figure
    im = ax.imshow(shown, cmap='Blues', vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    text = {'id': ('Prediksi', 'Label sebenarnya', 'Proporsi per baris', 'Jumlah'),
            'en': ('Predicted', 'True label', 'Row proportion (recall)', 'Count')}[lang]
    ax.set_xlabel(text[0])
    ax.set_ylabel(text[1])
    if title:
        ax.set_title(title)
    thresh = 0.5 if normalize else shown.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(int(cm[i, j])), ha='center', va='center', fontsize=8,
                    color='white' if shown[i, j] > thresh else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label=text[2] if normalize else text[3])
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


# ----------------------------------------------------------------------------
# Tabel hasil
# ----------------------------------------------------------------------------
def result_row(method, split, metrics, n_features, ci=None, runtime_s=None,
               run_id='', seed='', note=''):
    """Susun satu baris main_table dari hasil evaluate(), tanpa menyalin angka manual."""
    ci_low, ci_high = (ci if ci is not None else ('', ''))
    return {
        'method': method, 'split': split, 'n_features': n_features,
        'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1'],
        'ci_low': ci_low, 'ci_high': ci_high,
        'runtime_s': '' if runtime_s is None else runtime_s,
        'run_id': run_id, 'seed': seed, 'note': note,
    }


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return f'{float(v):.6f}'
    if isinstance(v, np.integer):
        return str(int(v))
    return '' if v is None else str(v)


def append_result(row, path=DEFAULT_TABLE_PATH, allow_test=False):
    """Tambah satu baris ke results/main_table.csv.

    Pengaman test: baris dengan split yang diawali 'test' (test, test_noisy_white_10db, ...)
    ditolak kecuali allow_test=True. Hanya Anggota C yang memanggilnya dengan
    allow_test=True, sekali, pada evaluasi akhir. Kombinasi method + split test yang
    sudah tercatat juga ditolak, supaya hasil test tidak bisa diulang lalu dipilih
    yang terbaik. Waktu evaluasi dicatat otomatis di kolom evaluated_at.
    """
    missing = [c for c in ['method', 'split', 'n_features', 'accuracy', 'macro_f1'] if c not in row]
    if missing:
        raise ValueError(f'Kolom wajib tidak ada: {missing}')
    extra = set(row) - set(MAIN_TABLE_COLUMNS)
    if extra:
        raise ValueError(f'Kolom di luar kontrak main_table: {sorted(extra)}')
    split = str(row['split'])
    if split.startswith('test') and not allow_test:
        raise PermissionError(
            f"Split '{split}' adalah data test. Test hanya dievaluasi sekali oleh Anggota C "
            f"(append_result(..., allow_test=True)).")

    full = {c: row.get(c, '') for c in MAIN_TABLE_COLUMNS}
    full['evaluated_at'] = datetime.now().isoformat(timespec='seconds')

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    if exists:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            if header != MAIN_TABLE_COLUMNS:
                raise ValueError(f'Header {path} tidak sesuai kontrak:\n  ada  : {header}\n  harus: {MAIN_TABLE_COLUMNS}')
            if split.startswith('test'):
                i_method, i_split = header.index('method'), header.index('split')
                for old in reader:
                    if old and old[i_method] == str(row['method']) and old[i_split] == split:
                        raise PermissionError(
                            f"Hasil test '{row['method']}' pada split '{split}' sudah tercatat di {path}. "
                            f"Test hanya dievaluasi sekali; hasil lama tidak boleh ditimpa atau diulang.")
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(MAIN_TABLE_COLUMNS)
        w.writerow([_fmt(full[c]) for c in MAIN_TABLE_COLUMNS])
    return full
