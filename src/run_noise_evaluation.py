"""
run_noise_evaluation.py - Uji noise SEKALI pada test bernoise (Langkah 10, Anggota C).

Penanggung jawab : Anggota C
Versi            : 1.0 (24 November 2026)

Protokol (Rencana Kerja dan pesan langkah 9 Anggota A): model yang SAMA dengan evaluasi
test bersih (dilatih di train + validation, dipilih di validation bersih) memprediksi
6 kondisi test bernoise: white noise dan ESC-50, SNR 20 / 10 / 0 dB, versi fitur peak.
Tidak ada model yang dilatih, di-tuning, atau dipilih ulang di sini.

Dua mode:

  check (bawaan, aman)
      Hanya membaca kolom file dan actor dari 6 tabel bernoise (bukan fitur, bukan label)
      untuk memastikan tabelnya lengkap, kolomnya sama dengan features_peak.csv, dan berisi
      240 file yang sama dengan test bersih. Juga memeriksa semua syarat mode noise.

  noise (SEKALI, hanya Anggota C, setelah evaluasi test bersih)
      Syarat: evaluasi test bersih sudah selesai (TEST_DONE.json), mode check bersih,
      belum ada baris test_noisy di main_table.csv, dan frasa konfirmasi diketik persis.
      Hasil:
        - 72 baris di results/main_table.csv (12 metode x 6 kondisi), split test_noisy_<jenis>_<snr>db
        - prediksi per file: results/noise_eval/noise_predictions.csv
        - metrik, penurunan dari test bersih, uji berpasangan SA vs 90 fitur: results/noise_eval/noise_summary.json
        - kunci results/noise_eval/NOISE_DONE.json
      Gambar dibuat sesudahnya dari hasil tersimpan: src/plot_test_figures.py

Jalankan dari root repo:
    .venv\\Scripts\\python.exe src\\run_noise_evaluation.py
    .venv\\Scripts\\python.exe src\\run_noise_evaluation.py --mode noise --confirm "EVALUASI NOISE SEKALI"
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)
import evaluate as ev  # noqa: E402
import run_test_evaluation as rt  # noqa: E402

RUNNER_VERSION = '1.0'
SEED = 42
CONFIRM_PHRASE = 'EVALUASI NOISE SEKALI'
NOISY_DIR = os.path.join(REPO_ROOT, 'data', 'features', 'noisy')
CLEAN_PREDICTIONS = os.path.join(REPO_ROOT, 'results', 'test_eval', 'test_predictions.csv')
CLEAN_SUMMARY = os.path.join(REPO_ROOT, 'results', 'test_eval', 'test_summary.json')
OUT_DIR = os.path.join(REPO_ROOT, 'results', 'noise_eval')
LOCK_PATH = os.path.join(OUT_DIR, 'NOISE_DONE.json')
TABLE_PATH = ev.DEFAULT_TABLE_PATH

# (jenis noise, SNR dB). Urutan dari ringan ke berat.
CONDITIONS = [('white', 20), ('white', 10), ('white', 0), ('esc50', 20), ('esc50', 10), ('esc50', 0)]
# Pertanyaan riset untuk noise: apakah subset SA lebih rapuh daripada semua fitur?
COMPARISONS = [('SVM_SA52', 'SVM_all90'), ('SVM_SA52', 'SVM_MI52'),
               ('RF_SA52', 'RF_all90'), ('MLP_SA52', 'MLP_full90')]


def split_name(kind, snr):
    return f'test_noisy_{kind}_{snr}db'


def noisy_path(kind, snr):
    return os.path.join(NOISY_DIR, f'test_noisy_{kind}_{snr}db_peak.csv')


# ----------------------------------------------------------------------------
# Mode check: tidak membaca fitur maupun label
# ----------------------------------------------------------------------------
def check_tables():
    """Daftar masalah pada 6 tabel bernoise (kosong = aman)."""
    problems = []
    expected_cols = list(pd.read_csv(rt.FEATURES_CSV, nrows=0).columns)
    splits = rt.ds.load_splits(rt.REPO_ROOT)
    clean_files = None
    if os.path.exists(CLEAN_PREDICTIONS):
        clean_files = set(pd.read_csv(CLEAN_PREDICTIONS, usecols=['file'])['file'])
    for kind, snr in CONDITIONS:
        path = noisy_path(kind, snr)
        name = os.path.basename(path)
        if not os.path.exists(path):
            problems.append(f'{name} tidak ada')
            continue
        cols = list(pd.read_csv(path, nrows=0).columns)
        if cols != expected_cols:
            problems.append(f'{name}: kolom tidak sama dengan features_peak.csv')
        ids = pd.read_csv(path, usecols=['file', 'actor'])
        if len(ids) != 240 or ids['file'].duplicated().any():
            problems.append(f'{name}: harus 240 file unik, ada {len(ids)}')
        if not set(ids['actor']) <= set(splits['test']):
            problems.append(f'{name}: berisi aktor di luar test {sorted(set(ids["actor"]) - set(splits["test"]))}')
        if clean_files is not None and set(ids['file']) != clean_files:
            problems.append(f'{name}: daftar file berbeda dari test bersih')
    return problems


def preflight(confirm):
    """Daftar alasan mode noise DITOLAK (kosong = boleh jalan)."""
    problems = []
    if confirm != CONFIRM_PHRASE:
        problems.append(f'frasa konfirmasi harus persis "{CONFIRM_PHRASE}"')
    if not os.path.exists(rt.LOCK_PATH):
        problems.append('evaluasi test bersih belum selesai (TEST_DONE.json tidak ada)')
    if os.path.exists(LOCK_PATH):
        problems.append(f'uji noise sudah pernah dijalankan ({os.path.basename(LOCK_PATH)} ada)')
    if os.path.exists(TABLE_PATH) and os.path.getsize(TABLE_PATH) > 0:
        t = pd.read_csv(TABLE_PATH, dtype=str)
        if t['split'].fillna('').str.startswith('test_noisy').any():
            problems.append('main_table.csv sudah berisi baris test_noisy')
    missing = [m['method'] for m in rt.METHODS if not os.path.exists(os.path.join(rt.REPO_ROOT, m['path']))]
    if missing:
        problems.append(f'model belum ada: {missing}')
    problems += check_tables()
    return problems


# ----------------------------------------------------------------------------
# Mode noise
# ----------------------------------------------------------------------------
def run_noise(confirm):
    problems = preflight(confirm)
    if problems:
        raise SystemExit('Mode noise DITOLAK:\n  - ' + '\n  - '.join(problems))

    ev.set_seed(SEED)
    cols = rt.feature_columns()
    with open(CLEAN_SUMMARY, encoding='utf-8') as f:
        clean = json.load(f)['results']
    bundles = {m['method']: rt.load_bundle(os.path.join(rt.REPO_ROOT, m['path'])) for m in rt.METHODS}

    results, pairs, pred_frames = {}, [], []
    for kind, snr in CONDITIONS:
        split = split_name(kind, snr)
        df = pd.read_csv(noisy_path(kind, snr))
        y = df['emotion'].to_numpy()
        out = df[['file', 'actor', 'emotion']].copy()
        out.insert(0, 'condition', split)
        preds, results[split] = {}, {}
        for m in rt.METHODS:
            b = bundles[m['method']]
            idx = list(b['selected_indices'])
            t0 = time.perf_counter()
            pred = rt.to_labels(b['pipeline'].predict(rt.matrix(df, cols, idx)), b)
            runtime = time.perf_counter() - t0
            met = ev.evaluate(y, pred)
            _, lo, hi = ev.bootstrap_ci(y, pred, seed=SEED)
            ev.append_result(ev.result_row(
                m['method'], split, met, n_features=len(idx), ci=(lo, hi), runtime_s=round(runtime, 4),
                run_id=os.path.basename(m['path']), seed=SEED,
                note=f'model final train+validation; {kind} noise SNR {snr} dB; uji noise sekali oleh C'),
                path=TABLE_PATH, allow_test=True)
            preds[m['method']] = pred
            out[m['method']] = pred
            clean_f1 = clean[m['method']]['macro_f1']
            results[split][m['method']] = {
                'k': len(idx), 'accuracy': met['accuracy'], 'macro_f1': met['macro_f1'], 'uar': met['uar'],
                'ci_low': lo, 'ci_high': hi, 'clean_macro_f1': clean_f1,
                'drop_from_clean': clean_f1 - met['macro_f1'],
                'relative_drop': (clean_f1 - met['macro_f1']) / clean_f1 if clean_f1 else None,
                'per_class_f1': met['per_class_f1'], 'confusion_matrix': met['confusion_matrix'].tolist(),
            }
        for a, b_ in COMPARISONS:
            pb = ev.paired_bootstrap(y, preds[a], preds[b_], seed=SEED)
            pairs.append({'condition': split, 'a': a, 'b': b_, **pb,
                          'mcnemar': ev.mcnemar_test(y, preds[a], preds[b_])})
        pred_frames.append(out)
        line = '  '.join(f"{k} {results[split][k]['macro_f1']:.3f}"
                         for k in ('SVM_all90', 'SVM_SA52', 'SVM_MI52') if k in results[split])
        print(f'  {split:24s} {line}')

    os.makedirs(OUT_DIR, exist_ok=True)
    pd.concat(pred_frames).to_csv(os.path.join(OUT_DIR, 'noise_predictions.csv'), index=False)
    summary = {
        'runner_version': RUNNER_VERSION, 'evaluate_version': ev.EVALUATE_VERSION,
        'evaluated_at': datetime.now().isoformat(timespec='seconds'), 'seed': SEED,
        'conditions': [split_name(k, s) for k, s in CONDITIONS], 'feature_version': 'peak',
        'results': results, 'paired_comparisons': pairs,
        'note': ('Model sama dengan evaluasi test bersih; tidak ada pelatihan atau pemilihan ulang. '
                 'drop_from_clean = macro F1 bersih - macro F1 bernoise.'),
    }
    with open(os.path.join(OUT_DIR, 'noise_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(LOCK_PATH, 'w', encoding='utf-8') as f:
        json.dump({'evaluated_at': summary['evaluated_at'], 'conditions': summary['conditions'],
                   'methods': [m['method'] for m in rt.METHODS]}, f, indent=2)
    print(f'Selesai. Hasil: {OUT_DIR}; kunci: {LOCK_PATH}')
    print('Gambar: .venv\\Scripts\\python.exe src\\plot_test_figures.py')
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['check', 'noise'], default='check')
    p.add_argument('--confirm', default='', help=f'wajib untuk mode noise: "{CONFIRM_PHRASE}"')
    args = p.parse_args(argv)
    if args.mode == 'check':
        print('Cek tabel bernoise (hanya kolom file dan actor; fitur dan label tidak dibaca).')
        problems = preflight(CONFIRM_PHRASE)
        print('Kesiapan mode noise: ' + ('SIAP' if not problems else 'BELUM SIAP'))
        for pr in problems:
            print(f'  - {pr}')
    else:
        run_noise(args.confirm)


if __name__ == '__main__':
    main()
