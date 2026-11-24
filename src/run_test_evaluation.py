"""
run_test_evaluation.py - Evaluasi test SEKALI untuk semua model final (Langkah 7, Anggota C).

Penanggung jawab : Anggota C
Versi            : 1.0 (24 November 2026)

Dua mode:

  dryrun (bawaan, aman dijalankan kapan saja)
      Data test (aktor 21-24) TIDAK dibaca: baris test dibuang tepat setelah CSV dimuat.
      Untuk setiap model final, pipeline-nya di-clone (hyperparameter dan subset fitur sama),
      di-fit ulang di train (aktor 1-16), lalu dinilai di validation (aktor 17-20).
      Angkanya dicocokkan dengan angka validation yang dilaporkan Anggota A/B.
      Jika cocok, berarti subset fitur, urutan kolom, dan pemetaan label di skrip ini benar,
      sehingga evaluasi test nanti memakai jalur yang sama persis.
      Hasil: results/test_eval/dryrun_validation.{csv,json}. main_table.csv tidak disentuh.

  test (SEKALI, hanya Anggota C, setelah daftar metode final disepakati)
      Syarat: semua model wajib ada, dryrun lolos, audit anti-leakage 0 FAIL,
      belum ada baris test di main_table.csv, dan frasa konfirmasi diketik persis.
      Model final (train + validation) memprediksi test dengan predict(), lalu:
        - satu baris per metode di results/main_table.csv (macro F1 + CI bootstrap 95%)
        - prediksi per file: results/test_eval/test_predictions.csv
        - metrik lengkap, F1 per emosi, uji berpasangan: results/test_eval/test_summary.json
        - confusion matrix: figures/test_eval/
        - kunci results/test_eval/TEST_DONE.json, supaya mode test tidak bisa diulang

Jalankan dari root repo:
    .venv\\Scripts\\python.exe src\\run_test_evaluation.py
    .venv\\Scripts\\python.exe src\\run_test_evaluation.py --mode test --confirm "EVALUASI TEST SEKALI"
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)
import data_splits as ds  # noqa: E402
import evaluate as ev  # noqa: E402

RUNNER_VERSION = '1.0'
SEED = 42
CONFIRM_PHRASE = 'EVALUASI TEST SEKALI'
FEATURES_CSV = os.path.join(REPO_ROOT, 'data', 'features', 'features_peak.csv')
FEATURE_NAMES_CSV = os.path.join(REPO_ROOT, 'data', 'features', 'feature_names.csv')
AUDIT_JSON = os.path.join(REPO_ROOT, 'results', 'audit_leakage.json')
OUT_DIR = os.path.join(REPO_ROOT, 'results', 'test_eval')
FIG_DIR = os.path.join(REPO_ROOT, 'figures', 'test_eval')
LOCK_PATH = os.path.join(OUT_DIR, 'TEST_DONE.json')
TABLE_PATH = ev.DEFAULT_TABLE_PATH

# ----------------------------------------------------------------------------
# Daftar metode final. Ubah HANYA sebelum mode test dijalankan, dan catat di log C.
#   val_ref  : macro F1 validation yang dilaporkan pemilik model (fit di train saja)
#   val_tol  : toleransi pencocokan dryrun (SVM/RF/ensemble deterministik; MLP 5 seed)
#   required : model wajib ada sebelum mode test boleh jalan
# ----------------------------------------------------------------------------
METHODS = [
    dict(method='SVM_all90', path='models/svm_all90_final.joblib', owner='B',
         val_ref=0.5165, val_tol=0.001, source='baseline_results.json', required=True),
    dict(method='SVM_SA52', path='models/svm_sa52_final.joblib', owner='B',
         val_ref=0.4995, val_tol=0.001, source='sa_main_results.json', required=True),
    # B men-tuning SVM MI-52 di train (protokol sama dengan SA-52); hasilnya kernel linear.
    dict(method='SVM_MI52', path='models/svm_mi52_final.joblib', owner='B',
         val_ref=0.4257, val_tol=0.001, source='final_models_manifest.json mi52_validation_summary',
         required=True),
    dict(method='SVM_HC48', path='models/hill_climbing/svm_hc_final.joblib', owner='B',
         val_ref=0.5025, val_tol=0.001, source='comparators_results.json', required=True),
    dict(method='SVM_Random52_median', path='models/random_subset/svm_random52_median_final.joblib', owner='B',
         val_ref=0.4404, val_tol=0.001, source='comparators_results.json', required=True),
    dict(method='RF_all90', path='models/rf_all90_final.joblib', owner='B',
         val_ref=0.4644, val_tol=0.001, source='baseline_results.json', required=True),
    dict(method='RF_SA52', path='models/rf_sa52_final.joblib', owner='B',
         val_ref=0.4522, val_tol=0.001, source='sa_main_results.json', required=True),
    dict(method='Ensemble_SVM_RF_all90', path='models/ensemble/ensemble_svm_rf_all90_final.joblib', owner='B',
         val_ref=0.5239, val_tol=0.001, source='comparators_results.json', required=True),
    dict(method='Ensemble_SVM_RF_SA52', path='models/ensemble/ensemble_svm_rf_sa52_final.joblib', owner='B',
         val_ref=0.5055, val_tol=0.001, source='comparators_results.json', required=True),
    # MLP: model final A memakai seed 0, jadi rujukannya skor seed pertama di
    # mlp_val_macro_f1_per_seed (rata-rata 5 seed: 0,4769 / 0,4771 / 0,4523).
    dict(method='MLP_full90', path='models/mlp_full90.joblib', owner='A',
         val_ref=0.4933, val_tol=0.001, source='mlp_results.json (seed 0)', required=True),
    dict(method='MLP_SA52', path='models/mlp_sa52.joblib', owner='A',
         val_ref=0.4473, val_tol=0.001, source='mlp_results.json (seed 0)', required=True),
    dict(method='MLP_MI52', path='models/mlp_mi52.joblib', owner='A',
         val_ref=0.4127, val_tol=0.001, source='mlp_results.json (seed 0)', required=True),
]

# Perbandingan berpasangan yang dilaporkan di paper (A - B). Pertanyaan riset: SA vs semua fitur.
COMPARISONS = [
    ('SVM_SA52', 'SVM_all90'),
    ('SVM_SA52', 'SVM_MI52'),
    ('SVM_SA52', 'SVM_HC48'),
    ('SVM_SA52', 'SVM_Random52_median'),
    ('RF_SA52', 'RF_all90'),
    ('Ensemble_SVM_RF_SA52', 'Ensemble_SVM_RF_all90'),
    ('MLP_SA52', 'MLP_full90'),
]
CONFUSION_METHODS = ['SVM_all90', 'SVM_SA52', 'SVM_MI52']


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def feature_columns():
    """90 nama fitur dalam urutan indeks resmi (feature_names.csv, Anggota A)."""
    names = pd.read_csv(FEATURE_NAMES_CSV).sort_values('index')
    return list(names['name'])


def load_frames(include_test):
    """Kembalikan (train, validation, test atau None).

    include_test=False: baris aktor test dibuang segera setelah CSV dibaca,
    sehingga tidak ada fitur atau label test yang tersentuh di mode dryrun.
    """
    splits = ds.load_splits(REPO_ROOT)
    df = pd.read_csv(FEATURES_CSV)
    if not include_test:
        df = df[~df['actor'].isin(splits['test'])].reset_index(drop=True)
    tr, va, te = ds.split_frames(df, splits)
    if not include_test:
        assert len(te) == 0
        te = None
    return tr, va, te


def matrix(frame, cols, indices):
    return frame[cols].to_numpy(float)[:, list(indices)]


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def load_bundle(path):
    b = joblib.load(path)
    if not isinstance(b, dict) or 'pipeline' not in b or 'selected_indices' not in b:
        raise ValueError(f'{path}: bundle harus dict dengan kunci pipeline dan selected_indices')
    return b


def to_labels(pred, bundle):
    """Ubah keluaran predict() menjadi nama emosi.

    Model B memakai label string. MLP Anggota A memakai kode 0-7 dengan
    bundle['label_classes'] sebagai daftar nama sesuai kode.
    """
    pred = np.asarray(pred)
    if pred.dtype.kind in 'iu':
        names = bundle.get('label_classes')
        if names is None:
            raise ValueError('Model memprediksi kode angka tetapi bundle tidak punya label_classes')
        return np.asarray(names)[pred]
    return pred.astype(str)


def to_codes(y, bundle):
    """Kebalikan to_labels, untuk fit ulang pipeline yang dilatih dengan kode angka."""
    classes = getattr(bundle['pipeline'], 'classes_', None)
    if classes is not None and np.asarray(classes).dtype.kind in 'iu':
        lookup = {name: i for i, name in enumerate(bundle['label_classes'])}
        return np.array([lookup[v] for v in y])
    return np.asarray(y)


# ----------------------------------------------------------------------------
# Dryrun: validation saja
# ----------------------------------------------------------------------------
def dryrun(methods=None, verbose=True):
    methods = METHODS if methods is None else methods
    cols = feature_columns()
    tr, va, _ = load_frames(include_test=False)
    y_tr, y_va = tr['emotion'].to_numpy(), va['emotion'].to_numpy()
    rows = []
    for m in methods:
        path = os.path.join(REPO_ROOT, m['path'])
        row = dict(method=m['method'], owner=m['owner'], path=m['path'], k='',
                   val_ref=m['val_ref'], val_macro_f1='', val_uar='', diff='', status='', note='')
        if not os.path.exists(path):
            row.update(status='BELUM ADA', note='model final belum diunggah')
            rows.append(row)
            if verbose:
                print(f"  [BELUM ADA] {m['method']:22s} {m['path']}")
            continue
        b = load_bundle(path)
        idx = list(b['selected_indices'])
        est = clone(b['pipeline'])
        t0 = time.perf_counter()
        est.fit(matrix(tr, cols, idx), to_codes(y_tr, b))
        pred = to_labels(est.predict(matrix(va, cols, idx)), b)
        fit_s = time.perf_counter() - t0
        met = ev.evaluate(y_va, pred)
        diff = met['macro_f1'] - m['val_ref']
        ok = abs(diff) <= m['val_tol']
        # Model final juga harus bisa dimuat dan memprediksi label yang sah.
        final_pred = to_labels(b['pipeline'].predict(matrix(va.head(8), cols, idx)), b)
        labels_ok = set(final_pred) <= set(ev.LABELS)
        row.update(k=len(idx), val_macro_f1=round(met['macro_f1'], 4), val_uar=round(met['uar'], 4),
                   diff=round(diff, 4), status='OK' if ok and labels_ok else 'BEDA',
                   note=f"rujukan: {m['source']}; toleransi {m['val_tol']}; fit+prediksi {fit_s:.1f} s")
        if not labels_ok:
            row['note'] += '; label model final tidak dikenal'
        rows.append(row)
        if verbose:
            print(f"  [{row['status']:9s}] {m['method']:22s} k={len(idx):2d}  val F1 {met['macro_f1']:.4f}"
                  f"  rujukan {m['val_ref']:.4f}  selisih {diff:+.4f}")
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'dryrun_validation.csv'), index=False)
    summary = {
        'runner_version': RUNNER_VERSION, 'evaluate_version': ev.EVALUATE_VERSION,
        'run_at': datetime.now().isoformat(timespec='seconds'),
        'data': 'train (aktor 1-16) untuk fit, validation (aktor 17-20) untuk nilai; test tidak dibaca',
        'counts': {s: sum(r['status'] == s for r in rows) for s in ('OK', 'BEDA', 'BELUM ADA')},
        'rows': rows,
    }
    with open(os.path.join(OUT_DIR, 'dryrun_validation.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


# ----------------------------------------------------------------------------
# Mode test
# ----------------------------------------------------------------------------
def preflight(confirm):
    """Daftar alasan mode test DITOLAK (kosong = boleh jalan). Tidak membaca data test."""
    methods, table_path, lock_path, audit_path = METHODS, TABLE_PATH, LOCK_PATH, AUDIT_JSON
    problems = []
    if confirm != CONFIRM_PHRASE:
        problems.append(f'frasa konfirmasi harus persis "{CONFIRM_PHRASE}"')
    if os.path.exists(lock_path):
        problems.append(f'test sudah pernah dievaluasi ({os.path.relpath(lock_path, REPO_ROOT)} ada)')
    if os.path.exists(table_path) and os.path.getsize(table_path) > 0:
        t = pd.read_csv(table_path, dtype=str)
        if t['split'].fillna('').str.startswith('test').any():
            problems.append('main_table.csv sudah berisi baris test')
    missing = [m['method'] for m in methods
               if m['required'] and not os.path.exists(os.path.join(REPO_ROOT, m['path']))]
    if missing:
        problems.append(f'model wajib belum ada: {missing}')
    if not os.path.exists(audit_path):
        problems.append('results/audit_leakage.json belum ada; jalankan src/audit_leakage.py')
    else:
        with open(audit_path, encoding='utf-8') as f:
            audit = json.load(f)
        n_fail = sum(1 for c in audit.get('results', []) if c.get('status') == 'FAIL')
        if not audit.get('results'):
            problems.append('results/audit_leakage.json kosong; jalankan ulang src/audit_leakage.py')
        elif n_fail:
            problems.append(f'audit anti-leakage masih {n_fail} FAIL')
    return problems


def run_test(confirm):
    problems = preflight(confirm)
    if problems:
        raise SystemExit('Mode test DITOLAK:\n  - ' + '\n  - '.join(problems))

    print('Dryrun validation sebelum test ...')
    dr = dryrun(verbose=True)
    if dr['counts']['BEDA'] or dr['counts']['BELUM ADA']:
        raise SystemExit(f"Mode test DITOLAK: dryrun belum bersih {dr['counts']}")

    ev.set_seed(SEED)
    cols = feature_columns()
    _, _, te = load_frames(include_test=True)
    y_te = te['emotion'].to_numpy()
    preds, results = {}, {}
    out = te[['file', 'actor', 'gender', 'emotion', 'intensity']].copy()
    for m in METHODS:
        b = load_bundle(os.path.join(REPO_ROOT, m['path']))
        idx = list(b['selected_indices'])
        t0 = time.perf_counter()
        pred = to_labels(b['pipeline'].predict(matrix(te, cols, idx)), b)
        runtime = time.perf_counter() - t0
        met = ev.evaluate(y_te, pred)
        _, lo, hi = ev.bootstrap_ci(y_te, pred, seed=SEED)
        row = ev.result_row(m['method'], 'test', met, n_features=len(idx), ci=(lo, hi),
                            runtime_s=round(runtime, 4), run_id=os.path.basename(m['path']), seed=SEED,
                            note='model final train+validation; predict(); evaluasi test sekali oleh C')
        ev.append_result(row, path=TABLE_PATH, allow_test=True)
        preds[m['method']] = pred
        out[m['method']] = pred
        results[m['method']] = {
            'k': len(idx), 'accuracy': met['accuracy'], 'macro_f1': met['macro_f1'], 'uar': met['uar'],
            'ci_low': lo, 'ci_high': hi, 'per_class_f1': met['per_class_f1'],
            'confusion_matrix': met['confusion_matrix'].tolist(), 'model_file': m['path'],
        }
        if m['method'] in CONFUSION_METHODS:
            ev.plot_confusion(met['confusion_matrix'], title=f"{m['method']} (test)",
                              save_path=os.path.join(FIG_DIR, f"cm_{m['method']}.png"))
        print(f"  {m['method']:22s} k={len(idx):2d}  macro F1 {met['macro_f1']:.4f}  [{lo:.4f}, {hi:.4f}]")

    pairs = []
    for a, b in COMPARISONS:
        pb = ev.paired_bootstrap(y_te, preds[a], preds[b], seed=SEED)
        mc = ev.mcnemar_test(y_te, preds[a], preds[b])
        pairs.append({'a': a, 'b': b, **pb, 'mcnemar': mc})

    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(os.path.join(OUT_DIR, 'test_predictions.csv'), index=False)
    summary = {
        'runner_version': RUNNER_VERSION, 'evaluate_version': ev.EVALUATE_VERSION,
        'evaluated_at': datetime.now().isoformat(timespec='seconds'), 'seed': SEED,
        'n_test': int(len(te)), 'test_actors': sorted(int(a) for a in te['actor'].unique()),
        'results': results, 'paired_comparisons': pairs,
        'note': ('CI bootstrap per file (1.000 ulangan); file dari aktor yang sama tidak independen, '
                 'jadi interval cenderung terlalu sempit. McNemar membandingkan accuracy, bukan macro F1.'),
    }
    with open(os.path.join(OUT_DIR, 'test_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(LOCK_PATH, 'w', encoding='utf-8') as f:
        json.dump({'evaluated_at': summary['evaluated_at'], 'methods': list(results)}, f, indent=2)
    print(f'Selesai. Hasil: {os.path.relpath(OUT_DIR, REPO_ROOT)}; kunci: {os.path.relpath(LOCK_PATH, REPO_ROOT)}')
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['dryrun', 'test'], default='dryrun')
    p.add_argument('--confirm', default='', help=f'wajib untuk mode test: "{CONFIRM_PHRASE}"')
    args = p.parse_args(argv)
    if args.mode == 'dryrun':
        print('Dryrun: fit ulang di train, nilai di validation. Data test tidak dibaca.')
        s = dryrun()
        print(f"Ringkasan: {s['counts']}")
        print(f'Hasil: {os.path.relpath(OUT_DIR, REPO_ROOT)}/dryrun_validation.csv')
        problems = preflight(CONFIRM_PHRASE)
        print('Kesiapan mode test: ' + ('SIAP' if not problems else 'BELUM SIAP'))
        for pr in problems:
            print(f'  - {pr}')
    else:
        run_test(args.confirm)


if __name__ == '__main__':
    main()
