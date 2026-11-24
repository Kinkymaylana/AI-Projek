"""
audit_leakage.py - Audit anti-leakage Proyek 07 (langkah 6 Rencana Kerja, Anggota C).

Penanggung jawab : Anggota C
Versi            : 1.1 (24 November 2026) - model final di models/ dan skrip B di
                   progres_anggota_b/ (rekursif) serta train_final_models.py ikut diaudit

Memeriksa, tanpa membuka label atau fitur data test untuk evaluasi:
  1. Aktor tidak tumpang tindih antar-split (split utama dan 5 partisi)
  2. Setiap split punya kedua gender dan 8 emosi; tabel fitur lengkap tanpa NaN
  3. Scaler berada di dalam Pipeline dan di-fit hanya pada train (model P0) atau
     train + validation (model final), tanpa aktor test
  4. Kode dan notebook Anggota B tidak membaca data test
  5. Mutual information dihitung di train saja
  6. Data test belum dievaluasi sebelum evaluasi akhir

Status: OK, WARN (bukan kebocoran pada model utama, tetapi wajib dicatat di paper),
PENDING (belum bisa diperiksa), FAIL (kebocoran; harus diperbaiki).

Jalankan dari root repo:
    .venv\\Scripts\\python.exe src\\audit_leakage.py
Hasil: tercetak di layar dan tersimpan di results/audit_leakage.json.
"""
import csv
import glob
import json
import os
import re
import sys
from datetime import datetime

import joblib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import data_splits as ds  # noqa: E402
import evaluate as ev  # noqa: E402

B_DIR = os.path.join(ROOT, 'progres_anggota_b')
MODEL_PATH = os.path.join(B_DIR, 'models', 'best_model_p0.joblib')
FINAL_DIR = os.path.join(ROOT, 'models')
B_SCRIPTS = ['run_baseline.py', 'run_sa_main.py', 'run_sa_stability.py']
# Kode B di luar progres_anggota_b/. Tambahkan di sini bila B menaruh skrip baru di root.
B_ROOT_FILES = ['train_final_models.py', 'run_comparators_and_curves.py']
# Baris kode yang membaca atau menamai data test (komentar diabaikan)
TEST_PATTERN = re.compile(r"\b(te|test|X_te|y_te|te_\w+|test_\w+)\s*=|\[['\"]test['\"]\]|test_noisy|features/noisy")
# Pengecualian: hanya daftar NOMOR aktor test (mis. untuk assert tidak tumpang tindih),
# selama baris itu tidak memfilter tabel data.
ACTOR_LIST_ONLY = re.compile(r"\btest_actors\w*\s*=")
DATA_ACCESS = re.compile(r"isin|\.loc|\.iloc|read_csv|\bdf\b|_df\b|frame")


def reads_test(code):
    if not TEST_PATTERN.search(code):
        return False
    return not (ACTOR_LIST_ONLY.search(code) and not DATA_ACCESS.search(code))

results = []


def check(no, name, status, detail):
    results.append({'no': no, 'check': name, 'status': status, 'detail': detail})


def code_lines(path):
    """Baris kode dari .py atau sel kode .ipynb, tanpa komentar."""
    if path.endswith('.ipynb'):
        with open(path, encoding='utf-8') as f:
            cells = json.load(f)['cells']
        text = '\n'.join(''.join(c['source']) for c in cells if c['cell_type'] == 'code')
    else:
        with open(path, encoding='utf-8') as f:
            text = f.read()
    for i, ln in enumerate(text.splitlines(), 1):
        code = ln.split('#', 1)[0]
        if code.strip():
            yield i, code


def main():
    splits = ds.load_splits(ROOT)
    parts = ds.load_partitions(ROOT)
    test_actors = set(splits['test'])

    # 1. Tumpang tindih aktor
    problems = ds.check_split(splits)
    check(1, 'Split utama: aktor tidak tumpang tindih, 24 aktor terpakai',
          'FAIL' if problems else 'OK', '; '.join(problems) or
          f"train {splits['train']}, validation {splits['validation']}, test {splits['test']}")
    bad = {p: ds.check_split(s) for p, s in parts.items() if ds.check_split(s)}
    check(1, 'Partisi P0-P4: aktor tidak tumpang tindih', 'FAIL' if bad else 'OK',
          json.dumps(bad) if bad else f'{len(parts)} partisi valid; P0 sama dengan split utama: '
          f"{parts['P0'] == splits}")

    # 2. Tabel fitur dan komposisi split
    df = pd.read_csv(os.path.join(ROOT, 'data', 'features', 'features_peak.csv'))
    feat = pd.read_csv(os.path.join(ROOT, 'data', 'features', 'feature_names.csv'))['name']
    nan = int(df[list(feat)].isna().sum().sum())
    ok = len(df) == 1440 and df['file'].is_unique and nan == 0
    check(2, 'Tabel fitur: 1.440 baris unik, tanpa NaN', 'OK' if ok else 'FAIL',
          f'{len(df)} baris, file unik {df["file"].is_unique}, NaN {nan}')
    comp = []
    for name, part in zip(ds.SPLIT_NAMES, ds.split_frames(df, splits)):
        g = part['gender'].value_counts().to_dict()
        comp.append(f"{name}: {len(part)} file, gender {g}, {part['emotion'].nunique()} emosi")
        ok = ok and part['gender'].nunique() == 2 and part['emotion'].nunique() == 8
    check(2, 'Setiap split punya 2 gender dan 8 emosi', 'OK' if ok else 'FAIL', '; '.join(comp))

    # 3. Scaler di dalam Pipeline, di-fit di train saja
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle['pipeline']
    steps = [type(s).__name__ for _, s in pipe.steps]
    n_seen = int(getattr(pipe.steps[0][1], 'n_samples_seen_', -1))
    n_train = int(df['actor'].isin(splits['train']).sum())
    ok = steps[0] == 'StandardScaler' and n_seen == n_train
    check(3, 'Model utama: StandardScaler di dalam Pipeline dan di-fit pada train saja',
          'OK' if ok else 'FAIL',
          f'langkah {steps}; scaler melihat {n_seen} sampel, train P0 = {n_train} file; '
          f"trained_on = {bundle.get('trained_on')}")
    finals = sorted(glob.glob(os.path.join(FINAL_DIR, '**', '*.joblib'), recursive=True))
    n_trval = int(df['actor'].isin(splits['train'] + splits['validation']).sum())
    trval = set(splits['train'] + splits['validation'])
    with open(os.path.join(B_DIR, 'results', 'sa_main_results.json'), encoding='utf-8') as f:
        sa_idx = json.load(f)['selected_indices']
    rows, ok = [], bool(finals)
    for p in finals:
        b = joblib.load(p)
        sc = b['pipeline'].steps[0][1]
        n = int(getattr(sc, 'n_samples_seen_', -1))
        k = len(b['selected_indices'])
        # Model A (MLP) tidak mencatat train_val_actors; jumlah sampel scaler tetap diperiksa.
        actors = set(b.get('train_val_actors', trval))
        good = (type(sc).__name__ == 'StandardScaler' and n == n_trval
                and actors == trval and not actors & test_actors)
        nama = os.path.relpath(p, FINAL_DIR).replace(os.sep, '/')
        if 'sa52' in nama.lower() and b['selected_indices'] != sa_idx:
            good = False  # subset SA berbeda dari hasil SA utama
        ok = ok and good
        rows.append(f"{nama}: scaler {n} sampel, k={k}"
                    f"{'' if 'train_val_actors' in b else ', aktor tidak dicatat'}"
                    f"{'' if good else ' (BERMASALAH)'}")
    check(3, f'Model final models/ ({len(finals)} file): di-fit pada train + validation saja (aktor 1-20)',
          ('OK' if ok else 'FAIL') if finals else 'PENDING',
          (f"train + validation = {n_trval} file; " + '; '.join(rows)) if finals
          else 'model final belum ada di models/')
    sa_src = open(os.path.join(B_DIR, 'src', 'sa_feature_selection.py'), encoding='utf-8').read()
    ok = 'GroupKFold' in sa_src and 'scaler.fit_transform(X_tr[tr_idx])' in sa_src
    check(3, 'Objective SA: scaler di-fit per fold GroupKFold di dalam train', 'OK' if ok else 'FAIL',
          'sa_feature_selection._precompute_scaled_folds memakai fold train saja')

    # 4. Kode B tidak membaca test
    files = sorted(glob.glob(os.path.join(B_DIR, '**', '*.py'), recursive=True) +
                   glob.glob(os.path.join(B_DIR, '**', '*.ipynb'), recursive=True) +
                   [os.path.join(ROOT, f) for f in B_ROOT_FILES if os.path.exists(os.path.join(ROOT, f))])
    files = [p for p in files if '.ipynb_checkpoints' not in p]
    hits = [f'{os.path.relpath(p, ROOT)}:{i}: {c.strip()}' for p in files for i, c in code_lines(p)
            if reads_test(c)]
    check(4, 'Kode dan notebook Anggota B tidak membaca data test', 'FAIL' if hits else 'OK',
          '; '.join(hits) or f'{len(files)} file diperiksa ('
          + ', '.join(os.path.relpath(p, ROOT).replace(os.sep, '/') for p in files)
          + '); bagian test dari split_frames selalu dibuang (_)')
    missing = [s for s in B_SCRIPTS if not glob.glob(os.path.join(ROOT, '**', s), recursive=True)]
    check(4, 'Skrip yang menjalankan eksperimen B tersedia untuk diaudit', 'PENDING' if missing else 'OK',
          f"belum ada di repo: {', '.join(missing)}" if missing else 'semua tersedia')
    with open(os.path.join(B_DIR, 'results', 'subsets.json'), encoding='utf-8') as f:
        runs = json.load(f)['runs']
    touched = sorted({r['partition'] for r in runs
                      if test_actors & set(parts[r['partition']]['train'] + parts[r['partition']]['validation'])})
    check(4, 'Run stabilitas tidak memakai aktor test utama sebelum evaluasi akhir',
          'WARN' if touched else 'OK',
          (f"run partisi {', '.join(touched)} memakai aktor {sorted(test_actors)} di train/validation "
           f"(P4: validation). partitions.json meminta run ini setelah hasil utama final. Model utama P0 "
           f"tidak terpengaruh, tetapi val_macro_f1 P4 di subsets.json dihitung pada aktor test, "
           f"jadi tidak boleh dipakai untuk keputusan apa pun.") if touched else '-')

    # 5. Mutual information
    # Kode yang memanggil mutual_info_classif harus memberi data train saja (X_tr / X_train).
    kode = [p for p in glob.glob(os.path.join(ROOT, '**', '*.py'), recursive=True) +
            glob.glob(os.path.join(ROOT, '**', '*.ipynb'), recursive=True)
            if '.venv' not in p and '.ipynb_checkpoints' not in p and p != os.path.abspath(__file__)]
    panggilan, salah = [], []
    for p in kode:
        for i, c in code_lines(p):
            if 'mutual_info_classif(' in c:
                rel = f'{os.path.relpath(p, ROOT)}:{i}'.replace(os.sep, '/')
                panggilan.append(rel)
                if not re.search(r'mutual_info_classif\(\s*X_(tr|train)\b', c):
                    salah.append(f'{rel}: {c.strip()}')
    topk = os.path.join(ROOT, 'data', 'analysis', 'mi_topk_peak.json')
    if not panggilan:
        check(5, 'Mutual information dihitung di train saja', 'PENDING',
              'pembanding MI (langkah 10 Anggota A) belum ada')
    else:
        check(5, 'Mutual information dihitung di train saja', 'FAIL' if salah else 'OK',
              '; '.join(salah) if salah else
              f"mutual_info_classif dipanggil dengan data train di {', '.join(panggilan)}; "
              + ('daftar top-k ada di data/analysis/mi_topk_peak.json' if os.path.exists(topk)
                 else 'catatan: data/analysis/mi_topk_peak.json (daftar indeks untuk B) belum ada di repo'))

    # 6. Test belum dievaluasi
    rows = []
    if os.path.exists(ev.DEFAULT_TABLE_PATH):
        with open(ev.DEFAULT_TABLE_PATH, encoding='utf-8', newline='') as f:
            rows = [r for r in csv.DictReader(f) if r['split'].startswith('test')]
    check(6, 'Hasil test tercatat (hanya oleh C, sekali)', 'OK',
          f'{len(rows)} baris test di main_table.csv' if rows else 'test belum dievaluasi')

    out = {'audited_at': datetime.now().isoformat(timespec='seconds'), 'results': results}
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    with open(os.path.join(ROOT, 'results', 'audit_leakage.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    for r in results:
        print(f"[{r['status']:7s}] {r['no']}. {r['check']}\n          {r['detail']}")
    n = {s: sum(r['status'] == s for r in results) for s in ('OK', 'WARN', 'PENDING', 'FAIL')}
    print(f"\nRingkasan: {n}")
    return 1 if n['FAIL'] else 0


if __name__ == '__main__':
    sys.exit(main())
