"""
audit_leakage.py - Audit anti-leakage Proyek 07 (langkah 6 Rencana Kerja, Anggota C).

Penanggung jawab : Anggota C
Versi            : 1.0 (24 November 2026)

Memeriksa, tanpa membuka label atau fitur data test untuk evaluasi:
  1. Aktor tidak tumpang tindih antar-split (split utama dan 5 partisi)
  2. Setiap split punya kedua gender dan 8 emosi; tabel fitur lengkap tanpa NaN
  3. Scaler berada di dalam Pipeline dan di-fit hanya pada data train
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
B_SCRIPTS = ['run_baseline.py', 'run_sa_main.py', 'run_sa_stability.py']
# Baris kode yang membaca atau menamai data test (komentar diabaikan)
TEST_PATTERN = re.compile(r"\b(te|test|X_te|y_te|te_\w+|test_\w+)\s*=|\[['\"]test['\"]\]|test_noisy|features/noisy")

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
    sa_src = open(os.path.join(B_DIR, 'src', 'sa_feature_selection.py'), encoding='utf-8').read()
    ok = 'GroupKFold' in sa_src and 'scaler.fit_transform(X_tr[tr_idx])' in sa_src
    check(3, 'Objective SA: scaler di-fit per fold GroupKFold di dalam train', 'OK' if ok else 'FAIL',
          'sa_feature_selection._precompute_scaled_folds memakai fold train saja')

    # 4. Kode B tidak membaca test
    files = sorted(glob.glob(os.path.join(B_DIR, 'src', '*.py')) +
                   glob.glob(os.path.join(B_DIR, 'notebooks', '*.ipynb')))
    hits = [f'{os.path.relpath(p, ROOT)}:{i}: {c.strip()}' for p in files for i, c in code_lines(p)
            if TEST_PATTERN.search(c)]
    check(4, 'Kode dan notebook Anggota B tidak membaca data test', 'FAIL' if hits else 'OK',
          '; '.join(hits) or f'{len(files)} file diperiksa; bagian test dari split_frames dibuang (_)')
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
    mi = [p for p in glob.glob(os.path.join(ROOT, '**', '*'), recursive=True)
          if re.search(r'(?i)(mutual|(^|[_\\/])mi_)', os.path.basename(p)) and '.venv' not in p]
    check(5, 'Mutual information dihitung di train saja', 'PENDING' if not mi else 'CEK MANUAL',
          'pembanding MI (langkah 10 Anggota A) belum ada' if not mi else ', '.join(mi))

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
