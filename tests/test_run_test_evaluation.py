"""
Uji src/run_test_evaluation.py dengan data dan model tiruan (tanpa data proyek).

Tujuannya memastikan mode test yang hanya boleh dijalankan SEKALI tidak gagal
di tengah jalan, dan semua pengamannya bekerja.

Jalankan dari root repo:
    python -m pytest tests/
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import evaluate as ev  # noqa: E402
import run_test_evaluation as rt  # noqa: E402

N_FEAT = 6


def _fake_repo(tmp_path, monkeypatch):
    """Repo tiruan: 24 aktor x 16 file, 6 fitur, dua model (label string dan kode angka)."""
    rng = np.random.default_rng(0)
    names = [f'f{i}' for i in range(N_FEAT)]
    rows = []
    for actor in range(1, 25):
        for j in range(16):
            emo = ev.LABELS[j % 8]
            x = rng.normal(size=N_FEAT) + ev.LABELS.index(emo) * 0.8
            rows.append({'file': f'a{actor}_{j}.wav', 'actor': actor, 'gender': 'male' if actor % 2 else 'female',
                         'emotion': emo, 'intensity': 'normal', **dict(zip(names, x))})
    df = pd.DataFrame(rows)
    (tmp_path / 'data' / 'features').mkdir(parents=True)
    df.to_csv(tmp_path / 'data' / 'features' / 'f.csv', index=False)
    pd.DataFrame({'index': range(N_FEAT), 'name': names}).to_csv(
        tmp_path / 'data' / 'features' / 'names.csv', index=False)
    split = {'train': list(range(1, 17)), 'validation': list(range(17, 21)), 'test': list(range(21, 25))}
    (tmp_path / 'data' / 'splits.json').write_text(json.dumps({'split': split}))

    trval = df[df['actor'] <= 20]
    X = trval[names].to_numpy(float)
    y = trval['emotion'].to_numpy()
    (tmp_path / 'models').mkdir()
    p_str = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=500))])
    p_str.fit(X[:, [0, 1, 2, 3]], y)
    joblib.dump({'pipeline': p_str, 'selected_indices': [0, 1, 2, 3]}, tmp_path / 'models' / 'str.joblib')
    codes = np.array([ev.LABELS.index(v) for v in y])
    p_int = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=500))])
    p_int.fit(X, codes)
    joblib.dump({'pipeline': p_int, 'selected_indices': list(range(N_FEAT)), 'label_classes': ev.LABELS},
                tmp_path / 'models' / 'int.joblib')

    # Rujukan validation dihitung dari fit ulang di train, sama seperti dryrun.
    methods = [dict(method='M_str', path='models/str.joblib', owner='T', val_ref=0.0, val_tol=1.0,
                    source='tiruan', required=True),
               dict(method='M_int', path='models/int.joblib', owner='T', val_ref=0.0, val_tol=1.0,
                    source='tiruan', required=True)]
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps({'results': [{'status': 'OK'}, {'status': 'WARN'}]}))
    for name, value in {
        'REPO_ROOT': str(tmp_path),
        'FEATURES_CSV': str(tmp_path / 'data' / 'features' / 'f.csv'),
        'FEATURE_NAMES_CSV': str(tmp_path / 'data' / 'features' / 'names.csv'),
        'AUDIT_JSON': str(audit),
        'OUT_DIR': str(tmp_path / 'out'),
        'FIG_DIR': str(tmp_path / 'fig'),
        'LOCK_PATH': str(tmp_path / 'out' / 'TEST_DONE.json'),
        'TABLE_PATH': str(tmp_path / 'main_table.csv'),
        'METHODS': methods,
        'COMPARISONS': [('M_str', 'M_int')],
        'CONFUSION_METHODS': ['M_str'],
    }.items():
        monkeypatch.setattr(rt, name, value)
    return tmp_path


def test_dryrun_never_reads_test_rows(tmp_path, monkeypatch):
    _fake_repo(tmp_path, monkeypatch)
    tr, va, te = rt.load_frames(include_test=False)
    assert te is None
    assert set(tr['actor']) == set(range(1, 17))
    assert set(va['actor']) == set(range(17, 21))


def test_to_labels_maps_codes_and_strings():
    b = {'label_classes': ev.LABELS}
    assert list(rt.to_labels(np.array([0, 7]), b)) == ['neutral', 'surprised']
    assert list(rt.to_labels(np.array(['sad']), {})) == ['sad']
    with pytest.raises(ValueError):
        rt.to_labels(np.array([1]), {})


def test_dryrun_handles_both_label_types(tmp_path, monkeypatch):
    _fake_repo(tmp_path, monkeypatch)
    s = rt.dryrun(verbose=False)
    assert s['counts'] == {'OK': 2, 'BEDA': 0, 'BELUM ADA': 0}
    assert all(0 < r['val_macro_f1'] <= 1 for r in s['rows'])
    assert not os.path.exists(rt.TABLE_PATH)


def test_dryrun_flags_mismatch_and_missing(tmp_path, monkeypatch):
    _fake_repo(tmp_path, monkeypatch)
    methods = [dict(rt.METHODS[0], val_ref=0.01, val_tol=0.001),
               dict(rt.METHODS[1], method='M_none', path='models/none.joblib')]
    s = rt.dryrun(methods=methods, verbose=False)
    assert s['counts'] == {'OK': 0, 'BEDA': 1, 'BELUM ADA': 1}


def test_preflight_guards(tmp_path, monkeypatch):
    _fake_repo(tmp_path, monkeypatch)
    assert rt.preflight(rt.CONFIRM_PHRASE) == []
    assert any('frasa' in p for p in rt.preflight('evaluasi test sekali'))
    # audit dengan FAIL
    with open(rt.AUDIT_JSON, 'w') as f:
        json.dump({'results': [{'status': 'FAIL'}]}, f)
    assert any('FAIL' in p for p in rt.preflight(rt.CONFIRM_PHRASE))


def test_preflight_missing_model(tmp_path, monkeypatch):
    _fake_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(rt, 'METHODS', rt.METHODS + [dict(rt.METHODS[0], method='X', path='models/x.joblib')])
    assert any('belum ada' in p for p in rt.preflight(rt.CONFIRM_PHRASE))


def test_run_test_end_to_end_then_locked(tmp_path, monkeypatch):
    _fake_repo(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        rt.run_test('salah')
    assert not os.path.exists(rt.TABLE_PATH)

    s = rt.run_test(rt.CONFIRM_PHRASE)
    assert s['n_test'] == 64 and s['test_actors'] == [21, 22, 23, 24]
    table = pd.read_csv(rt.TABLE_PATH)
    assert list(table['method']) == ['M_str', 'M_int']
    assert set(table['split']) == {'test'}
    assert (table['ci_low'] <= table['macro_f1']).all() and (table['macro_f1'] <= table['ci_high']).all()
    pred = pd.read_csv(os.path.join(rt.OUT_DIR, 'test_predictions.csv'))
    assert len(pred) == 64 and set(pred['M_int']) <= set(ev.LABELS)
    assert os.path.exists(os.path.join(rt.FIG_DIR, 'cm_M_str.png'))
    assert s['paired_comparisons'][0]['a'] == 'M_str'

    # Kedua kalinya harus ditolak: kunci ada dan main_table sudah berisi test.
    problems = rt.preflight(rt.CONFIRM_PHRASE)
    assert any('sudah pernah' in p for p in problems)
    assert any('main_table' in p for p in problems)
    with pytest.raises(SystemExit):
        rt.run_test(rt.CONFIRM_PHRASE)
    assert len(pd.read_csv(rt.TABLE_PATH)) == 2
