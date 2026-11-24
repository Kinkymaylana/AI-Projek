"""
Uji src/run_noise_evaluation.py dengan data dan model tiruan (tanpa data proyek).

Jalankan dari root repo:
    python -m pytest tests/
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import run_noise_evaluation as rn  # noqa: E402
import run_test_evaluation as rt  # noqa: E402
from test_run_test_evaluation import _fake_repo  # noqa: E402


def _fake_noise(tmp_path, monkeypatch, run_clean=True):
    _fake_repo(tmp_path, monkeypatch)
    # Repo tiruan berisi 16 file per aktor; test bernoise harus 240 file seperti aslinya,
    # jadi cek jumlah dilonggarkan lewat salinan tabel test bersih.
    monkeypatch.setattr(rt, 'METHODS', [dict(m, method=n) for m, n in
                                        zip(rt.METHODS, ['SVM_SA52', 'SVM_all90'])])
    monkeypatch.setattr(rt, 'COMPARISONS', [('SVM_SA52', 'SVM_all90')])
    monkeypatch.setattr(rt, 'CONFUSION_METHODS', ['SVM_SA52'])
    noisy_dir = tmp_path / 'data' / 'features' / 'noisy'
    noisy_dir.mkdir(parents=True)
    df = pd.read_csv(rt.FEATURES_CSV)
    te = df[df['actor'] >= 21]
    rng = np.random.default_rng(1)
    feat = [c for c in df.columns if c[0] == 'f' and c[1:].isdigit()]
    for kind, snr in rn.CONDITIONS:
        noisy = te.copy()
        noisy[feat] = noisy[feat] + rng.normal(scale=(30 - snr) / 10, size=noisy[feat].shape)
        noisy.to_csv(noisy_dir / f'test_noisy_{kind}_{snr}db_peak.csv', index=False)
    for name, value in {
        'NOISY_DIR': str(noisy_dir),
        'CLEAN_PREDICTIONS': os.path.join(rt.OUT_DIR, 'test_predictions.csv'),
        'CLEAN_SUMMARY': os.path.join(rt.OUT_DIR, 'test_summary.json'),
        'OUT_DIR': str(tmp_path / 'noise_out'),
        'LOCK_PATH': str(tmp_path / 'noise_out' / 'NOISE_DONE.json'),
        'TABLE_PATH': rt.TABLE_PATH,
        'COMPARISONS': [('SVM_SA52', 'SVM_all90')],
    }.items():
        monkeypatch.setattr(rn, name, value)
    if run_clean:
        rt.run_test(rt.CONFIRM_PHRASE)
    return len(te)


def _allow_small_tables(monkeypatch, n):
    real = rn.check_tables

    def patched():
        return [p for p in real() if 'harus 240' not in p]
    monkeypatch.setattr(rn, 'check_tables', patched)


def test_noise_refused_before_clean_test(tmp_path, monkeypatch):
    n = _fake_noise(tmp_path, monkeypatch, run_clean=False)
    _allow_small_tables(monkeypatch, n)
    problems = rn.preflight(rn.CONFIRM_PHRASE)
    assert any('test bersih belum selesai' in p for p in problems)


def test_check_detects_bad_tables(tmp_path, monkeypatch):
    _fake_noise(tmp_path, monkeypatch)
    os.remove(rn.noisy_path('white', 0))
    bad = pd.read_csv(rn.noisy_path('esc50', 0)).drop(columns=['f0'])
    bad.to_csv(rn.noisy_path('esc50', 0), index=False)
    problems = rn.check_tables()
    assert any('white_0db' in p and 'tidak ada' in p for p in problems)
    assert any('esc50_0db' in p and 'kolom' in p for p in problems)


def test_noise_end_to_end_then_locked(tmp_path, monkeypatch):
    n = _fake_noise(tmp_path, monkeypatch)
    _allow_small_tables(monkeypatch, n)
    with pytest.raises(SystemExit):
        rn.run_noise('salah')

    s = rn.run_noise(rn.CONFIRM_PHRASE)
    table = pd.read_csv(rn.TABLE_PATH)
    noisy = table[table['split'].str.startswith('test_noisy')]
    assert len(noisy) == 6 * 2
    assert set(noisy['split']) == {rn.split_name(k, s_) for k, s_ in rn.CONDITIONS}
    r = s['results']['test_noisy_white_0db']['SVM_SA52']
    assert r['drop_from_clean'] == pytest.approx(r['clean_macro_f1'] - r['macro_f1'])
    pred = pd.read_csv(os.path.join(rn.OUT_DIR, 'noise_predictions.csv'))
    assert len(pred) == 6 * n
    assert len(s['paired_comparisons']) == 6

    problems = rn.preflight(rn.CONFIRM_PHRASE)
    assert any('sudah pernah' in p for p in problems)
    assert any('test_noisy' in p for p in problems)
    with pytest.raises(SystemExit):
        rn.run_noise(rn.CONFIRM_PHRASE)
    assert len(pd.read_csv(rn.TABLE_PATH)) == len(table)
    with open(rn.LOCK_PATH) as f:
        assert len(json.load(f)['conditions']) == 6
