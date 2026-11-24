"""
Uji src/evaluate.py dengan data dummy (tanpa data proyek).

Jalankan dari root repo:
    python tests/test_evaluate.py      atau      python -m pytest tests/
"""
import os
import sys
import tempfile
import warnings

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import evaluate as ev  # noqa: E402


def test_hand_computed():
    labels = ['neutral', 'calm', 'happy']
    m = ev.evaluate(['neutral', 'calm', 'calm', 'happy'],
                    ['neutral', 'calm', 'happy', 'happy'], labels=labels)
    # F1: neutral 1, calm 2/3, happy 2/3 -> macro 7/9; recall: 1, 1/2, 1 -> UAR 5/6
    assert np.isclose(m['accuracy'], 0.75)
    assert np.isclose(m['macro_f1'], 7 / 9)
    assert np.isclose(m['uar'], 5 / 6)
    assert m['confusion_matrix'].tolist() == [[1, 0, 0], [0, 1, 1], [0, 0, 1]]


def test_matches_sklearn_and_label_order():
    rng = np.random.default_rng(0)
    yt = rng.choice(ev.LABELS, 300)
    yp = np.where(rng.random(300) < 0.6, yt, rng.choice(ev.LABELS, 300))
    m = ev.evaluate(yt, yp)
    assert np.isclose(m['macro_f1'], f1_score(yt, yp, average='macro'))
    assert np.isclose(m['accuracy'], accuracy_score(yt, yp))
    # Baris/kolom confusion matrix selalu mengikuti LABELS, bukan urutan kemunculan
    perm = rng.permutation(300)
    m2 = ev.evaluate(yt[perm], yp[perm])
    assert (m['confusion_matrix'] == m2['confusion_matrix']).all()
    assert m['confusion_matrix'][ev.LABELS.index('neutral')].sum() == np.sum(yt == 'neutral')


def test_unknown_label_rejected():
    for bad in (['angry', 'fear'], [5, 5]):   # typo dan kode angka
        try:
            ev.evaluate(['angry', 'angry'], bad)
        except ValueError:
            continue
        raise AssertionError(f'label tidak sah {bad} lolos')


def test_absent_label_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        ev.evaluate(['angry', 'sad'], ['angry', 'sad'])
    assert any('tidak muncul' in str(x.message) for x in w)


def test_bootstrap_deterministic_and_brackets_point():
    rng = np.random.default_rng(1)
    yt = rng.choice(ev.LABELS, 240)
    yp = np.where(rng.random(240) < 0.5, yt, rng.choice(ev.LABELS, 240))
    a = ev.bootstrap_ci(yt, yp, n_boot=300, seed=7)
    b = ev.bootstrap_ci(yt, yp, n_boot=300, seed=7)
    assert a == b
    assert a[1] <= a[0] <= a[2]
    d = ev.paired_bootstrap(yt, yp, yp, n_boot=200)
    assert d['diff'] == 0 and d['ci_low'] == 0 and d['ci_high'] == 0


def test_mcnemar_exact():
    yt = ['angry'] * 10
    a = ['angry'] * 10                       # A benar semua
    b = ['sad'] * 5 + ['angry'] * 5          # B salah 5
    r = ev.mcnemar_test(yt, a, b)
    assert (r['a_only_correct'], r['b_only_correct']) == (5, 0)
    assert np.isclose(r['p_value'], 2 * 0.5 ** 5)
    assert ev.mcnemar_test(yt, a, a)['p_value'] == 1.0


def test_append_result_contract_and_test_guard():
    m = ev.evaluate(['angry', 'sad'], ['angry', 'angry'], labels=['angry', 'sad'])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'results', 'main_table.csv')
        ev.append_result(ev.result_row('SVM_full', 'validation', m, 90, run_id='r1', seed=42), path)
        ev.append_result(ev.result_row('SA', 'validation', m, 52), path)
        lines = open(path, encoding='utf-8').read().strip().splitlines()
        assert lines[0].split(',') == ev.MAIN_TABLE_COLUMNS and len(lines) == 3
        for split in ('test', 'test_noisy_white_10db'):
            try:
                ev.append_result(ev.result_row('SVM_full', split, m, 90), path)
            except PermissionError:
                continue
            raise AssertionError(f'split {split} lolos tanpa allow_test')
        ev.append_result(ev.result_row('SVM_full', 'test', m, 90), path, allow_test=True)
        try:
            ev.append_result({**ev.result_row('X', 'validation', m, 1), 'kolom_asing': 1}, path)
            raise AssertionError('kolom di luar kontrak lolos')
        except ValueError:
            pass


def test_test_split_recorded_once():
    m = ev.evaluate(['angry', 'sad'], ['angry', 'angry'], labels=['angry', 'sad'])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'main_table.csv')
        ev.append_result(ev.result_row('SA', 'test', m, 52), path, allow_test=True)
        try:
            ev.append_result(ev.result_row('SA', 'test', m, 52), path, allow_test=True)
            raise AssertionError('hasil test yang sama tercatat dua kali')
        except PermissionError:
            pass
        # Metode lain, split test lain, dan validation ulang tetap boleh
        ev.append_result(ev.result_row('SVM_full', 'test', m, 90), path, allow_test=True)
        ev.append_result(ev.result_row('SA', 'test_noisy_white_10db', m, 52), path, allow_test=True)
        ev.append_result(ev.result_row('SA', 'validation', m, 52), path)
        ev.append_result(ev.result_row('SA', 'validation', m, 52), path)
        assert len(open(path, encoding='utf-8').read().strip().splitlines()) == 6


def test_default_table_path_in_repo():
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    assert ev.DEFAULT_TABLE_PATH == os.path.join(root, 'results', 'main_table.csv')


def test_set_seed():
    ev.set_seed(42)
    a = np.random.rand(3)
    ev.set_seed(42)
    assert (a == np.random.rand(3)).all()


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print(f'OK  {t.__name__}')
    print(f'{len(tests)} uji lolos')
