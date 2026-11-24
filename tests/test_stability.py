"""
Uji src/stability.py dengan data buatan (tanpa data proyek).

Jalankan dari root repo:
    python tests/test_stability.py      atau      python -m pytest tests/
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import stability as st  # noqa: E402


def test_selection_matrix():
    z = st.selection_matrix([[0, 2], [1]], 3)
    assert z.tolist() == [[1, 0, 1], [0, 1, 0]]
    try:
        st.selection_matrix([[3]], 3)
        raise AssertionError('indeks di luar rentang lolos')
    except ValueError:
        pass


def test_nogueira_hand_computed():
    # Subset identik -> 1
    assert np.isclose(st.nogueira(st.selection_matrix([[0, 1], [0, 1], [0, 1]], 5)), 1.0)
    # d=2, dua run saling lepas: p=[.5,.5], s2=2*.25=.5, k_bar=1, penyebut=.25 -> 1-.5/.25 = -1
    assert np.isclose(st.nogueira([[1, 0], [0, 1]]), -1.0)


def test_nogueira_random_near_zero_and_handles_unequal_sizes():
    rng = np.random.default_rng(0)
    subsets = [rng.choice(90, k, replace=False) for k in rng.integers(30, 60, 400)]
    assert abs(st.nogueira(st.selection_matrix(subsets, 90))) < 0.02


def test_nogueira_undefined():
    for z in ([[0, 0], [0, 0]], [[1, 1], [1, 1]], [[1, 0]]):
        try:
            st.nogueira(z)
            raise AssertionError(f'{z} lolos')
        except ValueError:
            pass


def test_random_jaccard():
    # Dua subset acak berukuran 45 dari 90: E|A∩B| = 22,5 -> Jaccard ~ 22,5/67,5 = 1/3
    assert abs(st.random_jaccard([45, 45], 90, n_sim=3000) - 1 / 3) < 0.01
    assert st.random_jaccard([45, 45], 90, seed=1) == st.random_jaccard([45, 45], 90, seed=1)


def test_cluster_selection_rate():
    cl = pd.DataFrame({'index': [0, 1, 2], 'name': ['a', 'b', 'c'], 'cluster': [7, 7, 8]})
    r = st.cluster_selection_rate([[0], [1], [0, 2]], cl)
    assert len(r) == 1 and r[0]['cluster'] == 7
    assert np.isclose(r[0]['cluster_rate'], 1.0) and np.isclose(r[0]['mean_member_rate'], 0.5)


def test_transfer_matrix():
    class Majority:
        def fit(self, x, y):
            self.v = np.bincount(y).argmax()
            return self

        def predict(self, x):
            return np.full(len(x), self.v)

    x = np.zeros((4, 3))
    data = {'P0': (x, np.array([1, 1, 1, 0]), x, np.array([1, 1, 0, 0])),
            'P1': (x, np.array([0, 0, 0, 1]), x, np.array([0, 0, 0, 0]))}
    names, m = st.transfer_matrix({'P0': [[0], [1, 2]], 'P1': [[2]]}, data, Majority,
                                  lambda yt, yp: float(np.mean(yt == yp)))
    assert names == ['P0', 'P1'] and m.tolist() == [[0.5, 1.0], [0.5, 1.0]]


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print(f'OK  {t.__name__}')
    print(f'{len(tests)} uji lolos')
