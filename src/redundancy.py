"""
redundancy.py — Kelompok fitur redundan (berkorelasi tinggi) untuk Proyek 07.

Penanggung jawab : Anggota A
Versi            : 1.0

Dibuat di langkah 11 (analisis korelasi). Dipakai Anggota C untuk analisis stabilitas:
dua subset bisa tampak berbeda per fitur, tetapi sama jika fitur-fiturnya berasal
dari kelompok redundan yang sama (misalnya centroid_mean dan rolloff_mean).

Contoh:
    import redundancy as rd
    cl = rd.load_clusters(BASE_DIR)                     # DataFrame: index, name, cluster, ...
    rd.jaccard([0, 5, 9], [0, 6, 9])                    # Jaccard per fitur
    rd.cluster_jaccard([0, 5, 9], [0, 6, 9], cl)        # Jaccard per kelompok redundan
"""
import os

import numpy as np
import pandas as pd

REDUNDANCY_VERSION = '1.0'


def load_clusters(base_dir, version='none'):
    """Baca data/analysis/corr_clusters_{version}.csv."""
    return pd.read_csv(os.path.join(base_dir, 'data', 'analysis', f'corr_clusters_{version}.csv'))


def to_clusters(indices, clusters):
    """Ubah daftar indeks fitur (0-89) menjadi himpunan nomor kelompok redundan."""
    m = dict(zip(clusters['index'].astype(int), clusters['cluster'].astype(int)))
    return {m[int(i)] for i in indices}


def jaccard(a, b):
    """Indeks Jaccard dua himpunan: |A ∩ B| / |A ∪ B| (1 jika keduanya kosong)."""
    a, b = set(a), set(b)
    return 1.0 if not (a | b) else len(a & b) / len(a | b)


def cluster_jaccard(indices_a, indices_b, clusters):
    """Jaccard dua subset fitur setelah diubah ke kelompok redundan."""
    return jaccard(to_clusters(indices_a, clusters), to_clusters(indices_b, clusters))


def mean_pairwise(subsets, fn):
    """Rata-rata fn(a, b) untuk semua pasangan subset (misalnya 15 run stabilitas)."""
    vals = [fn(subsets[i], subsets[j]) for i in range(len(subsets)) for j in range(i + 1, len(subsets))]
    return float(np.mean(vals)) if vals else float('nan')
