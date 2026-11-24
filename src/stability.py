"""
stability.py - Ukuran stabilitas seleksi fitur untuk Proyek 07.

Penanggung jawab : Anggota C
Versi            : 1.0 (24 November 2026)

Dipakai untuk failure experiment (langkah 9 Rencana Kerja): apakah subset hasil SA
tetap sama jika aktor train diganti (5 partisi) dan seed diganti (3 seed)?

Isi modul:
  selection_matrix()        matriks biner run x fitur (1 = fitur terpilih)
  nogueira()                indeks stabilitas Nogueira et al. (2018); sah untuk ukuran subset berbeda
  random_jaccard()          Jaccard rata-rata subset ACAK dengan ukuran yang sama (pembanding)
  cluster_selection_rate()  per kelompok redundan: seberapa sering kelompok itu terwakili
  transfer_matrix()         subset dari partisi i dilatih dan dinilai di partisi j

Jaccard per fitur dan per kelompok redundan ada di src/redundancy.py (Anggota A).
Kuncheva TIDAK dipakai karena hanya sah untuk subset berukuran sama, sedangkan k hasil SA berbeda-beda.

Referensi:
  Nogueira, S., Sechidis, K., & Brown, G. (2018). On the Stability of Feature Selection
  Algorithms. JMLR, 18(174), 1-54.
"""
import numpy as np

STABILITY_VERSION = '1.0'


def selection_matrix(subsets, n_features):
    """Matriks biner Z (n_run x n_features): Z[i, f] = 1 jika fitur f terpilih di run i."""
    z = np.zeros((len(subsets), int(n_features)), dtype=int)
    for i, s in enumerate(subsets):
        idx = np.asarray(list(s), dtype=int)
        if idx.size and (idx.min() < 0 or idx.max() >= n_features):
            raise ValueError(f'Run {i}: indeks fitur di luar 0..{n_features - 1}')
        z[i, idx] = 1
    return z


def nogueira(z):
    """Indeks stabilitas Nogueira et al. (2018), persamaan (4).

    phi = 1 - mean_f(s_f^2) / ((k_bar/d) * (1 - k_bar/d)),
    dengan s_f^2 = M/(M-1) * p_f * (1 - p_f) adalah varians tak bias seleksi fitur f
    di M run, dan k_bar rata-rata ukuran subset.

    Tafsiran: 1 = semua run memilih subset yang sama; sekitar 0 = setara pemilihan acak
    dengan ukuran yang sama; negatif = lebih tidak konsisten daripada acak.
    """
    z = np.asarray(z, dtype=float)
    m, d = z.shape
    if m < 2:
        raise ValueError('Butuh minimal 2 run')
    k_bar = z.sum(axis=1).mean()
    if k_bar <= 0 or k_bar >= d:
        raise ValueError('Stabilitas tidak terdefinisi jika semua run memilih 0 atau semua fitur')
    p = z.mean(axis=0)
    s2 = m / (m - 1) * p * (1 - p)
    return float(1 - s2.mean() / ((k_bar / d) * (1 - k_bar / d)))


def random_jaccard(sizes, n_features, n_sim=2000, seed=42, mapper=None):
    """Jaccard rata-rata antarpasangan untuk subset ACAK berukuran `sizes` (pembanding).

    Jaccard dipengaruhi ukuran subset: dua subset besar acak pun banyak beririsan.
    Angka ini adalah titik nol yang wajar untuk Jaccard hasil SA.

    mapper : opsional, fungsi set indeks -> set lain sebelum Jaccard dihitung,
             mis. lambda s: redundancy.to_clusters(s, clusters) untuk Jaccard per kelompok.
    """
    rng = np.random.default_rng(seed)
    sizes = [int(k) for k in sizes]
    mapper = mapper or (lambda s: s)
    vals = []
    for _ in range(n_sim):
        a, b = rng.choice(len(sizes), 2, replace=False)
        sa = mapper({int(f) for f in rng.choice(n_features, sizes[a], replace=False)})
        sb = mapper({int(f) for f in rng.choice(n_features, sizes[b], replace=False)})
        vals.append(len(sa & sb) / len(sa | sb))
    return float(np.mean(vals))


def cluster_selection_rate(subsets, clusters):
    """Per kelompok redundan berisi >= 2 fitur: proporsi run yang memilih minimal satu anggotanya,
    dibandingkan dengan rata-rata frekuensi tiap anggota.

    Jika rate_kelompok jauh di atas rata-rata anggota, SA konsisten mewakili kelompok itu
    tetapi bergantian memilih anggotanya.

    clusters : DataFrame dari redundancy.load_clusters() (kolom index, name, cluster).
    Returns  : list of dict, diurutkan dari selisih terbesar.
    """
    sets = [set(int(i) for i in s) for s in subsets]
    out = []
    for c, g in clusters.groupby('cluster'):
        members = [int(i) for i in g['index']]
        if len(members) < 2:
            continue
        rate = np.mean([bool(s & set(members)) for s in sets])
        member_freq = np.mean([[m in s for s in sets] for m in members])
        out.append({'cluster': int(c), 'members': list(g['name']), 'n_members': len(members),
                    'cluster_rate': float(rate), 'mean_member_rate': float(member_freq),
                    'gap': float(rate - member_freq)})
    return sorted(out, key=lambda r: (-r['gap'], r['cluster']))


def transfer_matrix(subsets_by_partition, data_by_partition, make_model, score):
    """Matriks transfer: subset dari partisi i dilatih di train partisi j, dinilai di validation j.

    subsets_by_partition : {Pi: [indeks run 1, indeks run 2, ...]}
    data_by_partition    : {Pj: (X_tr, y_tr, X_va, y_va)}  X berupa array 2D semua fitur
    make_model           : fungsi tanpa argumen yang mengembalikan estimator baru
    score                : fungsi (y_true, y_pred) -> angka, mis. macro F1 dari evaluate.py

    Returns (names, M) dengan M[i, j] = rata-rata skor semua run partisi i di partisi j.
    """
    names = list(subsets_by_partition)
    out = np.zeros((len(names), len(names)))
    for i, pi in enumerate(names):
        for j, pj in enumerate(names):
            x_tr, y_tr, x_va, y_va = data_by_partition[pj]
            vals = []
            for s in subsets_by_partition[pi]:
                idx = sorted(int(f) for f in s)
                model = make_model().fit(x_tr[:, idx], y_tr)
                vals.append(score(y_va, model.predict(x_va[:, idx])))
            out[i, j] = float(np.mean(vals))
    return names, out
