"""
data_splits.py — Pembagian data per aktor untuk Proyek 07 (RAVDESS speech).

Dibuat oleh Anggota A di langkah 7. Dipakai Anggota B (baseline, SA) dan
Anggota C (evaluasi, analisis stabilitas).

Semua pembagian dilakukan PER AKTOR (speaker-independent): satu aktor hanya
berada di satu bagian (train, validation, atau test) dalam satu skema.

Contoh:
    import data_splits as ds
    splits = ds.load_splits(BASE_DIR)                 # split utama
    tr, va, te = ds.split_frames(fn, splits)          # tiga DataFrame
    parts = ds.load_partitions(BASE_DIR)              # 5 partisi
    tr, va, te = ds.split_frames(fn, parts['P2'])     # partisi P2
"""
import os
import json

import numpy as np

SPLITS_VERSION = '1.0'
SPLIT_NAMES = ('train', 'validation', 'test')


def _path(base_dir, name):
    return os.path.join(base_dir, 'data', name)


def load_splits(base_dir):
    """Baca data/splits.json dan kembalikan dict {'train': [...], 'validation': [...], 'test': [...]}."""
    with open(_path(base_dir, 'splits.json'), encoding='utf-8') as f:
        d = json.load(f)
    return {k: list(d['split'][k]) for k in SPLIT_NAMES}


def load_partitions(base_dir):
    """Baca data/partitions.json dan kembalikan dict {'P0': {...}, ..., 'P4': {...}}."""
    with open(_path(base_dir, 'partitions.json'), encoding='utf-8') as f:
        d = json.load(f)
    return {p: {k: list(v[k]) for k in SPLIT_NAMES} for p, v in d['partitions'].items()}


def split_masks(actors, split):
    """Mask boolean (train, validation, test) untuk array/Series nomor aktor."""
    a = np.asarray(actors).astype(int)
    return tuple(np.isin(a, split[k]) for k in SPLIT_NAMES)


def split_frames(df, split, actor_col='actor'):
    """Potong DataFrame menjadi (train, validation, test) berdasarkan kolom aktor."""
    m_tr, m_va, m_te = split_masks(df[actor_col], split)
    return (df[m_tr].reset_index(drop=True),
            df[m_va].reset_index(drop=True),
            df[m_te].reset_index(drop=True))


def check_split(split, all_actors=range(1, 25)):
    """Kembalikan daftar masalah (kosong = aman): tumpang tindih atau aktor hilang."""
    masalah = []
    s = {k: set(split[k]) for k in SPLIT_NAMES}
    for i, a in enumerate(SPLIT_NAMES):
        for b in SPLIT_NAMES[i + 1:]:
            if s[a] & s[b]:
                masalah.append(f'{a} dan {b} tumpang tindih: {sorted(s[a] & s[b])}')
    semua = s['train'] | s['validation'] | s['test']
    hilang = set(all_actors) - semua
    if hilang:
        masalah.append(f'aktor tidak terpakai: {sorted(hilang)}')
    return masalah
