"""
run_stability_analysis.py - Analisis stabilitas seleksi fitur SA (langkah 9 Rencana Kerja).

Penanggung jawab : Anggota C
Versi            : 1.0 (24 November 2026)

Input  : progres_anggota_b/results/subsets.json (15 run: P0-P4 x seed 42, 123, 2026),
         data/analysis/corr_clusters_<versi>.csv dan data/partitions.json (Anggota A)
Output : results/stability_summary.json
         results/stability_cluster_rate.csv
         figures/stability/stab_01_heatmap_seleksi.png
         figures/stability/stab_02_ringkasan_stabilitas.png
         (dengan --transfer) results/stability_transfer.csv, figures/stability/stab_03_transfer.png

Teks gambar berbahasa Inggris (untuk paper IEEE); output layar berbahasa Indonesia.

Analisis utama hanya memakai daftar fitur terpilih; tidak ada model yang dilatih dan
tidak ada label yang dibaca. Matriks transfer (--transfer) melatih model di partisi P1-P4,
yang memakai aktor 21-24; sesuai aturan partitions.json, ia hanya dijalankan setelah
hasil test tercatat di results/main_table.csv.

Jalankan dari root repo:
    .venv\\Scripts\\python.exe src\\run_stability_analysis.py
"""
import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import evaluate as ev  # noqa: E402
import redundancy as rd  # noqa: E402
import stability as st  # noqa: E402

SUBSETS_PATH = os.path.join(ROOT, 'progres_anggota_b', 'results', 'subsets.json')
RESULTS_DIR = os.path.join(ROOT, 'results')
FIG_DIR = os.path.join(ROOT, 'figures', 'stability')

# Palet referensi (light): seri 1 biru, tinta, garis, permukaan
BLUE, BLUE_DARK, EMPTY = '#2a78d6', '#184f95', '#ebeae5'
INK, INK_2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'


def koma(v, d=3):
    """Angka untuk output layar (bahasa Indonesia, koma desimal)."""
    return f'{v:.{d}f}'.replace('.', ',')


def num(v, d=3):
    """Angka untuk gambar paper (bahasa Inggris, titik desimal)."""
    return f'{v:.{d}f}'


def load():
    with open(SUBSETS_PATH, encoding='utf-8') as f:
        sub = json.load(f)
    names = pd.read_csv(os.path.join(ROOT, 'data', 'features', 'feature_names.csv'))
    clusters = rd.load_clusters(ROOT, version=sub['normalization'])
    with open(os.path.join(ROOT, 'data', 'partitions.json'), encoding='utf-8') as f:
        parts = json.load(f)['partitions']
    return sub, names, clusters, parts


def pairwise(runs, fn, same_partition):
    vals = [fn(a['indices'], b['indices']) for a, b in itertools.combinations(runs, 2)
            if (a['partition'] == b['partition']) == same_partition]
    return float(np.mean(vals))


def analyse(sub, names, clusters, parts):
    runs = sub['runs']
    d = len(names)
    subsets = [r['indices'] for r in runs]
    sizes = [len(s) for s in subsets]
    z = st.selection_matrix(subsets, d)
    cj = lambda a, b: rd.cluster_jaccard(a, b, clusters)  # noqa: E731
    to_cl = lambda s: rd.to_clusters(s, clusters)  # noqa: E731

    by_part = {}
    for r in runs:
        by_part.setdefault(r['partition'], []).append(r['indices'])
    nog_within = {p: st.nogueira(st.selection_matrix(s, d)) for p, s in by_part.items()}

    train_sets = [p['train'] for p in parts.values()]
    out = {
        'n_runs': len(runs), 'n_features': d, 'partitions': sorted(by_part),
        'k_mean': float(np.mean(sizes)), 'k_std': float(np.std(sizes, ddof=1)),
        'k_min': int(min(sizes)), 'k_max': int(max(sizes)),
        'nogueira_all': st.nogueira(z),
        'nogueira_within_partition_mean': float(np.mean(list(nog_within.values()))),
        'nogueira_within_partition': nog_within,
        'jaccard_feature_all': rd.mean_pairwise(subsets, rd.jaccard),
        'jaccard_feature_same_partition': pairwise(runs, rd.jaccard, True),
        'jaccard_feature_diff_partition': pairwise(runs, rd.jaccard, False),
        'jaccard_feature_random': st.random_jaccard(sizes, d),
        'jaccard_cluster_all': rd.mean_pairwise(subsets, cj),
        'jaccard_cluster_same_partition': pairwise(runs, cj, True),
        'jaccard_cluster_diff_partition': pairwise(runs, cj, False),
        'jaccard_cluster_random': st.random_jaccard(sizes, d, mapper=to_cl),
        'actor_overlap_train_jaccard': rd.mean_pairwise(train_sets, rd.jaccard),
        'always_selected': [names['name'][i] for i in np.where(z.all(axis=0))[0]],
        'never_selected': [names['name'][i] for i in np.where(~z.any(axis=0))[0]],
        'source': os.path.relpath(SUBSETS_PATH, ROOT).replace('\\', '/'),
        'stability_version': st.STABILITY_VERSION,
    }
    return out, z, st.cluster_selection_rate(subsets, clusters)


def fig_heatmap(z, runs, names, path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    freq = z.sum(axis=0)
    order = sorted(range(z.shape[1]), key=lambda f: (-freq[f], names['name'][f]))
    zs = z[:, order]
    fig, ax = plt.subplots(figsize=(15, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.imshow(zs, aspect='auto', cmap=ListedColormap([EMPTY, BLUE]), vmin=0, vmax=1, interpolation='none')
    # garis pemisah 2px warna permukaan antar-sel
    ax.set_xticks(np.arange(-.5, zs.shape[1]), minor=True)
    ax.set_yticks(np.arange(-.5, zs.shape[0]), minor=True)
    ax.grid(which='minor', color=SURFACE, linewidth=1.2)
    ax.tick_params(which='minor', length=0)
    ax.set_xticks(range(zs.shape[1]))
    ax.set_xticklabels([names['name'][f] for f in order], rotation=90, fontsize=6, color=INK_2)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([f"{r['run_id']}  (k={r['n_features']})" for r in runs], fontsize=8, color=INK_2)
    for lab, r in zip(ax.get_yticklabels(), runs):
        if r.get('is_main_run'):
            lab.set_fontweight('bold')
            lab.set_color(INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(0.01, 0.975, f'Features selected across {len(runs)} SA runs (5 speaker partitions x 3 seeds), '
             'sorted by selection frequency', fontsize=11, color=INK, va='top')
    fig.text(0.01, 0.93, 'Blue = selected. Bold row = main model run (P0, seed 42).',
             fontsize=8.5, color=INK_2, va='top')
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_summary(s, path):
    import matplotlib.pyplot as plt

    rows = [
        (f"Nogueira index (all {s['n_runs']} runs)", s['nogueira_all'], 0.0),
        ('Jaccard, redundant-feature clusters', s['jaccard_cluster_all'], s['jaccard_cluster_random']),
        ('Jaccard, individual features', s['jaccard_feature_all'], s['jaccard_feature_random']),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 3.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], height=0.42, color=BLUE, zorder=3)
    for yi, (lab, v, ref) in zip(y, rows):
        ax.text(v + 0.012, yi, num(v), va='center', fontsize=9, color=INK, zorder=4)
        ax.plot([ref, ref], [yi - 0.32, yi + 0.32], color=INK_2, linewidth=2, zorder=4,
                solid_capstyle='round')
        ax.text(ref, yi + 0.36, f'random {num(ref)}', ha='center', va='bottom', fontsize=7.5, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9, color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: num(v, 1)))
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(axis='x', colors=MUTED, labelsize=8)
    ax.tick_params(axis='y', length=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(AXIS)
    ax.set_title('Stability of SA subsets vs. random subsets of the same size', loc='left',
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_transfer(names, m, path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list('biru', ['#cde2fb', '#6da7ec', '#2a78d6', '#184f95', '#0d366b'])
    fig, ax = plt.subplots(figsize=(5.6, 4.6), facecolor=SURFACE)
    im = ax.imshow(m, cmap=cmap)
    lo, hi = m.min(), m.max()
    for i in range(len(names)):
        for j in range(len(names)):
            light = (m[i, j] - lo) / ((hi - lo) or 1) > 0.55
            ax.text(j, i, num(m[i, j]), ha='center', va='center', fontsize=9,
                    color='#ffffff' if light else INK)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    ax.set_xlabel('Evaluated on partition (validation)')
    ax.set_ylabel('Subset selected on partition')
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Macro F1 (mean over 3 seeds)')
    ax.set_title('Transfer matrix of SA subsets', loc='left', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def test_recorded():
    path = ev.DEFAULT_TABLE_PATH
    if not os.path.exists(path):
        return False
    with open(path, encoding='utf-8', newline='') as f:
        return any(r['split'].startswith('test') for r in csv.DictReader(f))


def run_transfer(sub, parts, names):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    import data_splits as ds

    df = pd.read_csv(os.path.join(ROOT, 'data', 'features', f"features_{sub['normalization']}.csv"))
    cols = list(names['name'])
    data = {}
    for p, split in parts.items():
        tr, va, _ = ds.split_frames(df, split)
        data[p] = (tr[cols].values, tr['emotion'].values, va[cols].values, va['emotion'].values)
    by_part = {}
    for r in sub['runs']:
        by_part.setdefault(r['partition'], []).append(r['indices'])
    params = sub['sa_config']['svm_params']

    def make():
        return Pipeline([('scaler', StandardScaler()),
                         ('clf', SVC(class_weight='balanced', random_state=42, **params))])

    return st.transfer_matrix(by_part, data, make, lambda yt, yp: ev.evaluate(yt, yp)['macro_f1'])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--transfer', action='store_true',
                    help='Hitung matriks transfer 5x5 (hanya setelah hasil test tercatat)')
    args = ap.parse_args()
    ev.set_seed(42)

    sub, names, clusters, parts = load()
    summary, z, crate = analyse(sub, names, clusters, parts)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    pd.DataFrame([{**r, 'members': ' | '.join(r['members'])} for r in crate]).to_csv(
        os.path.join(RESULTS_DIR, 'stability_cluster_rate.csv'), index=False)
    fig_heatmap(z, sub['runs'], names, os.path.join(FIG_DIR, 'stab_01_heatmap_seleksi.png'))
    fig_summary(summary, os.path.join(FIG_DIR, 'stab_02_ringkasan_stabilitas.png'))

    if args.transfer:
        if not test_recorded():
            sys.exit('Ditolak: matriks transfer memakai aktor 21-24 (P1-P4). Jalankan setelah hasil test '
                     'tercatat di results/main_table.csv (aturan partitions.json).')
        tnames, m = run_transfer(sub, parts, names)
        pd.DataFrame(m, index=tnames, columns=tnames).to_csv(os.path.join(RESULTS_DIR, 'stability_transfer.csv'))
        fig_transfer(tnames, m, os.path.join(FIG_DIR, 'stab_03_transfer.png'))
        summary['transfer_diagonal_mean'] = float(np.mean(np.diag(m)))
        summary['transfer_offdiagonal_mean'] = float(m[~np.eye(len(m), dtype=bool)].mean())

    with open(os.path.join(RESULTS_DIR, 'stability_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    s = summary
    print(f"k = {koma(s['k_mean'], 2)} ± {koma(s['k_std'], 2)} (rentang {s['k_min']}-{s['k_max']})")
    print(f"Nogueira semua run      : {koma(s['nogueira_all'])}  (dalam partisi rata-rata "
          f"{koma(s['nogueira_within_partition_mean'])})")
    for kind, label in (('feature', 'Jaccard per fitur   '), ('cluster', 'Jaccard per kelompok')):
        print(f"{label}    : semua {koma(s[f'jaccard_{kind}_all'])} | partisi sama "
              f"{koma(s[f'jaccard_{kind}_same_partition'])} | partisi beda "
              f"{koma(s[f'jaccard_{kind}_diff_partition'])} | acak {koma(s[f'jaccard_{kind}_random'])}")
    print(f"Tumpang-tindih aktor train antarpartisi: {koma(s['actor_overlap_train_jaccard'])}")
    print('Selalu terpilih :', ', '.join(s['always_selected']) or '-')
    print('Tidak pernah    :', ', '.join(s['never_selected']) or '-')
    if 'transfer_diagonal_mean' in s:
        print(f"Transfer: diagonal {koma(s['transfer_diagonal_mean'])} | luar diagonal "
              f"{koma(s['transfer_offdiagonal_mean'])}")


if __name__ == '__main__':
    main()
