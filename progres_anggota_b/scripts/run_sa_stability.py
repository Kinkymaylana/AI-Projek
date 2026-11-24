r"""
run_sa_stability.py — Menjalankan Tahap 4 (Uji Stabilitas SA: 5 partisi P0-P4 x 3 seed = 15 run).

Aturan isolasi:
- P0 seed 42 langsung menggunakan ulang hasil SA utama (results/sa_main_results.json).
- P1-P4 hanya dipakai untuk analisis stabilitas (subsets.json), TIDAK dipakai untuk
  mengubah model utama / normalisasi / hyperparameter yang sudah dikunci pada P0.
- Untuk setiap partisi Pk, te_p TIDAK pernah disentuh.
"""
import json
import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

REPO_DIR = r'E:\College\UB\S2\AI\Projek akhir\AI-Projek'
OUT_DIR = r'E:\College\UB\S2\AI\Projek akhir\Progres Anggota B'

sys.path.insert(0, os.path.join(REPO_DIR, 'src'))
sys.path.insert(0, os.path.join(OUT_DIR, 'src'))

import data_splits as ds
import redundancy as rd
import sa_feature_selection as sa

SEEDS = [42, 123, 2026]
PARTITIONS = ['P0', 'P1', 'P2', 'P3', 'P4']


def _clean_params(params):
    return {k.replace('clf__', ''): v for k, v in params.items()}


def _run_one(part_name, seed, split_dict, feat_df, feat_names, svm_params, sa_cfg):
    tr, va, _ = ds.split_frames(feat_df, split_dict)
    X_tr = tr[feat_names].values
    y_tr = tr['emotion'].values
    g_tr = tr['actor'].values
    X_va = va[feat_names].values
    y_va = va['emotion'].values

    res = sa.run_simulated_annealing(
        X_tr, y_tr, g_tr,
        svm_params=svm_params,
        seed=seed,
        n_iter=sa_cfg['n_iter'],
        t0=sa_cfg['t0'],
        t_min=sa_cfg['t_min'],
        lambda_penalty=sa_cfg['lambda_penalty'],
        min_features=sa_cfg['min_features'],
        init_prob=sa_cfg['init_prob'],
        stagnation_limit=sa_cfg['stagnation_limit'],
        max_reheats=sa_cfg['max_reheats'],
        n_splits=sa_cfg['n_splits'],
        verbose=False,
    )

    sel_idx = res['best_indices']
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(**_clean_params(svm_params), class_weight='balanced', random_state=42)),
    ])
    pipe.fit(X_tr[:, sel_idx], y_tr)
    y_pred_va = pipe.predict(X_va[:, sel_idx])
    val_f1 = round(float(f1_score(y_va, y_pred_va, average='macro')), 4)

    return {
        'run_id': f'{part_name}_seed{seed}',
        'partition': part_name,
        'seed': seed,
        'is_main_run': False,
        'n_features': len(sel_idx),
        'indices': sel_idx,
        'feature_names': [feat_names[i] for i in sel_idx],
        'best_cv_f1': res['best_cv_f1'],
        'val_macro_f1': val_f1,
        'best_obj': res['best_obj'],
        'reheats_done': res['reheats_done'],
        'cache_hits': res['cache_hits'],
        'elapsed_s': res['elapsed_s'],
    }


def main():
    t0_all = time.time()
    os.makedirs(os.path.join(OUT_DIR, 'results'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'figures'), exist_ok=True)

    with open(os.path.join(OUT_DIR, 'results', 'sa_main_results.json'), encoding='utf-8') as f:
        main_data = json.load(f)

    norm_ver = main_data['normalization']
    sa_cfg = main_data['config']
    svm_params = sa_cfg['svm_params']

    feat_df = pd.read_csv(os.path.join(REPO_DIR, 'data', 'features', f'features_{norm_ver}.csv'))
    fnames_df = pd.read_csv(os.path.join(REPO_DIR, 'data', 'features', 'feature_names.csv'))
    feat_names = fnames_df['name'].tolist()
    parts = ds.load_partitions(REPO_DIR)

    # P0 seed 42 langsung ambil dari hasil SA utama
    main_run_entry = {
        'run_id': 'P0_seed42',
        'partition': 'P0',
        'seed': 42,
        'is_main_run': True,
        'n_features': main_data['k_selected'],
        'indices': main_data['selected_indices'],
        'feature_names': main_data['selected_feature_names'],
        'best_cv_f1': main_data['sa_metrics']['best_cv_f1'],
        'val_macro_f1': main_data['validation_comparison']['svm_sa_subset'],
        'best_obj': main_data['sa_metrics']['best_obj'],
        'reheats_done': main_data['sa_metrics']['reheats_done'],
        'cache_hits': main_data['sa_metrics']['cache_hits'],
        'elapsed_s': main_data['sa_metrics']['elapsed_s'],
    }

    jobs = [(p, s) for p in PARTITIONS for s in SEEDS if not (p == 'P0' and s == 42)]
    print(f'Menjalankan {len(jobs)} run SA tambahan (P0-P4 x 3 seed, total 15 run) secara paralel...')

    extra_runs = Parallel(n_jobs=-1, verbose=5)(
        delayed(_run_one)(p, s, parts[p], feat_df, feat_names, svm_params, sa_cfg)
        for p, s in jobs
    )

    all_runs = [main_run_entry] + extra_runs
    # Urutkan rapi berdasarkan partisi lalu seed
    all_runs.sort(key=lambda r: (PARTITIONS.index(r['partition']), SEEDS.index(r['seed'])))

    # Ringkasan stabilitas awal menggunakan src/redundancy.py dari Anggota A (tanpa menyentuh test)
    cl = rd.load_clusters(REPO_DIR, version=norm_ver)
    all_subsets = [r['indices'] for r in all_runs]
    j_feat_all = round(rd.mean_pairwise(all_subsets, rd.jaccard), 4)
    j_clust_all = round(rd.mean_pairwise(all_subsets, lambda a, b: rd.cluster_jaccard(a, b, cl)), 4)

    # Stabilitas dalam partisi yang sama (antar-seed) vs antar-partisi
    within_feat, within_clust = [], []
    for p in PARTITIONS:
        p_subs = [r['indices'] for r in all_runs if r['partition'] == p]
        within_feat.append(rd.mean_pairwise(p_subs, rd.jaccard))
        within_clust.append(rd.mean_pairwise(p_subs, lambda a, b: rd.cluster_jaccard(a, b, cl)))

    k_vals = [r['n_features'] for r in all_runs]
    cv_vals = [r['best_cv_f1'] for r in all_runs]
    va_vals = [r['val_macro_f1'] for r in all_runs]

    total_elapsed = round(time.time() - t0_all, 2)

    print('\n=== Ringkasan 15 Run Uji Stabilitas (P0-P4 x 3 seed) ===')
    df_runs = pd.DataFrame([
        {k: r[k] for k in ('run_id', 'partition', 'seed', 'n_features', 'best_cv_f1', 'val_macro_f1', 'elapsed_s')}
        for r in all_runs
    ])
    print(df_runs.to_string(index=False))
    print(f'\nUkuran subset (k)          : mean={np.mean(k_vals):.1f} ± {np.std(k_vals):.1f} (min={min(k_vals)}, max={max(k_vals)})')
    print(f'CV Macro F1 (15 run)       : mean={np.mean(cv_vals):.4f} ± {np.std(cv_vals):.4f}')
    print(f'Val Macro F1 (15 run)      : mean={np.mean(va_vals):.4f} ± {np.std(va_vals):.4f}')
    print(f'Jaccard per fitur (15 run) : {j_feat_all:.4f} (dalam partisi: {np.mean(within_feat):.4f})')
    print(f'Jaccard per klaster (15)   : {j_clust_all:.4f} (dalam partisi: {np.mean(within_clust):.4f})')
    print(f'Total waktu 14 run paralel : {total_elapsed:.1f} detik')

    payload = {
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'owner': 'Anggota B',
        'version': sa.SA_VERSION,
        'normalization': norm_ver,
        'partitions': PARTITIONS,
        'seeds': SEEDS,
        'n_runs': len(all_runs),
        'isolation_note': 'P0_seed42 identik dengan SA utama. P1-P4 hanya untuk uji stabilitas, bukan untuk mengubah model utama.',
        'stability_preview': {
            'k_mean': round(float(np.mean(k_vals)), 2),
            'k_std': round(float(np.std(k_vals)), 2),
            'k_min': int(min(k_vals)),
            'k_max': int(max(k_vals)),
            'cv_f1_mean': round(float(np.mean(cv_vals)), 4),
            'cv_f1_std': round(float(np.std(cv_vals)), 4),
            'val_f1_mean': round(float(np.mean(va_vals)), 4),
            'val_f1_std': round(float(np.std(va_vals)), 4),
            'jaccard_feature_all15': j_feat_all,
            'jaccard_cluster_all15': j_clust_all,
            'jaccard_feature_within_partition': round(float(np.mean(within_feat)), 4),
            'jaccard_cluster_within_partition': round(float(np.mean(within_clust)), 4),
        },
        'sa_config': sa_cfg,
        'runs': all_runs,
    }

    with open(os.path.join(OUT_DIR, 'results', 'subsets.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    df_runs.to_csv(os.path.join(OUT_DIR, 'results', 'stability_runs_summary.csv'), index=False)

    # Plot frekuensi pemilihan fitur (Top 25)
    counts = np.zeros(len(feat_names), dtype=int)
    for r in all_runs:
        for idx in r['indices']:
            counts[idx] += 1
    freq_df = fnames_df[['index', 'name', 'group']].copy()
    freq_df['count'] = counts
    freq_df['freq_pct'] = (freq_df['count'] / len(all_runs) * 100).round(1)
    freq_df.sort_values(['count', 'index'], ascending=[False, True]).to_csv(
        os.path.join(OUT_DIR, 'results', 'feature_selection_frequency.csv'), index=False
    )

    top25 = freq_df.sort_values(['count', 'index'], ascending=[False, True]).head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 6.8), dpi=200)
    ax.barh(top25['name'], top25['count'], color='#2b6cb0')
    ax.set_xlabel('Frekuensi Terpilih (dari 15 run: P0–P4 × 3 seed)')
    ax.set_title('25 Fitur Paling Sering Dipilih Simulated Annealing (15 Run Stabilitas)')
    ax.set_xlim(0, len(all_runs) + 1.5)
    for i, (_, row) in enumerate(top25.iterrows()):
        ax.text(row['count'] + 0.2, i, f"{int(row['count'])}/15 ({row['group']})", va='center', fontsize=8)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'figures', 'sa_stability_frequency.png'), dpi=200)
    plt.close(fig)

    print('\nHasil tersimpan di results/subsets.json, results/stability_runs_summary.csv, results/feature_selection_frequency.csv, dan figures/sa_stability_frequency.png')


if __name__ == '__main__':
    main()
