"""
run_comparators_and_curves.py — Menjalankan eksperimen pembanding, sweep lambda,
ensemble, ekspor log per iterasi (logs/), penyimpanan model final (train+val),
dan pembuatan kurva optimasi (figures/optimization/) untuk Proyek 07 (RAVDESS).

Aturan anti-leakage:
  - Seluruh optimasi, tuning, dan pemilihan subset HANYA menggunakan TRAIN (aktor 1-16).
  - Evaluasi pembanding dilakukan pada VALIDATION (aktor 17-20).
  - Model final dilatih ulang pada TRAIN + VALIDATION (aktor 1-20, 1.200 file).
  - Data TEST (aktor 21-24, 240 file) TIDAK pernah diakses oleh skrip ini.
"""
import json
import os
import sys
import time
from datetime import datetime

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_DIR, 'src')
B_SRC_DIR = os.path.join(REPO_DIR, 'progres_anggota_b', 'src')
for p in (SRC_DIR, B_SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import data_splits as ds  # noqa: E402
import sa_feature_selection as sa  # noqa: E402

LAMBDA_VALUES = [0.0, 0.005, 0.015, 0.03, 0.05, 0.08]
SA_SVM_PARAMS = {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}


def _clean_params(params):
    return {k.replace('clf__', ''): v for k, v in params.items()}


def _save_history_csv(history, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cols = ['iter', 'temp', 'cur_cv_f1', 'best_cv_f1', 'cur_obj', 'best_obj', 'cur_k', 'best_k']
    pd.DataFrame(history)[cols].to_csv(out_path, index=False)


def _retune_svm_on_train(X_tr_sub, y_tr, g_tr):
    search = GridSearchCV(
        Pipeline([('scaler', StandardScaler()), ('clf', SVC(class_weight='balanced', random_state=42))]),
        param_grid=[
            {'clf__kernel': ['rbf'], 'clf__C': [0.5, 1.0, 2.0, 5.0, 10.0], 'clf__gamma': ['scale', 'auto', 0.01, 0.005]},
            {'clf__kernel': ['linear'], 'clf__C': [0.05, 0.1, 0.5, 1.0, 5.0]},
        ],
        scoring='f1_macro',
        cv=GroupKFold(n_splits=4),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_tr_sub, y_tr, groups=g_tr)
    params = {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in search.best_params_.items()}
    return search.best_estimator_, _clean_params(params), round(float(search.best_score_), 4)


def _eval_val(est, X_va_sub, y_va):
    y_pred = est.predict(X_va_sub)
    return {
        'val_macro_f1': round(float(f1_score(y_va, y_pred, average='macro')), 4),
        'val_uar': round(float(balanced_accuracy_score(y_va, y_pred)), 4),
        'val_accuracy': round(float(accuracy_score(y_va, y_pred)), 4),
    }


def run_hill_climbing(
    X_tr, y_tr, g_tr,
    svm_params,
    seed=42,
    n_iter=500,
    lambda_penalty=0.015,
    min_features=5,
    init_prob=0.5,
    n_splits=4,
    verbose=True,
):
    """Greedy Hill Climbing (T=0, tidak pernah menerima solusi lebih buruk) dengan setup identik SA."""
    t_start = time.time()
    rng = np.random.default_rng(seed)
    D = X_tr.shape[1]
    folds = sa._precompute_scaled_folds(X_tr, y_tr, g_tr, n_splits=n_splits)

    cur_mask = rng.random(D) < init_prob
    if cur_mask.sum() < min_features:
        cur_mask[rng.choice(D, size=min_features, replace=False)] = True

    cache = {}
    cache_hits = 0

    def eval_cached(m):
        nonlocal cache_hits
        key = m.tobytes()
        if key in cache:
            cache_hits += 1
            return cache[key]
        res = sa.evaluate_subset_cv(m, folds, svm_params, lambda_penalty=lambda_penalty)
        cache[key] = res
        return res

    cur_E, cur_f1 = eval_cached(cur_mask)
    best_mask = cur_mask.copy()
    best_E, best_f1 = cur_E, cur_f1
    accepted = 0

    history = [{
        'iter': 0,
        'temp': 0.0,
        'cur_cv_f1': round(cur_f1, 4),
        'best_cv_f1': round(best_f1, 4),
        'cur_obj': round(-cur_E, 4),
        'best_obj': round(-best_E, 4),
        'cur_k': int(cur_mask.sum()),
        'best_k': int(best_mask.sum()),
    }]

    for it in range(1, n_iter + 1):
        cand_mask = sa._propose_neighbor(cur_mask, rng, high_temp=False, min_features=min_features)
        cand_E, cand_f1 = eval_cached(cand_mask)

        # Greedy: terima hanya jika energi <= saat ini (objektif >= saat ini)
        if cand_E <= cur_E:
            cur_mask, cur_E, cur_f1 = cand_mask, cand_E, cand_f1
            accepted += 1
            if cur_E < best_E:
                best_mask = cur_mask.copy()
                best_E, best_f1 = cur_E, cur_f1

        history.append({
            'iter': it,
            'temp': 0.0,
            'cur_cv_f1': round(cur_f1, 4),
            'best_cv_f1': round(best_f1, 4),
            'cur_obj': round(-cur_E, 4),
            'best_obj': round(-best_E, 4),
            'cur_k': int(cur_mask.sum()),
            'best_k': int(best_mask.sum()),
        })

        if verbose and (it % 100 == 0 or it == n_iter):
            print(f'  [HC iter {it:3d}/{n_iter}] best_obj={-best_E:.4f} | best_cv_f1={best_f1:.4f} | k={int(best_mask.sum()):2d}')

    elapsed = round(time.time() - t_start, 2)
    best_indices = [int(i) for i in np.flatnonzero(best_mask)]
    return {
        'best_indices': best_indices,
        'best_energy': round(float(best_E), 6),
        'best_obj': round(float(-best_E), 6),
        'best_cv_f1': round(float(best_f1), 4),
        'best_n_features': len(best_indices),
        'accepted_moves': accepted,
        'cache_hits': cache_hits,
        'elapsed_s': elapsed,
        'history': history,
    }


def run_random_subsets(X_tr, y_tr, g_tr, X_va, y_va, k=52, n_runs=30, svm_params=None):
    """Evaluasi 30 subset acak berukuran k (sama dengan SA) pada 4-fold GroupKFold train & validation."""
    svm_params = svm_params or SA_SVM_PARAMS
    D = X_tr.shape[1]
    folds = sa._precompute_scaled_folds(X_tr, y_tr, g_tr, n_splits=4)

    rows = []
    for seed in range(n_runs):
        rng = np.random.default_rng(1000 + seed)
        idx = sorted(int(i) for i in rng.choice(D, size=k, replace=False))
        mask = np.zeros(D, dtype=bool)
        mask[idx] = True
        _, cv_f1 = sa.evaluate_subset_cv(mask, folds, svm_params, lambda_penalty=0.0)

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(**svm_params, class_weight='balanced', random_state=42)),
        ])
        pipe.fit(X_tr[:, idx], y_tr)
        val_m = _eval_val(pipe, X_va[:, idx], y_va)
        rows.append({
            'run_idx': seed,
            'seed': 1000 + seed,
            'k': k,
            'cv_macro_f1': round(float(cv_f1), 4),
            'val_macro_f1': val_m['val_macro_f1'],
            'val_uar': val_m['val_uar'],
            'val_accuracy': val_m['val_accuracy'],
            'indices': idx,
        })
    return rows


def _save_final_model(
    name, filename, pipeline, X_tv, y_tv, g_tv,
    selected_indices, selected_feature_names, normalization,
    params, train_val_actors, test_actors, out_dir
):
    X_sub = X_tv[:, selected_indices]
    t0 = time.time()
    cv_scores = cross_val_score(
        pipeline, X_sub, y_tv, groups=g_tv, cv=GroupKFold(n_splits=5), scoring='f1_macro', n_jobs=-1
    )
    pipeline.fit(X_sub, y_tv)
    # Pastikan kompatibilitas dengan inference.py (cek getattr(clf, 'probability', False))
    clf = pipeline.named_steps.get('clf')
    if isinstance(clf, VotingClassifier) and clf.voting == 'soft':
        clf.probability = True

    elapsed_s = round(time.time() - t0, 2)
    artifact = {
        'pipeline': pipeline,
        'model_name': name,
        'normalization': normalization,
        'selected_indices': list(selected_indices),
        'selected_feature_names': list(selected_feature_names),
        'n_features': len(selected_indices),
        'params': dict(params),
        'trained_on': f'P0 train + validation (actors {min(train_val_actors)}-{max(train_val_actors)}, {len(y_tv)} files)',
        'train_val_actors': list(train_val_actors),
        'test_actors_untouched': list(test_actors),
        'n_train_val_samples': int(len(y_tv)),
        'cv5_trainval_macro_f1_mean': round(float(np.mean(cv_scores)), 4),
        'cv5_trainval_macro_f1_std': round(float(np.std(cv_scores)), 4),
        'fit_elapsed_s': elapsed_s,
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    out_path = os.path.join(out_dir, filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(artifact, out_path)
    size_kb = round(os.path.getsize(out_path) / 1024.0, 1)
    print(f'  [Model Final] {name:22s} k={len(selected_indices):2d} | 5-fold F1={artifact["cv5_trainval_macro_f1_mean"]:.4f} | {size_kb:7.1f} KB -> models/{filename}')
    entry = {k: v for k, v in artifact.items() if k != 'pipeline'}
    entry['file'] = f'models/{filename}'
    entry['size_kb'] = size_kb
    return entry


def plot_optimization_curves(logs_dir, sweep_df, rand_df, comp_summary, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)

    # 1. SA vs Hill Climbing Convergence (directly from logs/*/*.csv)
    sa_log = pd.read_csv(os.path.join(logs_dir, 'sa', 'sa_P0_seed42.csv'))
    hc_log = pd.read_csv(os.path.join(logs_dir, 'hill_climbing', 'hc_P0_seed42.csv'))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.4), dpi=200, sharex=True)
    ax1.plot(sa_log['iter'], sa_log['best_obj'], color='#2b6cb0', lw=2.0, label='SA Best Objective J(s)')
    ax1.plot(sa_log['iter'], sa_log['best_cv_f1'], color='#2b6cb0', lw=1.3, ls='--', alpha=0.8, label='SA Best CV Macro-F1')
    ax1.plot(hc_log['iter'], hc_log['best_obj'], color='#c53030', lw=2.0, label='Hill Climbing Best Objective J(s)')
    ax1.plot(hc_log['iter'], hc_log['best_cv_f1'], color='#c53030', lw=1.3, ls='--', alpha=0.8, label='Hill Climbing Best CV Macro-F1')
    ax1.set_ylabel('Score')
    ax1.set_title('Optimization Convergence: Simulated Annealing vs. Hill Climbing (P0, seed=42, lambda=0.015)')
    ax1.grid(linestyle='--', alpha=0.4)
    ax1.legend(loc='lower right', fontsize=8.5)

    ax2.plot(sa_log['iter'], sa_log['best_k'], color='#2b6cb0', lw=2.0, label=f'SA Best k (final={int(sa_log["best_k"].iloc[-1])})')
    ax2.plot(sa_log['iter'], sa_log['cur_k'], color='#90cdf4', lw=0.9, alpha=0.6, label='SA Current k')
    ax2.plot(hc_log['iter'], hc_log['best_k'], color='#c53030', lw=2.0, label=f'Hill Climbing Best k (final={int(hc_log["best_k"].iloc[-1])})')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Active Features (k)')
    ax2.grid(linestyle='--', alpha=0.4)
    ax2.legend(loc='upper right', fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'opt_01_sa_vs_hc_convergence.png'), dpi=200)
    plt.close(fig)

    # 2. Lambda Sweep Trade-off Curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=200)
    ax1_r = ax1.twinx()
    l1 = ax1.plot(sweep_df['lambda'], sweep_df['best_cv_f1'], 'o-', color='#2b6cb0', lw=2.0, label='Train 4-Fold CV Macro-F1')
    l2 = ax1.plot(sweep_df['lambda'], sweep_df['val_macro_f1'], 's--', color='#2f855a', lw=1.8, label='Validation Macro-F1')
    l3 = ax1_r.plot(sweep_df['lambda'], sweep_df['k'], '^:', color='#dd6b20', lw=1.8, label='Selected Features (k)')
    ax1.set_xlabel('Complexity Penalty (lambda)')
    ax1.set_ylabel('Macro-F1')
    ax1_r.set_ylabel('Selected Features (k)', color='#dd6b20')
    ax1.set_title('Effect of Penalty Weight (lambda) on F1 and Subset Size')
    ax1.grid(linestyle='--', alpha=0.4)
    lines = l1 + l2 + l3
    ax1.legend(lines, [l.get_label() for l in lines], loc='lower left', fontsize=8)

    ax2.scatter(sweep_df['k'], sweep_df['val_macro_f1'], s=70, color='#2f855a', zorder=3, label='SA Sweep (Validation)')
    ax2.scatter(sweep_df['k'], sweep_df['best_cv_f1'], s=55, marker='D', color='#2b6cb0', zorder=3, label='SA Sweep (Train CV)')
    for _, row in sweep_df.iterrows():
        ax2.annotate(f"λ={row['lambda']:g}", (row['k'], row['val_macro_f1']),
                     textcoords='offset points', xytext=(4, 4), fontsize=7.5)
    ax2.axhline(comp_summary['svm_all90_val_f1'], color='#718096', ls='--', lw=1.2, label='Baseline SVM 90 (Val)')
    ax2.set_xlabel('Number of Selected Features (k)')
    ax2.set_ylabel('Macro-F1')
    ax2.set_title('Feature Count vs. Macro-F1 Trade-off')
    ax2.grid(linestyle='--', alpha=0.4)
    ax2.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'opt_02_lambda_tradeoff.png'), dpi=200)
    plt.close(fig)

    # 3. Comparator Summary on Validation
    methods = [
        ('SVM Full (90)', comp_summary['svm_all90_val_f1'], '#4a5568'),
        (f"SVM SA ({comp_summary['sa_k']})", comp_summary['svm_sa_val_f1'], '#2b6cb0'),
        (f"SVM Hill Climb ({comp_summary['hc_k']})", comp_summary['svm_hc_val_f1'], '#c53030'),
        (f"SVM Random 30x ({comp_summary['sa_k']})", comp_summary['random30_val_f1_mean'], '#a0aec0'),
        ('Ensemble SVM+RF (90)', comp_summary['ens_all90_val_f1'], '#6b46c1'),
        (f"Ensemble SVM+RF (SA-{comp_summary['sa_k']})", comp_summary['ens_sa_val_f1'], '#2f855a'),
    ]
    labels = [m[0] for m in methods]
    vals = [m[1] for m in methods]
    clrs = [m[2] for m in methods]
    yerrs = [0.0, 0.0, 0.0, comp_summary['random30_val_f1_std'], 0.0, 0.0]

    fig, ax = plt.subplots(figsize=(9.0, 4.6), dpi=200)
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, yerr=yerrs, capsize=4, color=clrs, alpha=0.9, edgecolor='#2d3748', lw=0.6)
    for bar, v in zip(bars, vals):
        ax.annotate(f'{v:.4f}', xy=(bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 5), textcoords='offset points', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=8.5)
    ax.set_ylabel('Validation Macro-F1 (Actors 17-20)')
    ax.set_ylim(0.35, max(vals) + 0.07)
    ax.set_title('Comparator Methods Performance on Speaker-Independent Validation Split (P0)')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'opt_03_comparators_validation.png'), dpi=200)
    plt.close(fig)


def main():
    logs_dir = os.path.join(REPO_DIR, 'logs')
    results_dir = os.path.join(REPO_DIR, 'results')
    models_dir = os.path.join(REPO_DIR, 'models')
    fig_dir = os.path.join(REPO_DIR, 'figures', 'optimization')
    for d in (logs_dir, results_dir, models_dir, fig_dir):
        os.makedirs(d, exist_ok=True)

    # Load baseline & SA main results
    b_res_dir = os.path.join(REPO_DIR, 'progres_anggota_b', 'results')
    with open(os.path.join(b_res_dir, 'baseline_results.json'), encoding='utf-8') as f:
        bl_data = json.load(f)
    with open(os.path.join(b_res_dir, 'sa_main_results.json'), encoding='utf-8') as f:
        sa_main = json.load(f)

    norm_ver = bl_data['selected_normalization']
    feat_df = pd.read_csv(os.path.join(REPO_DIR, 'data', 'features', f'features_{norm_ver}.csv'))
    feat_names = pd.read_csv(os.path.join(REPO_DIR, 'data', 'features', 'feature_names.csv'))['name'].tolist()

    splits = ds.load_splits(REPO_DIR)
    assert not ds.check_split(splits)
    # PENTING: Hanya train (1-16) dan validation (17-20); test (21-24) dibuang (_)
    tr, va, _ = ds.split_frames(feat_df, splits)
    tv = pd.concat([tr, va], ignore_index=True)

    train_val_actors = sorted(int(a) for a in tv['actor'].unique())
    test_actors = sorted(int(a) for a in splits['test'])
    assert set(train_val_actors).isdisjoint(set(test_actors))

    X_tr, y_tr, g_tr = tr[feat_names].values, tr['emotion'].values, tr['actor'].values
    X_va, y_va = va[feat_names].values, va['emotion'].values
    X_tv, y_tv, g_tv = tv[feat_names].values, tv['emotion'].values, tv['actor'].values

    sa_idx = list(sa_main['selected_indices'])
    sa_names = [feat_names[i] for i in sa_idx]
    k_sa = len(sa_idx)

    # 1. Export existing SA main log
    print('=== 1. Ekspor log SA utama ke logs/sa/sa_P0_seed42.csv ===')
    _save_history_csv(sa_main['history'], os.path.join(logs_dir, 'sa', 'sa_P0_seed42.csv'))
    _save_history_csv(sa_main['history'], os.path.join(logs_dir, 'lambda_sweep', 'sa_lambda_0p015.csv'))
    print(f'  Tersimpan {len(sa_main["history"])} baris (iter 0..500).')

    # 2. Hill Climbing
    print('\n=== 2. Pembanding Hill Climbing (P0, seed=42, n_iter=500, lambda=0.015) ===')
    hc_res = run_hill_climbing(
        X_tr, y_tr, g_tr,
        svm_params=SA_SVM_PARAMS,
        seed=42, n_iter=500, lambda_penalty=0.015, min_features=5, init_prob=0.5, n_splits=4, verbose=True
    )
    _save_history_csv(hc_res['history'], os.path.join(logs_dir, 'hill_climbing', 'hc_P0_seed42.csv'))
    hc_idx = hc_res['best_indices']
    hc_names = [feat_names[i] for i in hc_idx]
    hc_est, hc_tuned_params, hc_tuned_cv = _retune_svm_on_train(X_tr[:, hc_idx], y_tr, g_tr)
    hc_val = _eval_val(hc_est, X_va[:, hc_idx], y_va)
    print(f'  HC selesai: k={len(hc_idx)}, CV F1={hc_res["best_cv_f1"]:.4f} (tuned={hc_tuned_cv:.4f}), Val F1={hc_val["val_macro_f1"]:.4f}')

    # 3. 30 Random Subsets (k = 52)
    print(f'\n=== 3. Pembanding 30 Random Subset (k={k_sa}) ===')
    t0 = time.time()
    rand_rows = run_random_subsets(X_tr, y_tr, g_tr, X_va, y_va, k=k_sa, n_runs=30, svm_params=SA_SVM_PARAMS)
    rand_df = pd.DataFrame(rand_rows)
    rand_df_csv = rand_df.drop(columns=['indices'])
    os.makedirs(os.path.join(results_dir, 'comparators'), exist_ok=True)
    rand_df_csv.to_csv(os.path.join(results_dir, 'comparators', 'random_subset_30.csv'), index=False)

    # Pilih median berdasarkan CV F1 di train (tanpa melihat validation!)
    sorted_by_cv = rand_df.sort_values('cv_macro_f1').reset_index(drop=True)
    med_row = sorted_by_cv.iloc[len(sorted_by_cv) // 2]
    med_idx = list(med_row['indices'])
    med_names = [feat_names[i] for i in med_idx]
    print(
        f'  30 Random selesai ({time.time() - t0:.1f}s) | '
        f'CV F1={rand_df["cv_macro_f1"].mean():.4f} +/- {rand_df["cv_macro_f1"].std(ddof=1):.4f} | '
        f'Val F1={rand_df["val_macro_f1"].mean():.4f} +/- {rand_df["val_macro_f1"].std(ddof=1):.4f}'
    )

    # 4. Sweep Lambda
    print(f'\n=== 4. Sweep Lambda ({LAMBDA_VALUES}) ===')
    sweep_rows = []
    for lam in LAMBDA_VALUES:
        lam_tag = str(lam).replace('.', 'p')
        if abs(lam - 0.015) < 1e-9:
            sweep_rows.append({
                'lambda': lam,
                'k': k_sa,
                'reduction_pct': round((1.0 - k_sa / len(feat_names)) * 100.0, 2),
                'best_cv_f1': sa_main['sa_metrics']['best_cv_f1'],
                'best_obj': sa_main['sa_metrics']['best_obj'],
                'val_macro_f1': sa_main['validation_comparison']['svm_sa_subset'],
                'val_uar': sa_main['validation_comparison']['svm_sa_uar'],
                'val_accuracy': sa_main['validation_comparison']['svm_sa_accuracy'],
                'log_file': 'logs/lambda_sweep/sa_lambda_0p015.csv',
                'elapsed_s': sa_main['sa_metrics']['elapsed_s'],
            })
            print(f'  [lambda={lam:5.3f}] (reused P0_seed42) k={k_sa:2d} | CV F1={sa_main["sa_metrics"]["best_cv_f1"]:.4f} | Val F1={sa_main["validation_comparison"]["svm_sa_subset"]:.4f}')
            continue

        res = sa.run_simulated_annealing(
            X_tr, y_tr, g_tr,
            svm_params=SA_SVM_PARAMS,
            seed=42, n_iter=500, t0=0.015, t_min=0.0002,
            lambda_penalty=lam, min_features=5, init_prob=0.5,
            stagnation_limit=80, max_reheats=2, n_splits=4, verbose=False,
        )
        log_rel = f'logs/lambda_sweep/sa_lambda_{lam_tag}.csv'
        _save_history_csv(res['history'], os.path.join(REPO_DIR, log_rel))
        idx = res['best_indices']
        est, _, _ = _retune_svm_on_train(X_tr[:, idx], y_tr, g_tr)
        val_m = _eval_val(est, X_va[:, idx], y_va)
        sweep_rows.append({
            'lambda': lam,
            'k': len(idx),
            'reduction_pct': round((1.0 - len(idx) / len(feat_names)) * 100.0, 2),
            'best_cv_f1': res['best_cv_f1'],
            'best_obj': res['best_obj'],
            'val_macro_f1': val_m['val_macro_f1'],
            'val_uar': val_m['val_uar'],
            'val_accuracy': val_m['val_accuracy'],
            'log_file': log_rel,
            'elapsed_s': res['elapsed_s'],
        })
        print(f'  [lambda={lam:5.3f}] k={len(idx):2d} | CV F1={res["best_cv_f1"]:.4f} | Val F1={val_m["val_macro_f1"]:.4f} | {res["elapsed_s"]:.1f}s -> {log_rel}')

    sweep_df = pd.DataFrame(sweep_rows)
    os.makedirs(os.path.join(results_dir, 'lambda_sweep'), exist_ok=True)
    sweep_df.to_csv(os.path.join(results_dir, 'lambda_sweep', 'lambda_sweep.csv'), index=False)

    # 5. Ensemble (Soft-Voting SVM + RF)
    print('\n=== 5. Pembanding Ensemble (Soft-Voting SVM + RF) ===')
    svm_all90_params = _clean_params(bl_data['results'][norm_ver]['SVM']['best_params'])
    svm_sa52_params = _clean_params(sa_main['validation_comparison']['svm_sa_tuned_params'])
    rf_params = _clean_params(bl_data['results'][norm_ver]['RF']['best_params'])

    def build_ensemble(svm_p, rf_p):
        svm_clf = SVC(**svm_p, class_weight='balanced', probability=True, random_state=42)
        rf_clf = RandomForestClassifier(**rf_p, class_weight='balanced', random_state=42, n_jobs=-1)
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', VotingClassifier(estimators=[('svm', svm_clf), ('rf', rf_clf)], voting='soft', weights=[2, 1])),
        ])

    ens_all90 = build_ensemble(svm_all90_params, rf_params)
    ens_all90_cv = cross_val_score(ens_all90, X_tr, y_tr, groups=g_tr, cv=GroupKFold(n_splits=4), scoring='f1_macro', n_jobs=-1)
    ens_all90.fit(X_tr, y_tr)
    ens_all90_val = _eval_val(ens_all90, X_va, y_va)
    print(f'  Ensemble SVM+RF (90 fitur) : CV F1={np.mean(ens_all90_cv):.4f} | Val F1={ens_all90_val["val_macro_f1"]:.4f} | Val UAR={ens_all90_val["val_uar"]:.4f}')

    ens_sa52 = build_ensemble(svm_sa52_params, rf_params)
    ens_sa52_cv = cross_val_score(ens_sa52, X_tr[:, sa_idx], y_tr, groups=g_tr, cv=GroupKFold(n_splits=4), scoring='f1_macro', n_jobs=-1)
    ens_sa52.fit(X_tr[:, sa_idx], y_tr)
    ens_sa52_val = _eval_val(ens_sa52, X_va[:, sa_idx], y_va)
    print(f'  Ensemble SVM+RF (SA 52)    : CV F1={np.mean(ens_sa52_cv):.4f} | Val F1={ens_sa52_val["val_macro_f1"]:.4f} | Val UAR={ens_sa52_val["val_uar"]:.4f}')

    # 6. Latih & simpan model final (train + validation, aktor 1-20) untuk pembanding baru
    print('\n=== 6. Melatih Model Final Pembanding (train + validation, aktor 1-20) ===')
    manifest_path = os.path.join(models_dir, 'final_models_manifest.json')
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding='utf-8') as f:
            manifest = json.load(f)
    else:
        manifest = {'models': {}}

    new_models = [
        (
            'SVM_HC',
            'hill_climbing/svm_hc_final.joblib',
            Pipeline([('scaler', StandardScaler()), ('clf', SVC(**hc_tuned_params, class_weight='balanced', probability=True, random_state=42))]),
            hc_idx, hc_names,
            {**hc_tuned_params, 'class_weight': 'balanced', 'probability': True, 'random_state': 42},
        ),
        (
            'SVM_Random52_Median',
            'random_subset/svm_random52_median_final.joblib',
            Pipeline([('scaler', StandardScaler()), ('clf', SVC(**SA_SVM_PARAMS, class_weight='balanced', probability=True, random_state=42))]),
            med_idx, med_names,
            {**SA_SVM_PARAMS, 'class_weight': 'balanced', 'probability': True, 'random_state': 42, 'median_seed': int(med_row['seed'])},
        ),
        (
            'Ensemble_SVM_RF_all90',
            'ensemble/ensemble_svm_rf_all90_final.joblib',
            build_ensemble(svm_all90_params, rf_params),
            list(range(len(feat_names))), feat_names,
            {'voting': 'soft', 'weights': [2, 1], 'svm': svm_all90_params, 'rf': rf_params},
        ),
        (
            'Ensemble_SVM_RF_SA52',
            'ensemble/ensemble_svm_rf_sa52_final.joblib',
            build_ensemble(svm_sa52_params, rf_params),
            sa_idx, sa_names,
            {'voting': 'soft', 'weights': [2, 1], 'svm': svm_sa52_params, 'rf': rf_params},
        ),
    ]
    for name, fname, pipe, idx, fnames, params in new_models:
        entry = _save_final_model(
            name, fname, pipe, X_tv, y_tv, g_tv,
            idx, fnames, norm_ver, params, train_val_actors, test_actors, models_dir
        )
        manifest['models'][name] = entry

    manifest['updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # 7. Simpan ringkasan lengkap & buat Kurva Optimasi
    comp_summary = {
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'normalization': norm_ver,
        'split_rule': 'train=actors 1-16, validation=actors 17-20; test (21-24) untouched',
        'svm_all90_val_f1': bl_data['results'][norm_ver]['SVM']['val_macro_f1'],
        'sa_k': k_sa,
        'svm_sa_val_f1': sa_main['validation_comparison']['svm_sa_subset'],
        'hc_k': len(hc_idx),
        'hc_best_cv_f1': hc_res['best_cv_f1'],
        'hc_tuned_params': hc_tuned_params,
        'svm_hc_val_f1': hc_val['val_macro_f1'],
        'svm_hc_val_uar': hc_val['val_uar'],
        'hc_selected_indices': hc_idx,
        'hc_selected_feature_names': hc_names,
        'random30_k': k_sa,
        'random30_cv_f1_mean': round(float(rand_df['cv_macro_f1'].mean()), 4),
        'random30_cv_f1_std': round(float(rand_df['cv_macro_f1'].std(ddof=1)), 4),
        'random30_val_f1_mean': round(float(rand_df['val_macro_f1'].mean()), 4),
        'random30_val_f1_std': round(float(rand_df['val_macro_f1'].std(ddof=1)), 4),
        'random30_median_seed': int(med_row['seed']),
        'random30_median_val_f1': float(med_row['val_macro_f1']),
        'ens_all90_cv_f1': round(float(np.mean(ens_all90_cv)), 4),
        'ens_all90_val_f1': ens_all90_val['val_macro_f1'],
        'ens_all90_val_uar': ens_all90_val['val_uar'],
        'ens_sa_cv_f1': round(float(np.mean(ens_sa52_cv)), 4),
        'ens_sa_val_f1': ens_sa52_val['val_macro_f1'],
        'ens_sa_val_uar': ens_sa52_val['val_uar'],
    }
    with open(os.path.join(results_dir, 'comparators', 'comparators_results.json'), 'w', encoding='utf-8') as f:
        json.dump(comp_summary, f, indent=2, ensure_ascii=False)

    print('\n=== 7. Membuat Kurva Optimasi di figures/optimization/ ===')
    plot_optimization_curves(logs_dir, sweep_df, rand_df, comp_summary, fig_dir)
    print('  Tersimpan: opt_01_sa_vs_hc_convergence.png, opt_02_lambda_tradeoff.png, opt_03_comparators_validation.png')
    print('\nSEMUA SELESAI!')


if __name__ == '__main__':
    main()
