r"""
run_sa_main.py — Menjalankan Tahap 3 (Simulated Annealing Utama pada Split Utama / P0).

Membaca data dari repo AI-Projek (read-only) dan baseline_results.json dari
Progres Anggota B, lalu menyimpan hasil SA utama ke Progres Anggota B.
"""
import json
import os
import sys
from datetime import datetime

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

REPO_DIR = r'E:\College\UB\S2\AI\Projek akhir\AI-Projek'
OUT_DIR = r'E:\College\UB\S2\AI\Projek akhir\Progres Anggota B'

sys.path.insert(0, os.path.join(REPO_DIR, 'src'))
sys.path.insert(0, os.path.join(OUT_DIR, 'src'))

import data_splits as ds
import sa_feature_selection as sa

EMOTION_ORDER = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']


def _clean_params(params):
    return {k.replace('clf__', ''): v for k, v in params.items()}


def main():
    os.makedirs(os.path.join(OUT_DIR, 'results'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'figures'), exist_ok=True)

    with open(os.path.join(OUT_DIR, 'results', 'baseline_results.json'), encoding='utf-8') as f:
        bl_data = json.load(f)

    norm_ver = bl_data['selected_normalization']
    svm_params = bl_data['results'][norm_ver]['SVM']['best_params']
    rf_params = bl_data['results'][norm_ver]['RF']['best_params']
    bl_svm_val_f1 = bl_data['results'][norm_ver]['SVM']['val_macro_f1']
    bl_rf_val_f1 = bl_data['results'][norm_ver]['RF']['val_macro_f1']

    print(f'Menggunakan versi normalisasi terpilih dari baseline: {norm_ver}')
    print(f'Parameter SVM baseline: {svm_params}')

    feat_df = pd.read_csv(os.path.join(REPO_DIR, 'data', 'features', f'features_{norm_ver}.csv'))
    fnames_df = pd.read_csv(os.path.join(REPO_DIR, 'data', 'features', 'feature_names.csv'))
    feat_names = fnames_df['name'].tolist()

    splits = ds.load_splits(REPO_DIR)
    assert not ds.check_split(splits)
    # PENTING: Hanya train (1-16) dan validation (17-20). Test (21-24) TIDAK dipakai.
    tr, va, _ = ds.split_frames(feat_df, splits)
    assert set(tr['actor']).isdisjoint({21, 22, 23, 24})
    assert set(va['actor']).isdisjoint({21, 22, 23, 24})

    X_tr = tr[feat_names].values
    y_tr = tr['emotion'].values
    g_tr = tr['actor'].values
    X_va = va[feat_names].values
    y_va = va['emotion'].values

    # Gunakan gamma='scale' di dalam SA agar skala kernel RBF otomatis beradaptasi dengan jumlah fitur k
    sa_svm_params = {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}
    print(f'\nMenjalankan Simulated Annealing utama (P0, seed=42, n_iter=500, svm={sa_svm_params})...')
    sa_res = sa.run_simulated_annealing(
        X_tr, y_tr, g_tr,
        svm_params=sa_svm_params,
        seed=42,
        n_iter=500,
        t0=0.015,
        t_min=0.0002,
        lambda_penalty=0.015,
        min_features=5,
        init_prob=0.5,
        stagnation_limit=80,
        max_reheats=2,
        n_splits=4,
        verbose=True,
    )

    sel_idx = sa_res['best_indices']
    sel_names = [feat_names[i] for i in sel_idx]
    k = len(sel_idx)
    reduction_pct = round((1.0 - k / len(feat_names)) * 100.0, 2)

    # Re-tune SVM pada subset S* dengan GroupKFold(n_splits=4) di train saja (tanpa menyentuh validation/test)
    from sklearn.model_selection import GridSearchCV, GroupKFold
    svm_search = GridSearchCV(
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
    svm_search.fit(X_tr[:, sel_idx], y_tr, groups=g_tr)
    svm_pipe = svm_search.best_estimator_
    tuned_svm_params = {k_: (v if not isinstance(v, np.generic) else v.item()) for k_, v in svm_search.best_params_.items()}
    y_pred_svm = svm_pipe.predict(X_va[:, sel_idx])
    val_f1_svm = round(float(f1_score(y_va, y_pred_svm, average='macro')), 4)
    val_uar_svm = round(float(balanced_accuracy_score(y_va, y_pred_svm)), 4)
    val_acc_svm = round(float(accuracy_score(y_va, y_pred_svm)), 4)
    rep_svm = classification_report(y_va, y_pred_svm, labels=EMOTION_ORDER, output_dict=True, zero_division=0)
    per_class_svm = {emo: round(float(rep_svm[emo]['f1-score']), 4) for emo in EMOTION_ORDER}

    rf_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(**_clean_params(rf_params), class_weight='balanced', random_state=42, n_jobs=-1)),
    ])
    rf_pipe.fit(X_tr[:, sel_idx], y_tr)
    y_pred_rf = rf_pipe.predict(X_va[:, sel_idx])
    val_f1_rf = round(float(f1_score(y_va, y_pred_rf, average='macro')), 4)
    val_uar_rf = round(float(balanced_accuracy_score(y_va, y_pred_rf)), 4)

    # Breakdown per kelompok fitur
    fnames_df['selected'] = fnames_df['index'].isin(sel_idx)
    grp_summary = (
        fnames_df.groupby('group', sort=False)
        .agg(total=('index', 'count'), selected=('selected', 'sum'))
        .reset_index()
    )
    grp_summary['pct'] = (grp_summary['selected'] / grp_summary['total'] * 100).round(1)

    print('\n=== Hasil SA Utama (P0, seed=42) ===')
    print(f'Jumlah fitur terpilih (k) : {k} / {len(feat_names)} (reduksi {reduction_pct:.1f}%)')
    print(f'CV Macro F1 (train 4-fold): {sa_res["best_cv_f1"]:.4f}')
    print(f'Validation Macro F1 SVM   : {val_f1_svm:.4f} (Baseline 90 fitur: {bl_svm_val_f1:.4f}, delta: {val_f1_svm - bl_svm_val_f1:+.4f})')
    print(f'Validation Macro F1 RF    : {val_f1_rf:.4f} (Baseline 90 fitur: {bl_rf_val_f1:.4f}, delta: {val_f1_rf - bl_rf_val_f1:+.4f})')
    print(f'Waktu komputasi           : {sa_res["elapsed_s"]:.2f} detik (cache hits: {sa_res["cache_hits"]}, reheats: {sa_res["reheats_done"]})')
    print('\nSebaran per kelompok fitur:')
    print(grp_summary.to_string(index=False))

    payload = {
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'owner': 'Anggota B',
        'version': sa.SA_VERSION,
        'partition': 'P0',
        'seed': 42,
        'normalization': norm_ver,
        'n_features_total': len(feat_names),
        'k_selected': k,
        'feature_reduction_pct': reduction_pct,
        'selected_indices': sel_idx,
        'selected_feature_names': sel_names,
        'group_breakdown': grp_summary.to_dict(orient='records'),
        'sa_metrics': {
            'best_cv_f1': sa_res['best_cv_f1'],
            'best_obj': sa_res['best_obj'],
            'best_energy': sa_res['best_energy'],
            'accepted_worse': sa_res['accepted_worse'],
            'cache_hits': sa_res['cache_hits'],
            'reheats_done': sa_res['reheats_done'],
            'elapsed_s': sa_res['elapsed_s'],
        },
        'validation_comparison': {
            'svm_baseline_90': bl_svm_val_f1,
            'svm_sa_subset': val_f1_svm,
            'svm_sa_tuned_params': tuned_svm_params,
            'svm_sa_tuned_cv_f1': round(float(svm_search.best_score_), 4),
            'svm_delta': round(val_f1_svm - bl_svm_val_f1, 4),
            'svm_sa_uar': val_uar_svm,
            'svm_sa_accuracy': val_acc_svm,
            'svm_sa_per_class_f1': per_class_svm,
            'rf_baseline_90': bl_rf_val_f1,
            'rf_sa_subset': val_f1_rf,
            'rf_delta': round(val_f1_rf - bl_rf_val_f1, 4),
            'rf_sa_uar': val_uar_rf,
        },
        'config': sa_res['config'],
        'history': sa_res['history'],
    }

    with open(os.path.join(OUT_DIR, 'results', 'sa_main_results.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Simpan artefak model P0 untuk Anggota C (dilatih pada train P0 dengan subset SA)
    model_artifact = {
        'pipeline': svm_pipe,
        'normalization': norm_ver,
        'selected_indices': sel_idx,
        'selected_feature_names': sel_names,
        'svm_params': _clean_params(tuned_svm_params),
        'trained_on': 'P0 train (actors 1-16, 960 files)',
        'val_macro_f1': val_f1_svm,
    }
    joblib.dump(model_artifact, os.path.join(OUT_DIR, 'results', 'best_model_p0.joblib'))

    # Plot 1: Konvergensi SA
    hist_df = pd.DataFrame(sa_res['history'])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.2), dpi=200, sharex=True)
    ax1.plot(hist_df['iter'], hist_df['cur_cv_f1'], alpha=0.45, color='#718096', label='CV Macro F1 (saat ini)')
    ax1.plot(hist_df['iter'], hist_df['best_cv_f1'], color='#2b6cb0', linewidth=2.0, label='CV Macro F1 (terbaik)')
    ax1.plot(hist_df['iter'], hist_df['best_obj'], color='#2f855a', linestyle='--', linewidth=1.6, label='Objektif J(s) (terbaik)')
    ax1.set_ylabel('Skor')
    ax1.set_title(f'Konvergensi Simulated Annealing Utama (P0, versi {norm_ver}, k={k} fitur)')
    ax1.grid(linestyle='--', alpha=0.4)
    ax1.legend(loc='lower right')

    ax2.plot(hist_df['iter'], hist_df['cur_k'], alpha=0.45, color='#a0aec0', label='Jumlah fitur saat ini')
    ax2.plot(hist_df['iter'], hist_df['best_k'], color='#dd6b20', linewidth=2.0, label='Jumlah fitur terbaik')
    ax2.set_xlabel('Iterasi')
    ax2.set_ylabel('Jumlah Fitur Aktif (k)')
    ax2.grid(linestyle='--', alpha=0.4)
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'figures', 'sa_main_convergence.png'), dpi=200)
    plt.close(fig)

    # Plot 2: Sebaran kelompok fitur terpilih
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=200)
    x = np.arange(len(grp_summary))
    w = 0.38
    ax.bar(x - w / 2, grp_summary['total'], width=w, label='Total fitur (90)', color='#cbd5e0')
    bars = ax.bar(x + w / 2, grp_summary['selected'], width=w, label=f'Terpilih SA ({k})', color='#2b6cb0')
    for bar, pct in zip(bars, grp_summary['pct']):
        h = bar.get_height()
        ax.annotate(f'{int(h)} ({pct:.0f}%)', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(grp_summary['group'], rotation=20, ha='right')
    ax.set_ylabel('Jumlah Fitur')
    ax.set_title(f'Proporsi Fitur Terpilih SA Utama per Kelompok Akustik (k={k}/90)')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'figures', 'sa_main_feature_groups.png'), dpi=200)
    plt.close(fig)

    print('\nHasil tersimpan di results/sa_main_results.json, results/best_model_p0.joblib, dan figures/sa_main_*.png')


if __name__ == '__main__':
    main()
