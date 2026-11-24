"""
train_final_models.py — Melatih dan menyimpan model final per metode pada gabungan
TRAIN + VALIDATION (aktor 1–20, 1.200 file) untuk Proyek 07 (RAVDESS speech).

Aturan anti-leakage:
  - Data TEST (aktor 21–24, 240 file) TIDAK pernah diakses oleh skrip ini.
  - Hyperparameter dan subset fitur diambil langsung dari hasil eksperimen
    Anggota B (baseline_results.json & sa_main_results.json) tanpa tuning ulang.

Model final yang dihasilkan (tersimpan di folder models/):
  1. svm_all90_final.joblib : SVM RBF (90 fitur, normalisasi peak)
  2. svm_sa52_final.joblib  : SVM RBF (52 fitur subset SA, normalisasi peak)
  3. rf_all90_final.joblib  : Random Forest (90 fitur, normalisasi peak)
  4. rf_sa52_final.joblib   : Random Forest (52 fitur subset SA, normalisasi peak)
  5. final_models_manifest.json : Ringkasan metadata seluruh model final

Format setiap berkas .joblib kompatibel langsung dengan
progres_anggota_b/src/inference.py (EmotionPredictor).
"""
import json
import os
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import data_splits as ds  # noqa: E402


def _clean_params(params):
    return {k.replace('clf__', ''): v for k, v in params.items()}


def _build_svm_pipeline(params, probability=True):
    clf_params = dict(params)
    clf_params.setdefault('class_weight', 'balanced')
    clf_params.setdefault('random_state', 42)
    clf_params['probability'] = probability
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(**clf_params)),
    ])


def _build_rf_pipeline(params):
    clf_params = dict(params)
    clf_params.setdefault('class_weight', 'balanced')
    clf_params.setdefault('random_state', 42)
    clf_params['n_jobs'] = -1
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(**clf_params)),
    ])


def train_and_save_model(
    name,
    filename,
    pipeline,
    X_tv,
    y_tv,
    g_tv,
    selected_indices,
    selected_feature_names,
    normalization,
    params,
    train_val_actors,
    test_actors,
    out_dir,
):
    X_sub = X_tv[:, selected_indices]

    t0 = time.time()
    # Diagnostik 5-fold GroupKFold pada train+val (20 aktor -> 4 aktor per fold), bukan untuk tuning
    cv = GroupKFold(n_splits=5)
    cv_scores = cross_val_score(pipeline, X_sub, y_tv, groups=g_tv, cv=cv, scoring='f1_macro', n_jobs=-1)

    # Fit final pada seluruh data train + validation (aktor 1–20, 1.200 file)
    pipeline.fit(X_sub, y_tv)
    elapsed_s = round(time.time() - t0, 2)

    created = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
        'created': created,
    }

    out_path = os.path.join(out_dir, filename)
    joblib.dump(artifact, out_path)

    size_kb = round(os.path.getsize(out_path) / 1024.0, 1)
    print(
        f'[{name:10s}] k={len(selected_indices):2d} | '
        f'5-fold GroupKFold F1={artifact["cv5_trainval_macro_f1_mean"]:.4f} '
        f'(+/- {artifact["cv5_trainval_macro_f1_std"]:.4f}) | '
        f'{elapsed_s:5.2f}s | {size_kb:6.1f} KB -> models/{filename}'
    )

    manifest_entry = {k: v for k, v in artifact.items() if k != 'pipeline'}
    manifest_entry['file'] = f'models/{filename}'
    manifest_entry['size_kb'] = size_kb
    return manifest_entry


def main():
    results_b_dir = os.path.join(REPO_DIR, 'progres_anggota_b', 'results')
    baseline_path = os.path.join(results_b_dir, 'baseline_results.json')
    sa_main_path = os.path.join(results_b_dir, 'sa_main_results.json')

    with open(baseline_path, encoding='utf-8') as f:
        bl_data = json.load(f)
    with open(sa_main_path, encoding='utf-8') as f:
        sa_data = json.load(f)

    norm_ver = bl_data['selected_normalization']
    assert norm_ver == sa_data['normalization'], (
        f'Ketidakcocokan normalisasi: baseline={norm_ver}, sa={sa_data["normalization"]}'
    )

    feat_path = os.path.join(REPO_DIR, 'data', 'features', f'features_{norm_ver}.csv')
    fnames_path = os.path.join(REPO_DIR, 'data', 'features', 'feature_names.csv')
    feat_df = pd.read_csv(feat_path)
    feat_names = pd.read_csv(fnames_path)['name'].tolist()
    assert len(feat_names) == 90, f'Jumlah fitur diharapkan 90, ditemukan {len(feat_names)}'

    splits = ds.load_splits(REPO_DIR)
    issues = ds.check_split(splits)
    assert not issues, f'Masalah pada splits.json: {issues}'

    tr, va, _ = ds.split_frames(feat_df, splits)
    tv = pd.concat([tr, va], ignore_index=True)

    train_val_actors = sorted(int(a) for a in tv['actor'].unique())
    test_actors = sorted(int(a) for a in splits['test'])
    assert set(train_val_actors).isdisjoint(set(test_actors)), 'KEBOCORAN: aktor test masuk ke train+val!'
    assert len(tv) == 1200, f'Jumlah baris train+val diharapkan 1200, ditemukan {len(tv)}'
    assert int(tv[feat_names].isna().sum().sum()) == 0, 'Terdapat nilai NaN pada fitur train+val!'

    X_tv = tv[feat_names].values
    y_tv = tv['emotion'].values
    g_tv = tv['actor'].values

    all90_indices = list(range(len(feat_names)))
    sa_indices = list(sa_data['selected_indices'])
    sa_names = [feat_names[i] for i in sa_indices]
    assert sa_names == list(sa_data['selected_feature_names']), 'Urutan nama fitur SA tidak cocok!'

    svm_all90_params = _clean_params(bl_data['results'][norm_ver]['SVM']['best_params'])
    svm_sa52_params = _clean_params(sa_data['validation_comparison']['svm_sa_tuned_params'])
    rf_params = _clean_params(bl_data['results'][norm_ver]['RF']['best_params'])

    out_dir = os.path.join(REPO_DIR, 'models')
    os.makedirs(out_dir, exist_ok=True)

    print(f'Melatih model final pada train + validation (normalisasi={norm_ver}, n={len(tv)}, aktor={train_val_actors[0]}..{train_val_actors[-1]})')
    print(f'Aktor TEST tetap dikunci dan tidak disentuh: {test_actors}\n')

    manifest = {
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'normalization': norm_ver,
        'train_val_actors': train_val_actors,
        'test_actors_untouched': test_actors,
        'n_train_val_samples': int(len(tv)),
        'sources': {
            'features': f'data/features/features_{norm_ver}.csv',
            'splits': 'data/splits.json',
            'baseline_results': 'progres_anggota_b/results/baseline_results.json',
            'sa_main_results': 'progres_anggota_b/results/sa_main_results.json',
        },
        'models': {},
    }

    specs = [
        (
            'SVM_all90',
            'svm_all90_final.joblib',
            _build_svm_pipeline(svm_all90_params, probability=True),
            all90_indices,
            feat_names,
            {**svm_all90_params, 'class_weight': 'balanced', 'probability': True, 'random_state': 42},
        ),
        (
            'SVM_SA52',
            'svm_sa52_final.joblib',
            _build_svm_pipeline(svm_sa52_params, probability=True),
            sa_indices,
            sa_names,
            {**svm_sa52_params, 'class_weight': 'balanced', 'probability': True, 'random_state': 42},
        ),
        (
            'RF_all90',
            'rf_all90_final.joblib',
            _build_rf_pipeline(rf_params),
            all90_indices,
            feat_names,
            {**rf_params, 'class_weight': 'balanced', 'random_state': 42},
        ),
        (
            'RF_SA52',
            'rf_sa52_final.joblib',
            _build_rf_pipeline(rf_params),
            sa_indices,
            sa_names,
            {**rf_params, 'class_weight': 'balanced', 'random_state': 42},
        ),
    ]

    for name, fname, pipe, idx, fnames, params in specs:
        entry = train_and_save_model(
            name=name,
            filename=fname,
            pipeline=pipe,
            X_tv=X_tv,
            y_tv=y_tv,
            g_tv=g_tv,
            selected_indices=idx,
            selected_feature_names=fnames,
            normalization=norm_ver,
            params=params,
            train_val_actors=train_val_actors,
            test_actors=test_actors,
            out_dir=out_dir,
        )
        manifest['models'][name] = entry

    manifest_path = os.path.join(out_dir, 'final_models_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f'\nSelesai! Manifest tersimpan di: models/final_models_manifest.json')


if __name__ == '__main__':
    main()
