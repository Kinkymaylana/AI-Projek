"""
baseline.py — Modul baseline modeling (SVM & Random Forest) untuk Proyek 07 (RAVDESS speech).

Penanggung jawab : Anggota B
Versi            : 1.0

Membandingkan versi normalisasi 'none' vs 'peak' menggunakan semua 90 fitur
dengan protokol speaker-independent (GroupKFold pada train, evaluasi di validation).
Data test (aktor 21-24) TIDAK pernah disentuh oleh modul ini.
"""
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASELINE_VERSION = '1.0'
EMOTION_ORDER = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']


def get_svm_search(cv_splits=4, n_jobs=-1):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(class_weight='balanced', random_state=42)),
    ])
    param_grid = [
        {
            'clf__kernel': ['rbf'],
            'clf__C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
            'clf__gamma': ['scale', 'auto', 0.01, 0.005, 0.001],
        },
        {
            'clf__kernel': ['linear'],
            'clf__C': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        },
    ]
    return GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=GroupKFold(n_splits=cv_splits),
        n_jobs=n_jobs,
        refit=True,
        return_train_score=False,
    )


def get_rf_search(cv_splits=4, n_jobs=-1):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=1)),
    ])
    param_grid = {
        'clf__n_estimators': [100, 200, 400],
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5],
        'clf__max_features': ['sqrt', 'log2', 0.3],
    }
    return GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=GroupKFold(n_splits=cv_splits),
        n_jobs=n_jobs,
        refit=True,
        return_train_score=False,
    )


def evaluate_model(search, X_tr, y_tr, g_tr, X_va, y_va):
    """Jalankan GridSearchCV di train (GroupKFold) lalu evaluasi sekali di validation."""
    t0 = time.time()
    search.fit(X_tr, y_tr, groups=g_tr)
    elapsed = time.time() - t0

    best_est = search.best_estimator_
    y_pred_va = best_est.predict(X_va)

    val_macro_f1 = float(f1_score(y_va, y_pred_va, average='macro'))
    val_uar = float(balanced_accuracy_score(y_va, y_pred_va))
    val_acc = float(accuracy_score(y_va, y_pred_va))
    report = classification_report(
        y_va, y_pred_va, labels=EMOTION_ORDER, output_dict=True, zero_division=0
    )
    per_class_f1 = {emo: round(float(report[emo]['f1-score']), 4) for emo in EMOTION_ORDER}

    return {
        'best_params': {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in search.best_params_.items()},
        'cv_macro_f1_mean': round(float(search.best_score_), 4),
        'cv_macro_f1_std': round(float(search.cv_results_['std_test_score'][search.best_index_]), 4),
        'val_macro_f1': round(val_macro_f1, 4),
        'val_uar': round(val_uar, 4),
        'val_accuracy': round(val_acc, 4),
        'per_class_f1_val': per_class_f1,
        'elapsed_s': round(elapsed, 2),
        'estimator': best_est,
    }


def run_baseline_comparison(tr_none, va_none, tr_peak, va_peak, feat_names, n_jobs=-1):
    """Bandingkan none vs peak untuk SVM dan RF pada split utama (train & validation saja)."""
    datasets = {
        'none': (tr_none, va_none),
        'peak': (tr_peak, va_peak),
    }
    models = {
        'SVM': get_svm_search,
        'RF': get_rf_search,
    }

    results = {}
    rows = []
    estimators = {}

    for version, (tr, va) in datasets.items():
        X_tr = tr[feat_names].values
        y_tr = tr['emotion'].values
        g_tr = tr['actor'].values
        X_va = va[feat_names].values
        y_va = va['emotion'].values

        results[version] = {}
        estimators[version] = {}
        for model_name, builder in models.items():
            res = evaluate_model(builder(cv_splits=4, n_jobs=n_jobs), X_tr, y_tr, g_tr, X_va, y_va)
            estimators[version][model_name] = res.pop('estimator')
            results[version][model_name] = res
            rows.append({
                'version': version,
                'model': model_name,
                'cv_macro_f1_mean': res['cv_macro_f1_mean'],
                'cv_macro_f1_std': res['cv_macro_f1_std'],
                'val_macro_f1': res['val_macro_f1'],
                'val_uar': res['val_uar'],
                'val_accuracy': res['val_accuracy'],
                'elapsed_s': res['elapsed_s'],
                'best_params': str(res['best_params']),
            })

    summary_df = pd.DataFrame(rows)
    best_row = summary_df.sort_values('val_macro_f1', ascending=False).iloc[0]
    selected_version = str(best_row['version'])
    selected_model = str(best_row['model'])

    return results, summary_df, estimators, selected_version, selected_model
