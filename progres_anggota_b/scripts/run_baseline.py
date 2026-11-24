r"""
run_baseline.py — Menjalankan Tahap 2 (Baseline SVM & RF: none vs peak) untuk Anggota B.

Membaca data dari repo AI-Projek (read-only) dan menyimpan semua output ke
E:\College\UB\S2\AI\Projek akhir\Progres Anggota B\.
"""
import json
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_DIR = r'E:\College\UB\S2\AI\Projek akhir\AI-Projek'
OUT_DIR = r'E:\College\UB\S2\AI\Projek akhir\Progres Anggota B'

sys.path.insert(0, os.path.join(REPO_DIR, 'src'))
sys.path.insert(0, os.path.join(OUT_DIR, 'src'))

import data_splits as ds
import baseline as bl


def main():
    os.makedirs(os.path.join(OUT_DIR, 'results'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'figures'), exist_ok=True)

    fn_none = pd.read_csv(os.path.join(REPO_DIR, 'data', 'features', 'features_none.csv'))
    fn_peak = pd.read_csv(os.path.join(REPO_DIR, 'data', 'features', 'features_peak.csv'))
    feat_names = pd.read_csv(os.path.join(REPO_DIR, 'data', 'features', 'feature_names.csv'))['name'].tolist()

    splits = ds.load_splits(REPO_DIR)
    issues = ds.check_split(splits)
    assert not issues, f'Split bermasalah: {issues}'

    # PENTING: Hanya pakai train (aktor 1-16) dan validation (aktor 17-20). Test (21-24) TIDAK dipakai.
    tr_none, va_none, _ = ds.split_frames(fn_none, splits)
    tr_peak, va_peak, _ = ds.split_frames(fn_peak, splits)

    assert len(tr_none) == 960 and len(va_none) == 240
    assert len(tr_peak) == 960 and len(va_peak) == 240
    assert set(tr_none['actor']).isdisjoint(set(va_none['actor']))
    assert set(tr_none['actor']).isdisjoint({21, 22, 23, 24})
    assert set(va_none['actor']).isdisjoint({21, 22, 23, 24})

    print('Menjalankan baseline (90 fitur, SVM & RF, none vs peak)...')
    results, summary_df, _, selected_version, selected_model = bl.run_baseline_comparison(
        tr_none, va_none, tr_peak, va_peak, feat_names, n_jobs=-1
    )

    print('\n=== Ringkasan Baseline (Validation Macro F1) ===')
    print(summary_df[['version', 'model', 'cv_macro_f1_mean', 'val_macro_f1', 'val_uar', 'val_accuracy', 'elapsed_s']].to_string(index=False))
    print(f'\nVersi normalisasi terpilih: {selected_version} (model terbaik: {selected_model})')

    payload = {
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'owner': 'Anggota B',
        'version': bl.BASELINE_VERSION,
        'n_features': len(feat_names),
        'split_rule': 'train=actors 1-16 (960), validation=actors 17-20 (240); test (21-24) untouched',
        'cv_scheme': 'GroupKFold(n_splits=4) on train actors',
        'selected_normalization': selected_version,
        'selected_baseline_model': selected_model,
        'results': results,
    }

    with open(os.path.join(OUT_DIR, 'results', 'baseline_results.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    summary_df.to_csv(os.path.join(OUT_DIR, 'results', 'baseline_summary.csv'), index=False)

    # Plot perbandingan none vs peak
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
    models = ['SVM', 'RF']
    x = np.arange(len(models))
    w = 0.34
    none_scores = [results['none'][m]['val_macro_f1'] for m in models]
    peak_scores = [results['peak'][m]['val_macro_f1'] for m in models]

    b1 = ax.bar(x - w / 2, none_scores, width=w, label='Tanpa normalisasi (none)', color='#2b6cb0')
    b2 = ax.bar(x + w / 2, peak_scores, width=w, label='Normalisasi puncak (peak)', color='#dd6b20')

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.4f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Macro F1 (Validation, Aktor 17–20)')
    ax.set_title('Perbandingan Baseline 90 Fitur: Tanpa Normalisasi vs Normalisasi Puncak')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, max(none_scores + peak_scores) + 0.12)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'figures', 'baseline_none_vs_peak.png'), dpi=200)
    plt.close(fig)
    print('\nHasil tersimpan di results/baseline_results.json, results/baseline_summary.csv, dan figures/baseline_none_vs_peak.png')


if __name__ == '__main__':
    main()
