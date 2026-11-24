r"""
generate_figures_en.py — Generates English-language (IEEE paper ready, 200 DPI)
versions of Anggota B's figures into progres_anggota_b/figures/sa_en/.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RES_B = os.path.join(REPO_DIR, 'progres_anggota_b', 'results')
OUT_DIR = os.path.join(REPO_DIR, 'progres_anggota_b', 'figures', 'sa_en')
os.makedirs(OUT_DIR, exist_ok=True)

GROUP_LABELS_EN = {
    'mfcc': 'MFCC',
    'delta_mfcc': 'Delta MFCC',
    'prosodi': 'Prosody (F0)',
    'energi': 'Energy (RMS)',
    'zcr': 'ZCR',
    'spektral': 'Spectral Shape',
    'contrast': 'Spectral Contrast',
    'chroma': 'Chroma',
}


def plot_baseline_en():
    with open(os.path.join(RES_B, 'baseline_results.json'), encoding='utf-8') as f:
        bl = json.load(f)['results']

    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=200)
    models = ['SVM', 'RF']
    x = np.arange(len(models))
    w = 0.34
    none_scores = [bl['none'][m]['val_macro_f1'] for m in models]
    peak_scores = [bl['peak'][m]['val_macro_f1'] for m in models]

    b1 = ax.bar(x - w / 2, none_scores, width=w, label='No Normalization (none)', color='#2b6cb0')
    b2 = ax.bar(x + w / 2, peak_scores, width=w, label='Peak Normalization (peak)', color='#dd6b20')

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f'{h:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold',
            )

    ax.set_ylabel('Validation Macro-F1 (Actors 17–20)')
    ax.set_title('90-Feature Baseline Comparison: No Normalization vs. Peak Normalization')
    ax.set_xticks(x)
    ax.set_xticklabels(['SVM (RBF)', 'Random Forest'])
    ax.set_ylim(0, max(none_scores + peak_scores) + 0.12)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'baseline_none_vs_peak_en.png'), dpi=200)
    plt.close(fig)


def plot_sa_convergence_en():
    with open(os.path.join(RES_B, 'sa_main_results.json'), encoding='utf-8') as f:
        sa = json.load(f)

    hist_df = pd.DataFrame(sa['history'])
    k = sa['k_selected']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 6.0), dpi=200, sharex=True)

    ax1.plot(hist_df['iter'], hist_df['cur_cv_f1'], alpha=0.45, color='#718096', label='Current 4-Fold CV Macro-F1')
    ax1.plot(hist_df['iter'], hist_df['best_cv_f1'], color='#2b6cb0', linewidth=2.0, label='Best 4-Fold CV Macro-F1')
    ax1.plot(hist_df['iter'], hist_df['best_obj'], color='#2f855a', linestyle='--', linewidth=1.6, label='Best Objective J(s)')
    ax1.set_ylabel('Score')
    ax1.set_title(f'Simulated Annealing Convergence (Main Split P0, Peak Normalization, k={k})')
    ax1.grid(linestyle='--', alpha=0.4)
    ax1.legend(loc='lower right', frameon=True)

    ax2.plot(hist_df['iter'], hist_df['cur_k'], alpha=0.45, color='#a0aec0', label='Current Active Features')
    ax2.plot(hist_df['iter'], hist_df['best_k'], color='#dd6b20', linewidth=2.0, label='Best Subset Size (k)')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Active Features (k)')
    ax2.grid(linestyle='--', alpha=0.4)
    ax2.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'sa_main_convergence_en.png'), dpi=200)
    plt.close(fig)


def plot_sa_groups_en():
    with open(os.path.join(RES_B, 'sa_main_results.json'), encoding='utf-8') as f:
        sa = json.load(f)

    grp = pd.DataFrame(sa['group_breakdown'])
    grp['label'] = grp['group'].map(lambda g: GROUP_LABELS_EN.get(g, g))
    k = sa['k_selected']

    fig, ax = plt.subplots(figsize=(8.2, 4.5), dpi=200)
    x = np.arange(len(grp))
    w = 0.38
    ax.bar(x - w / 2, grp['total'], width=w, label='Total Features (90)', color='#cbd5e0')
    bars = ax.bar(x + w / 2, grp['selected'], width=w, label=f'Selected by SA ({k})', color='#2b6cb0')
    for bar, pct in zip(bars, grp['pct']):
        h = bar.get_height()
        ax.annotate(
            f'{int(h)} ({pct:.0f}%)',
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(grp['label'], rotation=20, ha='right')
    ax.set_ylabel('Number of Features')
    ax.set_title(f'Selected Features by Acoustic Group (Main SA Split P0, k={k}/90)')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'sa_main_feature_groups_en.png'), dpi=200)
    plt.close(fig)


def plot_stability_frequency_en():
    freq_df = pd.read_csv(os.path.join(RES_B, 'feature_selection_frequency.csv'))
    top25 = freq_df.sort_values(['count', 'index'], ascending=[False, True]).head(25).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.6, 6.8), dpi=200)
    ax.barh(top25['name'], top25['count'], color='#2b6cb0')
    ax.set_xlabel('Selection Frequency Across 15 Runs (5 Partitions P0–P4 × 3 Seeds)')
    ax.set_title('Top 25 Most Frequently Selected Features by Simulated Annealing')
    ax.set_xlim(0, 17.5)
    for i, (_, row) in enumerate(top25.iterrows()):
        grp_en = GROUP_LABELS_EN.get(row['group'], row['group'])
        ax.text(row['count'] + 0.2, i, f"{int(row['count'])}/15 ({grp_en})", va='center', fontsize=8)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'sa_stability_frequency_en.png'), dpi=200)
    plt.close(fig)


def plot_lambda_sweep_en():
    sw = pd.read_csv(os.path.join(REPO_DIR, 'results', 'lambda_sweep', 'lambda_sweep.csv'))
    fig, ax1 = plt.subplots(figsize=(7.6, 4.5), dpi=200)

    l1 = ax1.plot(sw['lambda'], sw['best_cv_f1'], 'o-', color='#2b6cb0', linewidth=2.0, label='4-Fold CV Macro-F1 (Train)')
    l2 = ax1.plot(sw['lambda'], sw['val_macro_f1'], 's-', color='#2f855a', linewidth=2.0, label='Validation Macro-F1 (Actors 17–20)')
    ax1.axvline(0.015, color='#e53e3e', linestyle=':', linewidth=1.6, label='Selected λ = 0.015')
    ax1.set_xlabel('Parsimony Penalty Weight (λ)')
    ax1.set_ylabel('Macro-F1 Score')
    ax1.set_ylim(0.42, 0.55)
    ax1.grid(linestyle='--', alpha=0.4)

    ax2 = ax1.twinx()
    l3 = ax2.plot(sw['lambda'], sw['k'], '^--', color='#dd6b20', linewidth=1.8, label='Selected Subset Size (k)')
    ax2.set_ylabel('Number of Selected Features (k)', color='#dd6b20')
    ax2.tick_params(axis='y', labelcolor='#dd6b20')
    ax2.set_ylim(25, 65)

    lines = l1 + l2 + [ax1.get_lines()[2]] + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left', fontsize=8.5, frameon=True)
    ax1.set_title('Simulated Annealing Trade-Off Across Parsimony Penalty (λ)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'lambda_sweep_tradeoff_en.png'), dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    plot_baseline_en()
    plot_sa_convergence_en()
    plot_sa_groups_en()
    plot_stability_frequency_en()
    plot_lambda_sweep_en()
    print('Generated 5 English figures in:', OUT_DIR)
