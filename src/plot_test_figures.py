"""
plot_test_figures.py - Gambar paper (bahasa Inggris) dari hasil test yang SUDAH tersimpan.

Penanggung jawab : Anggota C
Versi            : 1.0 (24 November 2026)

Skrip ini hanya membaca results/test_eval/test_summary.json. Model dan data test
tidak dibuka lagi, jadi aman dijalankan berulang kali.

Jalankan dari root repo:
    .venv\\Scripts\\python.exe src\\plot_test_figures.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)
import evaluate as ev  # noqa: E402

SUMMARY = os.path.join(REPO_ROOT, 'results', 'test_eval', 'test_summary.json')
FIG_DIR = os.path.join(REPO_ROOT, 'figures', 'test_eval')
CM_METHODS = {
    'SVM_all90': 'SVM, all 90 features',
    'SVM_SA52': 'SVM, SA subset (52 features)',
    'SVM_MI52': 'SVM, MI top-52',
}


NOISE_SUMMARY = os.path.join(REPO_ROOT, 'results', 'noise_eval', 'noise_summary.json')
# Warna mengikuti entitas, bukan peringkat; 90 fitur abu-abu sebagai rujukan.
NOISE_LINES = [('SVM_all90', 'All 90 features', '#6b6b6b', 'o'),
               ('SVM_SA52', 'SA subset (52)', '#2a78d6', 's'),
               ('SVM_MI52', 'MI top-52', '#d9822b', '^')]


def plot_noise(clean, noise):
    """Macro F1 SVM terhadap SNR, satu panel per jenis noise (skala y sama)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    x_labels = ['Clean', '20 dB', '10 dB', '0 dB']
    for ax, (kind, title) in zip(axes, [('white', 'White noise'), ('esc50', 'ESC-50 environmental noise')]):
        for method, label, color, marker in NOISE_LINES:
            ys = [clean[method]['macro_f1']] + [noise['results'][f'test_noisy_{kind}_{snr}db'][method]['macro_f1']
                                                for snr in (20, 10, 0)]
            ax.plot(range(4), ys, color=color, marker=marker, lw=2, ms=7, label=label)
        ax.set_xticks(range(4))
        ax.set_xticklabels(x_labels)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', color='#e5e5e5', lw=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel('Macro F1 (test, n=240)')
    axes[0].set_ylim(0, max(0.6, axes[0].get_ylim()[1]))
    axes[1].legend(frameon=False, loc='lower left', fontsize=9)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'noise_macro_f1_svm_en.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('figures/test_eval/noise_macro_f1_svm_en.png')


def main():
    with open(SUMMARY, encoding='utf-8') as f:
        s = json.load(f)
    os.makedirs(FIG_DIR, exist_ok=True)
    for method, title in CM_METHODS.items():
        cm = np.asarray(s['results'][method]['confusion_matrix'])
        fig = ev.plot_confusion(cm, title=f'{title} (test, n={cm.sum()})', lang='en',
                                save_path=os.path.join(FIG_DIR, f'cm_{method}_en.png'))
        plt.close(fig)
        print(f'figures/test_eval/cm_{method}_en.png')
    if os.path.exists(NOISE_SUMMARY):
        with open(NOISE_SUMMARY, encoding='utf-8') as f:
            plot_noise(s['results'], json.load(f))
    else:
        print('noise_summary.json belum ada; gambar noise dilewati')


if __name__ == '__main__':
    main()
