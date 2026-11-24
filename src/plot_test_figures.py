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


if __name__ == '__main__':
    main()
