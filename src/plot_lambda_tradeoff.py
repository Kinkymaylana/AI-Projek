"""
plot_lambda_tradeoff.py - Gambar paper trade-off lambda (Langkah 8, Anggota C).

Penanggung jawab : Anggota C
Versi            : 1.0 (24 November 2026)

Membaca results/lambda_sweep/lambda_sweep.csv (sweep Anggota B, seed 42, train dan
validation saja). Data test tidak dipakai: hanya lambda final yang dievaluasi di test.

Dua panel dengan sumbu x yang sama, bukan satu grafik dua sumbu y:
  atas  : macro F1 (CV 4-fold di train dan validation) terhadap lambda
  bawah : jumlah fitur terpilih k terhadap lambda

Jalankan dari root repo:
    .venv\\Scripts\\python.exe src\\plot_lambda_tradeoff.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP_CSV = os.path.join(REPO_ROOT, 'results', 'lambda_sweep', 'lambda_sweep.csv')
FIG_PATH = os.path.join(REPO_ROOT, 'figures', 'lambda', 'lambda_tradeoff_en.png')
SELECTED = 0.015  # lambda final (docs/jawaban_anggota_B_ke_C.md, poin 4)
N_FEATURES = 90


def main():
    d = pd.read_csv(SWEEP_CSV).sort_values('lambda')
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True,
                                      gridspec_kw={'height_ratios': [3, 2]})
    top.plot(d['lambda'], d['best_cv_f1'], color='#9a9a9a', marker='o', lw=2, ms=7,
             label='4-fold CV on train (actors 1-16)')
    top.plot(d['lambda'], d['val_macro_f1'], color='#2a78d6', marker='s', lw=2, ms=7,
             label='Validation (actors 17-20)')
    top.set_ylabel('Macro F1')
    top.legend(frameon=False, fontsize=9, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2,
               borderaxespad=0.2)

    bottom.plot(d['lambda'], d['k'], color='#3d3d3d', marker='D', lw=2, ms=6)
    for lam, k in zip(d['lambda'], d['k']):
        bottom.annotate(str(int(k)), (lam, k), textcoords='offset points', xytext=(0, 7),
                        ha='center', fontsize=8.5, color='#3d3d3d')
    bottom.set_ylabel(f'Selected features k\n(of {N_FEATURES})')
    bottom.set_xlabel('Parsimony penalty λ')
    bottom.set_ylim(min(d['k']) - 8, max(d['k']) + 10)

    for ax in (top, bottom):
        ax.axvline(SELECTED, color='#555555', ls=':', lw=1.2)
        ax.grid(axis='y', color='#e5e5e5', lw=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    lo, hi = top.get_ylim()
    top.annotate(f'selected λ = {SELECTED}', (SELECTED, lo + 0.25 * (hi - lo)), xytext=(5, 0),
                 textcoords='offset points', fontsize=8.5, color='#555555', va='center')
    fig.text(0.5, -0.03, 'SA sweep by Member B, seed 42; test set not used. '
             'λ = 0.005 and λ = 0.015 select the same 52-feature subset.',
             ha='center', fontsize=8.5, color='#555555')
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    fig.savefig(FIG_PATH, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(os.path.relpath(FIG_PATH, REPO_ROOT))


if __name__ == '__main__':
    main()
