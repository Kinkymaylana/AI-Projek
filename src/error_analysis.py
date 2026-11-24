"""
error_analysis.py - Error analysis pada hasil test yang SUDAH tersimpan (Langkah 11, Anggota C).

Penanggung jawab : Anggota C
Versi            : 1.0 (24 November 2026)

Tidak ada model yang dievaluasi ulang di test. Skrip membaca:
  - results/test_eval/test_predictions.csv   (prediksi 12 metode, dibuat sekali oleh run_test_evaluation.py)
  - data/features/features_peak.csv           (fitur, untuk melihat kenapa file mirip kelas lain)
  - models/svm_sa52_final.joblib               (hanya scaler-nya, untuk skala fitur yang sama dengan model)
  - audio asli RAVDESS (opsional, --audio-dir) untuk spektrogram dan kontur F0

Hasil:
  results/error_analysis/error_summary.json   pola kesalahan per emosi, intensitas, aktor, gender
  results/error_analysis/error_cases.csv      file contoh: label, prediksi, fitur yang paling "menarik" ke kelas salah
  figures/error_analysis/case_*.png           spektrogram + F0 file contoh vs file aktor yang sama di kelas tebakan

Jalankan dari root repo (audio tidak disimpan di repo):
    .venv\\Scripts\\python.exe src\\error_analysis.py --audio-dir "<folder ravdess_speech>"
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)
import data_splits as ds  # noqa: E402
import evaluate as ev  # noqa: E402

PRED_CSV = os.path.join(REPO_ROOT, 'results', 'test_eval', 'test_predictions.csv')
FEATURES_CSV = os.path.join(REPO_ROOT, 'data', 'features', 'features_peak.csv')
NAMES_CSV = os.path.join(REPO_ROOT, 'data', 'features', 'feature_names.csv')
MODEL = os.path.join(REPO_ROOT, 'models', 'svm_sa52_final.joblib')
OUT_DIR = os.path.join(REPO_ROOT, 'results', 'error_analysis')
FIG_DIR = os.path.join(REPO_ROOT, 'figures', 'error_analysis')
INFO = ['file', 'actor', 'gender', 'emotion', 'intensity']
EMO_CODE = {e: f'{i + 1:02d}' for i, e in enumerate(ev.LABELS)}

# File contoh dipilih dengan aturan tertulis dari prediksi tersimpan, mewakili pola kesalahan terbesar:
CASES = [
    ('Actor_23/03-01-04-01-02-02-23.wav',
     'sad -> disgust: confusion sad paling sering (13/32 pada SVM 90); salah di 12/12 model'),
    ('Actor_22/03-01-04-01-01-01-22.wav',
     'sad -> neutral: semua sad normal aktor 22 ditebak neutral; efek speaker'),
    ('Actor_21/03-01-04-02-01-01-21.wav',
     'sad strong -> happy: sad intensitas kuat tertukar dengan emosi arousal tinggi; salah di 12/12 model'),
    ('Actor_21/03-01-07-01-01-02-21.wav',
     'disgust -> angry: confusion disgust terbesar (13/32 pada SVM SA52); salah di 12/12 model'),
]


def summary_tables(p):
    models = [c for c in p.columns if c not in INFO]
    wrong = sum((p[m] != p['emotion']).astype(int) for m in models)
    ok = p['SVM_SA52'] == p['emotion']
    return {
        'n_files': int(len(p)), 'n_models': len(models),
        'files_wrong_all_models': int((wrong == len(models)).sum()),
        'files_right_all_models': int((wrong == 0).sum()),
        'mean_models_wrong_by_emotion': wrong.groupby(p['emotion']).mean().round(2).sort_values().to_dict(),
        'sad_predicted_as': {m: p.loc[p['emotion'] == 'sad', m].value_counts().to_dict()
                             for m in ('SVM_all90', 'SVM_SA52')},
        'disgust_predicted_as': {m: p.loc[p['emotion'] == 'disgust', m].value_counts().to_dict()
                                 for m in ('SVM_all90', 'SVM_SA52')},
        'svm_sa52_accuracy_by_intensity': ok.groupby(p['intensity']).mean().round(3).to_dict(),
        'svm_sa52_accuracy_by_actor': {int(k): v for k, v in ok.groupby(p['actor']).mean().round(3).items()},
        'svm_sa52_accuracy_by_gender': ok.groupby(p['gender']).mean().round(3).to_dict(),
        'svm_both_wrong': int(((p['SVM_all90'] != p['emotion']) & ~ok).sum()),
        'svm_both_wrong_same_label': int(((p['SVM_all90'] != p['emotion']) & ~ok
                                          & (p['SVM_all90'] == p['SVM_SA52'])).sum()),
    }


def pull_features(row, pred, feats, bundle, names):
    """Fitur (skala model SA52) yang membuat file lebih mirip kelas tebakan daripada kelas aslinya.

    Pusat kelas dihitung di train + validation (data latih model final). Skor per fitur:
    |z - pusat_asli| - |z - pusat_tebakan|; positif = fitur menarik file ke kelas tebakan.
    """
    idx = list(bundle['selected_indices'])
    cols = [names[i] for i in idx]
    scaler = bundle['pipeline'].named_steps['scaler']
    splits = ds.load_splits(REPO_ROOT)
    trval = feats[feats['actor'].isin(splits['train'] + splits['validation'])]
    z_all = pd.DataFrame(scaler.transform(trval[cols].to_numpy(float)), columns=cols)
    centers = z_all.groupby(trval['emotion'].to_numpy()).mean()
    z = pd.Series(scaler.transform(row[cols].to_numpy(float).reshape(1, -1))[0], index=cols)
    pull = (z - centers.loc[row['emotion']]).abs() - (z - centers.loc[pred]).abs()
    dist = ((centers - z) ** 2).sum(axis=1).pow(0.5).sort_values()
    top = pull.sort_values(ascending=False).head(5)
    return {
        'nearest_centers': [f'{e} {d:.2f}' for e, d in dist.head(3).items()],
        'dist_true': round(float(dist[row['emotion']]), 2), 'dist_pred': round(float(dist[pred]), 2),
        'top_pull_features': [f'{f} (z {z[f]:+.2f}; {row["emotion"]} {centers.loc[row["emotion"], f]:+.2f}, '
                              f'{pred} {centers.loc[pred, f]:+.2f})' for f in top.index],
    }


def sibling(file, emotion):
    """File aktor, kalimat, dan pengulangan yang sama dalam emosi lain (intensitas normal)."""
    d, base = file.split('/')
    parts = base.split('-')
    parts[2], parts[3] = EMO_CODE[emotion], '01'
    return f'{d}/{"-".join(parts)}'


def plot_case(i, case, pred, note, audio_dir):
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    import features as ft
    import preprocess as pp

    files = [(case, f"true {note['emotion']}, predicted {pred}"),
             (sibling(case, pred), f'same actor and sentence, {pred} (normal)')]
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), gridspec_kw={'height_ratios': [3, 2]})
    c = ft._cfg(None)
    for col, (f, title) in enumerate(files):
        path = os.path.join(audio_dir, *f.split('/'))
        y, sr = pp.load_audio(path, c['sr'])
        y, _ = pp.trim_silence(y)
        y = pp.normalize(y, 'peak')
        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=160,
                                                                 n_mels=64), ref=np.max)
        librosa.display.specshow(mel, sr=sr, hop_length=160, x_axis='time', y_axis='mel', ax=axes[0, col],
                                 cmap='magma', vmin=-70, vmax=0)
        axes[0, col].set_title(f'{f.split("/")[1]}\n{title}', fontsize=9, loc='left')
        f0, voiced, _ = librosa.pyin(y, fmin=c['f0_fmin'], fmax=c['f0_fmax'], sr=sr,
                                     frame_length=c['f0_frame_length'], hop_length=c['hop_length'])
        t = librosa.times_like(f0, sr=sr, hop_length=c['hop_length'])
        axes[1, col].plot(t, f0, color='#2a78d6', lw=1.6)
        axes[1, col].set_ylabel('F0 (Hz)')
        axes[1, col].set_xlabel('Time (s)')
        axes[1, col].set_ylim(50, 400)
        axes[1, col].grid(axis='y', color='#e5e5e5', lw=0.8)
        med = np.nanmedian(f0) if np.any(voiced) else float('nan')
        axes[1, col].text(0.99, 0.92, f'median {med:.0f} Hz, voiced {np.mean(voiced):.0%}',
                          transform=axes[1, col].transAxes, ha='right', fontsize=8, color='#555555')
        for side in ('top', 'right'):
            axes[1, col].spines[side].set_visible(False)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f'case_{i}_{note["emotion"]}_as_{pred}.png')
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    return os.path.relpath(out, REPO_ROOT)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--audio-dir', default='', help='folder ravdess_speech berisi Actor_01..Actor_24')
    args = ap.parse_args(argv)

    p = pd.read_csv(PRED_CSV)
    feats = pd.read_csv(FEATURES_CSV)
    names = list(pd.read_csv(NAMES_CSV).sort_values('index')['name'])
    bundle = joblib.load(MODEL)
    os.makedirs(OUT_DIR, exist_ok=True)

    summary = summary_tables(p)
    rows = []
    for i, (case, reason) in enumerate(CASES, 1):
        r = p[p['file'] == case].iloc[0]
        pred = r['SVM_SA52']
        info = pull_features(feats[feats['file'] == case].iloc[0], pred, feats, bundle, names)
        models = [c for c in p.columns if c not in INFO]
        row = {'case': i, 'file': case, 'actor': int(r['actor']), 'gender': r['gender'],
               'intensity': r['intensity'], 'emotion': r['emotion'], 'svm_sa52': pred,
               'svm_all90': r['SVM_all90'], 'models_wrong': int(sum(r[m] != r['emotion'] for m in models)),
               'reason': reason, **info, 'figure': ''}
        if args.audio_dir:
            row['figure'] = plot_case(i, case, pred, row, args.audio_dir)
        rows.append(row)
        print(f"case {i}: {case} {r['emotion']} -> {pred} ({row['models_wrong']}/12 model salah); "
              f"pusat terdekat {info['nearest_centers'][0]}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'error_cases.csv'), index=False)
    summary['cases'] = rows
    with open(os.path.join(OUT_DIR, 'error_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'Hasil: {os.path.relpath(OUT_DIR, REPO_ROOT)}' + (f', {os.path.relpath(FIG_DIR, REPO_ROOT)}'
                                                            if args.audio_dir else ' (tanpa gambar)'))


if __name__ == '__main__':
    main()
