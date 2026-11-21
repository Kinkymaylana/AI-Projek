"""
features.py - Ekstraksi 90 fitur akustik untuk Proyek 07
(Finding the Best Acoustic Features).

Penanggung jawab : Anggota A
Versi            : 1.0 (22 November 2026)

Dipakai oleh:
  - Langkah 5 (Anggota A): membuat features_none.csv dan features_peak.csv
  - Demo Gradio (Anggota C): menghitung fitur rekaman pengguna
Masukan fungsi adalah audio yang SUDAH dipreprocessing dengan
src/preprocess.py (mono, 16 kHz, trim hening).

Setiap fitur adalah ringkasan (rata-rata, simpangan baku, dsb.) dari nilai
per frame. Frame: n_fft = 512 sampel (32 ms), hop = 160 sampel (10 ms).

Kelompok fitur (total 90):
  mfcc        26  MFCC ke-1..13, mean dan std (koefisien ke-0 tidak dipakai)
  delta_mfcc  26  delta MFCC ke-1..13, mean dan std
  prosodi      6  F0 (pyin): mean, std, persentil 5, persentil 95,
                  rentang (p95 - p5), proporsi frame bersuara
  energi       3  RMS: mean, std, max
  zcr          2  zero-crossing rate: mean, std
  spektral     8  centroid, bandwidth, rolloff, flatness: mean dan std
  contrast     7  spectral contrast 7 pita: mean
  chroma      12  chroma 12 kelas nada: mean

Hanya 3 fitur energi (rms_*) yang bergantung pada level suara absolut.
Fitur lain dihitung dari salinan sinyal yang dinormalisasi secara internal
(puncak = 1), sehingga benar-benar tidak berubah jika sinyal dikalikan
konstanta. Tanpa langkah ini, spectral flatness sedikit bergeser pada frame
yang sangat pelan karena librosa memakai batas bawah tetap (amin = 1e-10).
Karena itu F0 (bagian paling lambat) cukup dihitung sekali untuk semua
versi normalisasi.
"""

import numpy as np
import librosa

FEATURES_VERSION = '1.0'

DEFAULT_CONFIG = {
    'sr': 16000,
    'n_fft': 512,
    'hop_length': 160,
    'n_mfcc': 13,          # MFCC ke-1..13 (ke-0 dibuang)
    'n_mels': 40,
    'delta_width': 9,
    'f0_fmin': 65.0,
    'f0_fmax': 500.0,
    'f0_frame_length': 1024,
    'rolloff_percent': 0.85,
    'contrast_fmin': 100.0,
    'contrast_n_bands': 6,  # menghasilkan 7 nilai
}

CHROMA_LABELS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def _feature_table():
    """Daftar (nama, kelompok, deskripsi, bergantung_skala) untuk 90 fitur."""
    rows = []
    for stat in ('mean', 'std'):
        for i in range(1, 14):
            rows.append((f'mfcc{i:02d}_{stat}', 'mfcc',
                         f'MFCC koefisien ke-{i}, {stat} antarframe', False))
    for stat in ('mean', 'std'):
        for i in range(1, 14):
            rows.append((f'dmfcc{i:02d}_{stat}', 'delta_mfcc',
                         f'Delta (laju perubahan) MFCC ke-{i}, {stat} antarframe', False))
    rows += [
        ('f0_mean', 'prosodi', 'Rata-rata F0 (nada dasar) frame bersuara, Hz', False),
        ('f0_std', 'prosodi', 'Simpangan baku F0 frame bersuara, Hz', False),
        ('f0_p05', 'prosodi', 'Persentil 5 F0 frame bersuara, Hz', False),
        ('f0_p95', 'prosodi', 'Persentil 95 F0 frame bersuara, Hz', False),
        ('f0_range', 'prosodi', 'Rentang F0 = p95 - p05, Hz', False),
        ('voiced_ratio', 'prosodi', 'Proporsi frame bersuara (0-1)', False),
        ('rms_mean', 'energi', 'Rata-rata energi RMS antarframe', True),
        ('rms_std', 'energi', 'Simpangan baku energi RMS antarframe', True),
        ('rms_max', 'energi', 'Energi RMS frame terbesar', True),
        ('zcr_mean', 'zcr', 'Rata-rata zero-crossing rate', False),
        ('zcr_std', 'zcr', 'Simpangan baku zero-crossing rate', False),
    ]
    for name, desc in [('centroid', 'Spectral centroid (pusat massa spektrum), Hz'),
                       ('bandwidth', 'Spectral bandwidth (lebar sebaran spektrum), Hz'),
                       ('rolloff', 'Spectral rolloff (frekuensi batas 85% energi), Hz'),
                       ('flatness', 'Spectral flatness (0 = bernada, 1 = seperti noise)')]:
        rows.append((f'{name}_mean', 'spektral', f'{desc}, mean', False))
        rows.append((f'{name}_std', 'spektral', f'{desc}, std', False))
    for i in range(1, 8):
        rows.append((f'contrast{i}_mean', 'contrast',
                     f'Spectral contrast pita ke-{i} (selisih puncak-lembah), mean', False))
    for i, lab in enumerate(CHROMA_LABELS, start=1):
        rows.append((f'chroma{i:02d}_mean', 'chroma',
                     f'Chroma kelas nada {lab}, mean', False))
    return rows


FEATURE_TABLE = _feature_table()
FEATURE_NAMES = [r[0] for r in FEATURE_TABLE]
assert len(FEATURE_NAMES) == 90, len(FEATURE_NAMES)


def _cfg(config):
    merged = dict(DEFAULT_CONFIG)
    if config:
        merged.update({k: v for k, v in config.items() if k in DEFAULT_CONFIG})
    return merged


def _unit_peak(y):
    """Salinan sinyal dengan puncak 1 (untuk fitur yang tidak boleh bergantung volume)."""
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    return (y / peak).astype(np.float32) if peak > 0 else np.asarray(y, dtype=np.float32)


def extract_f0_features(y, config=None):
    """Hitung 6 fitur prosodi dengan pyin.

    Frame tanpa nada (unvoiced) diabaikan. Jika tidak ada frame bersuara,
    semua statistik F0 = 0 dan voiced_ratio = 0.

    Returns
    -------
    dict berisi f0_mean, f0_std, f0_p05, f0_p95, f0_range, voiced_ratio
    """
    c = _cfg(config)
    f0, voiced, _ = librosa.pyin(
        _unit_peak(np.asarray(y, dtype=np.float32)), fmin=c['f0_fmin'], fmax=c['f0_fmax'], sr=c['sr'],
        frame_length=c['f0_frame_length'], hop_length=c['hop_length'])
    ok = voiced & np.isfinite(f0)
    ratio = float(ok.mean()) if len(ok) else 0.0
    if ok.sum() == 0:
        return {'f0_mean': 0.0, 'f0_std': 0.0, 'f0_p05': 0.0, 'f0_p95': 0.0,
                'f0_range': 0.0, 'voiced_ratio': ratio}
    v = f0[ok]
    p05, p95 = np.percentile(v, [5, 95])
    return {'f0_mean': float(v.mean()), 'f0_std': float(v.std()),
            'f0_p05': float(p05), 'f0_p95': float(p95),
            'f0_range': float(p95 - p05), 'voiced_ratio': ratio}


def extract_features(y, config=None, f0_feats=None):
    """Hitung 90 fitur dari audio yang sudah dipreprocessing.

    Parameters
    ----------
    y : np.ndarray
        Audio mono 16 kHz hasil src/preprocess.py.
    config : dict, opsional
        Mengganti nilai DEFAULT_CONFIG.
    f0_feats : dict, opsional
        Hasil extract_f0_features untuk audio yang sama. Jika diberikan,
        pyin tidak dihitung ulang (F0 tidak bergantung pada volume).

    Returns
    -------
    np.ndarray (float64) berisi 90 nilai, urutannya sesuai FEATURE_NAMES
    """
    c = _cfg(config)
    sr, n_fft, hop = c['sr'], c['n_fft'], c['hop_length']
    y = np.asarray(y, dtype=np.float32)
    yu = _unit_peak(y)   # untuk semua fitur selain energi
    out = {}

    mfcc = librosa.feature.mfcc(y=yu, sr=sr, n_mfcc=c['n_mfcc'] + 1, n_mels=c['n_mels'],
                                n_fft=n_fft, hop_length=hop)[1:]
    n_frames = mfcc.shape[1]
    width = min(c['delta_width'], n_frames if n_frames % 2 == 1 else n_frames - 1)
    if width >= 3:
        dmfcc = librosa.feature.delta(mfcc, width=width)
    else:
        dmfcc = np.zeros_like(mfcc)
    for i in range(c['n_mfcc']):
        out[f'mfcc{i + 1:02d}_mean'] = mfcc[i].mean()
        out[f'mfcc{i + 1:02d}_std'] = mfcc[i].std()
        out[f'dmfcc{i + 1:02d}_mean'] = dmfcc[i].mean()
        out[f'dmfcc{i + 1:02d}_std'] = dmfcc[i].std()

    out.update(f0_feats if f0_feats is not None else extract_f0_features(yu, c))

    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    out['rms_mean'], out['rms_std'], out['rms_max'] = rms.mean(), rms.std(), rms.max()

    zcr = librosa.feature.zero_crossing_rate(yu, frame_length=n_fft, hop_length=hop)[0]
    out['zcr_mean'], out['zcr_std'] = zcr.mean(), zcr.std()

    spek = {
        'centroid': librosa.feature.spectral_centroid(y=yu, sr=sr, n_fft=n_fft, hop_length=hop)[0],
        'bandwidth': librosa.feature.spectral_bandwidth(y=yu, sr=sr, n_fft=n_fft, hop_length=hop)[0],
        'rolloff': librosa.feature.spectral_rolloff(y=yu, sr=sr, n_fft=n_fft, hop_length=hop,
                                                    roll_percent=c['rolloff_percent'])[0],
        'flatness': librosa.feature.spectral_flatness(y=yu, n_fft=n_fft, hop_length=hop)[0],
    }
    for name, v in spek.items():
        out[f'{name}_mean'], out[f'{name}_std'] = v.mean(), v.std()

    contrast = librosa.feature.spectral_contrast(y=yu, sr=sr, n_fft=n_fft, hop_length=hop,
                                                 fmin=c['contrast_fmin'], n_bands=c['contrast_n_bands'])
    for i in range(contrast.shape[0]):
        out[f'contrast{i + 1}_mean'] = contrast[i].mean()

    chroma = librosa.feature.chroma_stft(y=yu, sr=sr, n_fft=n_fft, hop_length=hop)
    for i in range(12):
        out[f'chroma{i + 1:02d}_mean'] = chroma[i].mean()

    return np.array([float(out[n]) for n in FEATURE_NAMES], dtype=np.float64)


def extract_features_dict(y, config=None, f0_feats=None):
    """Sama seperti extract_features, tetapi mengembalikan dict {nama: nilai}."""
    return dict(zip(FEATURE_NAMES, extract_features(y, config, f0_feats)))
