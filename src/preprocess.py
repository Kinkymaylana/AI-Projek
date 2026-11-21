"""
preprocess.py - Preprocessing audio RAVDESS untuk Proyek 07
(Finding the Best Acoustic Features).

Penanggung jawab : Anggota A
Versi            : 1.0 (21 November 2026)

Modul ini adalah SATU-SATUNYA tempat aturan preprocessing ditulis.
Dipakai oleh:
  - Langkah 4 (Anggota A): membuat audio terproses dan log preprocessing
  - Langkah 5 (Anggota A): ekstraksi fitur
  - Demo Gradio (Anggota C): memproses rekaman pengguna
Dengan begitu semua bagian proyek memproses audio dengan cara yang sama.

Urutan proses:
  1. Muat audio, ubah ke mono (rata-rata channel)
  2. Resample ke 16 kHz
  3. Trim hening di awal dan akhir (top_db = 30)
  4. Normalisasi amplitudo (opsional): 'none' atau 'peak'

Catatan: trim memakai ambang relatif terhadap bagian terkeras file,
sehingga hasil trim tidak berubah jika sinyal dikalikan konstanta.
Karena itu normalisasi puncak boleh dilakukan SETELAH trim, dan audio
terproses cukup disimpan satu versi (tanpa normalisasi).
"""

import numpy as np
import librosa

PREPROCESS_VERSION = '1.0'

DEFAULT_CONFIG = {
    'target_sr': 16000,        # sample rate hasil (Hz)
    'mono': True,              # gabungkan channel menjadi satu
    'trim_top_db': 30,         # ambang trim hening (dB di bawah puncak)
    'trim_frame_length': 2048,  # panjang frame untuk deteksi hening
    'trim_hop_length': 512,    # pergeseran frame untuk deteksi hening
    'peak_target': 0.99,       # puncak setelah normalisasi 'peak'
}

NORMALIZE_METHODS = ('none', 'peak')


def _cfg(config):
    """Gabungkan config pengguna dengan DEFAULT_CONFIG."""
    merged = dict(DEFAULT_CONFIG)
    if config:
        merged.update(config)
    return merged


def load_audio(path, target_sr=16000, mono=True):
    """Muat file audio, ubah ke mono, lalu resample.

    Parameters
    ----------
    path : str
        Lokasi file audio (WAV).
    target_sr : int
        Sample rate tujuan dalam Hz.
    mono : bool
        Jika True, semua channel dirata-rata menjadi satu.

    Returns
    -------
    y : np.ndarray (float32, 1 dimensi)
    sr : int
    """
    y, sr = librosa.load(path, sr=target_sr, mono=mono)
    return y.astype(np.float32), sr


def trim_silence(y, top_db=30, frame_length=2048, hop_length=512):
    """Buang hening di awal dan akhir sinyal.

    Returns
    -------
    y_trim : np.ndarray
    (start, end) : tuple of int
        Posisi awal dan akhir bagian yang dipertahankan, dalam sampel.
    """
    y_trim, (start, end) = librosa.effects.trim(
        y, top_db=top_db,
        frame_length=frame_length, hop_length=hop_length)
    return y_trim, (int(start), int(end))


def normalize(y, method='none', peak_target=0.99):
    """Normalisasi amplitudo.

    method = 'none' : sinyal dikembalikan apa adanya.
    method = 'peak' : sinyal diskalakan sehingga max(|y|) = peak_target.
    """
    if method not in NORMALIZE_METHODS:
        raise ValueError(
            f"method harus salah satu dari {NORMALIZE_METHODS}, "
            f"bukan {method!r}")
    if method == 'none':
        return y
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak == 0.0:
        return y
    return (y * (peak_target / peak)).astype(np.float32)


def preprocess_signal(y, sr, config=None, normalize_method='none'):
    """Preprocessing untuk sinyal yang sudah dimuat (dipakai demo).

    Parameters
    ----------
    y : np.ndarray
        Sinyal float. Mono berbentuk (n,), multichannel (channel, n).
        Catatan untuk demo Gradio: audio Gradio berbentuk (n, channel)
        dan bertipe int16; ubah dulu: y = y.T.astype(np.float32) / 32768.
    sr : int
        Sample rate sinyal masukan.

    Returns
    -------
    y_out : np.ndarray (float32)
    info : dict
    """
    c = _cfg(config)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1 and c['mono']:
        y = librosa.to_mono(y)
    if sr != c['target_sr']:
        y = librosa.resample(y, orig_sr=sr, target_sr=c['target_sr'])
        sr = c['target_sr']
    n_in = len(y)
    y_trim, (start, end) = trim_silence(
        y, c['trim_top_db'], c['trim_frame_length'], c['trim_hop_length'])
    peak_trim = float(np.max(np.abs(y_trim))) if len(y_trim) else 0.0
    rms_trim = float(np.sqrt(np.mean(y_trim ** 2))) if len(y_trim) else 0.0
    y_out = normalize(y_trim, normalize_method, c['peak_target'])
    info = {
        'sr': sr,
        'n_samples_in': n_in,
        'trim_start': start,
        'trim_end': end,
        'peak_trim': peak_trim,
        'rms_trim_db': 20 * np.log10(rms_trim + 1e-10),
        'normalize': normalize_method,
    }
    return y_out.astype(np.float32), info


def preprocess_file(path, config=None, normalize_method='none'):
    """Muat satu file lalu jalankan seluruh preprocessing.

    Returns
    -------
    y_out : np.ndarray (float32)
    sr : int
    info : dict (lihat preprocess_signal)
    """
    c = _cfg(config)
    y, sr = load_audio(path, c['target_sr'], c['mono'])
    y_out, info = preprocess_signal(y, sr, c, normalize_method)
    return y_out, info['sr'], info
