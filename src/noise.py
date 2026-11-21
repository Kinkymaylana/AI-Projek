"""
noise.py — Penambahan noise pada SNR tertentu untuk Proyek 07 (test bernoise).

Penanggung jawab : Anggota A
Versi            : 1.0

SNR (signal-to-noise ratio) dihitung terhadap daya ucapan yang sudah di-trim:
    SNR_dB = 10 * log10(P_ucapan / P_noise)
Semakin kecil SNR, semakin keras noise-nya. Pada 0 dB, daya noise sama dengan daya ucapan.

Contoh:
    import noise as nz
    rng = np.random.default_rng(0)
    seg = nz.white_noise(len(y), rng)
    y_mix, gain = nz.mix_at_snr(y, seg, snr_db=10)
    nz.measure_snr(y, y_mix)   # ~10.0
"""
import numpy as np
import librosa

NOISE_VERSION = '1.0'


def signal_power(y):
    """Daya rata-rata sinyal (mean kuadrat), dihitung dalam float64."""
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean(y ** 2)) if y.size else 0.0


def white_noise(n, rng):
    """White noise Gaussian sepanjang n sampel (float32)."""
    return rng.standard_normal(n).astype(np.float32)


def fit_length(noise, n, offset=0):
    """Ambil potongan noise sepanjang n mulai dari offset; diulang (tile) jika noise lebih pendek."""
    noise = np.asarray(noise, dtype=np.float32)
    if noise.size == 0:
        raise ValueError('noise kosong')
    reps = int(np.ceil((offset % noise.size + n) / noise.size)) + 1
    long = np.tile(noise, reps)
    start = offset % noise.size
    return long[start:start + n]


def mix_at_snr(speech, noise_seg, snr_db):
    """Campur ucapan dan noise sehingga SNR = snr_db. Kembalikan (campuran float32, faktor pengali noise)."""
    speech = np.asarray(speech, dtype=np.float32)
    noise_seg = np.asarray(noise_seg, dtype=np.float32)
    if noise_seg.shape != speech.shape:
        raise ValueError('panjang noise harus sama dengan ucapan')
    ps, pn = signal_power(speech), signal_power(noise_seg)
    if ps == 0 or pn == 0:
        raise ValueError('daya ucapan atau noise nol')
    gain = float(np.sqrt(ps / (pn * 10.0 ** (snr_db / 10.0))))
    return (speech + np.float32(gain) * noise_seg).astype(np.float32), gain


def measure_snr(speech, mixed):
    """SNR terukur (dB) dari ucapan bersih dan hasil campuran."""
    speech = np.asarray(speech, dtype=np.float64)
    noise = np.asarray(mixed, dtype=np.float64) - speech
    return float(10.0 * np.log10(signal_power(speech) / signal_power(noise)))


def clean_noise_clip(y, top_db=40, min_samples=16000):
    """Buang bagian hening di klip noise (klip ESC-50 kadang diberi padding hening).

    Kembalikan klip tanpa hening, atau None jika sisa bunyinya lebih pendek dari min_samples.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0 or float(np.max(np.abs(y))) == 0.0:
        return None
    iv = librosa.effects.split(y, top_db=top_db)
    if len(iv) == 0:
        return None
    out = np.concatenate([y[s:e] for s, e in iv])
    return out if out.size >= min_samples else None
