"""
inference.py - Inference wrapper untuk integrasi model emosi (Anggota B) ke front-end.

Penanggung jawab : Anggota B
Versi            : 1.0

Membungkus seluruh alur inferensi secara end-to-end:
  1. Konversi audio mentah (file WAV atau array numpy dari Gradio/Streamlit)
  2. Preprocessing menggunakan src/preprocess.py (mono, 16 kHz, trim, normalisasi)
  3. Ekstraksi 90 fitur akustik menggunakan src/features.py
  4. Seleksi 52 fitur sesuai selected_indices di best_model_p0.joblib
  5. Prediksi label emosi dan skor relatif untuk 8 kelas emosi
"""
import os
import sys
import joblib
import numpy as np


def _resolve_repo_paths():
    """Temukan root repo AI-Projek dan path model default secara otomatis."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    progres_b_dir = os.path.dirname(this_dir)
    repo_root = os.path.dirname(progres_b_dir)
    root_src = os.path.join(repo_root, 'src')
    default_model = os.path.join(progres_b_dir, 'models', 'best_model_p0.joblib')
    return repo_root, root_src, default_model


_, _ROOT_SRC, DEFAULT_MODEL_PATH = _resolve_repo_paths()
if _ROOT_SRC not in sys.path:
    sys.path.insert(0, _ROOT_SRC)

import preprocess as pp  # noqa: E402
import features as ft    # noqa: E402

VOICED_RATIO_IDX = ft.FEATURE_NAMES.index('voiced_ratio')


def _to_float32_audio(y):
    """
    Konversi array audio (termasuk format int16/int32/stereo dari Gradio)
    menjadi float32 dalam rentang [-1.0, 1.0] dengan layout (n,) atau (ch, n).
    """
    arr = np.asarray(y)
    if arr.size == 0:
        raise ValueError('Audio kosong (0 sampel).')

    # Konversi tipe integer PCM ke float32 [-1, 1]
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        scale = float(max(abs(info.min), abs(info.max)))
        arr = arr.astype(np.float32) / scale
    else:
        arr = arr.astype(np.float32)

    # Gradio/soundfile mengembalikan (n_samples, n_channels);
    # librosa.to_mono mengharapkan (n_channels, n_samples)
    if arr.ndim == 2:
        if arr.shape[0] > arr.shape[1]:
            arr = arr.T
    elif arr.ndim > 2:
        raise ValueError(f'Dimensi audio tidak didukung: {arr.shape}')

    return arr


def _softmax(x):
    """Softmax stabil secara numerik untuk mengubah decision_function ke skor [0, 1]."""
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


class EmotionPredictor:
    """Wrapper inferensi siap pakai untuk front-end (Gradio, Streamlit, FastAPI)."""

    def __init__(self, model_path=None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f'File model tidak ditemukan: {self.model_path}')

        bundle = joblib.load(self.model_path)
        self.pipeline = bundle['pipeline']
        self.normalization = bundle.get('normalization', 'peak')
        self.selected_indices = list(bundle['selected_indices'])
        self.selected_feature_names = list(bundle.get('selected_feature_names', []))
        self.classes = [str(c) for c in self.pipeline.classes_]
        self.trained_on = bundle.get('trained_on', 'unknown')

    def _predict_preprocessed(self, y_pre, sr, info):
        min_samples = int(0.1 * sr)  # minimal 100 ms setelah trim hening
        if len(y_pre) < min_samples:
            raise ValueError(
                f'Audio terlalu pendek atau hening setelah trim ({len(y_pre)} sampel / '
                f'{len(y_pre) / sr:.3f} detik). Rekam ulang dengan suara lebih jelas.'
            )

        x_all = ft.extract_features(y_pre)
        x_sel = x_all[self.selected_indices].reshape(1, -1)

        pred_label = str(self.pipeline.predict(x_sel)[0])

        clf = self.pipeline.named_steps.get('clf', None)
        if clf is not None and getattr(clf, 'probability', False):
            raw_scores = self.pipeline.predict_proba(x_sel)[0]
            score_type = 'probability'
        elif hasattr(self.pipeline, 'decision_function'):
            raw_scores = _softmax(self.pipeline.decision_function(x_sel)[0])
            score_type = 'softmax_decision'
        else:
            raw_scores = np.array([1.0 if c == pred_label else 0.0 for c in self.classes])
            score_type = 'hard_label'

        scores = {
            cls: round(float(sc), 4)
            for cls, sc in sorted(
                zip(self.classes, raw_scores),
                key=lambda kv: kv[1],
                reverse=True,
            )
        }

        return {
            'emotion': pred_label,
            'scores': scores,
            'score_type': score_type,
            'normalization': self.normalization,
            'n_features_used': len(self.selected_indices),
            'duration_after_trim_s': round(len(y_pre) / float(sr), 3),
            'voiced_ratio': round(float(x_all[VOICED_RATIO_IDX]), 4),
            'preprocess_info': info,
        }

    def predict_from_file(self, audio_path):
        """Prediksi emosi langsung dari path file audio (.wav, .flac, dll.)."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f'File audio tidak ditemukan: {audio_path}')
        y_pre, sr, info = pp.preprocess_file(audio_path, normalize_method=self.normalization)
        return self._predict_preprocessed(y_pre, sr, info)

    def predict_from_array(self, y, sr):
        """
        Prediksi emosi dari array audio numpy dan sample rate.
        Cocok untuk input Gradio gr.Audio(type='numpy') yang mengirim (sr, y).
        """
        y_f32 = _to_float32_audio(y)
        y_pre, info = pp.preprocess_signal(y_f32, sr=int(sr), normalize_method=self.normalization)
        return self._predict_preprocessed(y_pre, info['sr'], info)

    def predict_gradio(self, audio_input):
        """
        Helper khusus untuk callback Gradio gr.Audio.
        Mendukung type='filepath' (str) maupun type='numpy' (tuple (sr, y)).
        """
        if audio_input is None:
            raise ValueError('Belum ada audio yang dimasukkan.')
        if isinstance(audio_input, str):
            return self.predict_from_file(audio_input)
        if isinstance(audio_input, tuple) and len(audio_input) == 2:
            sr, y = audio_input
            return self.predict_from_array(y, sr)
        raise TypeError(f'Format audio_input tidak dikenali: {type(audio_input)}')


_DEFAULT_PREDICTOR = None


def get_predictor(model_path=None):
    """Singleton helper agar model hanya dimuat sekali oleh front-end."""
    global _DEFAULT_PREDICTOR
    if model_path is not None:
        return EmotionPredictor(model_path=model_path)
    if _DEFAULT_PREDICTOR is None:
        _DEFAULT_PREDICTOR = EmotionPredictor()
    return _DEFAULT_PREDICTOR


def predict_from_file(audio_path, model_path=None):
    return get_predictor(model_path).predict_from_file(audio_path)


def predict_from_array(y, sr, model_path=None):
    return get_predictor(model_path).predict_from_array(y, sr)
