# Panduan Penggunaan Model Final (Train + Validation)

Panduan singkat ini menjelaskan spesifikasi, kontrak interface, cara evaluasi batch pada data `test`, serta cara integrasi front-end untuk model final yang dilatih pada gabungan **`train + validation` (aktor 1–20, 1.200 file)** melalui `train_final_models.py`.

---

## 1. Ringkasan Model Final

### A. Spesifikasi Umum

| Aspek | Spesifikasi |
| --- | --- |
| Lokasi berkas | `models/*.joblib` & manifest `models/final_models_manifest.json` |
| Data latih | Gabungan `train + validation` split utama `P0` (aktor 1–20, 1.200 file RAVDESS speech) |
| Data terkunci | `test` (aktor 21–24, 240 file) **tidak pernah disentuh** saat pelatihan |
| Normalisasi audio | `peak` (otomatis dijalankan di `inference.py` melalui `src/preprocess.py`) |
| 8 kelas emosi | `angry`, `calm`, `disgust`, `fearful`, `happy`, `neutral`, `sad`, `surprised` |
| Jenis skor output | `probability` (probabilitas terkalibrasi 0–1; `SVC` memakai `probability=True`) |
| Environment pelatihan | `scikit-learn 1.6.1`, `joblib 1.5.0`, `numpy 2.1.3` (+ `librosa`, `soundfile` untuk audio mentah) |

### B. Daftar Model Tersedia

| Berkas Model | Arsitektur & Hyperparameter | Fitur (`k`) | Diagnostik 5-Fold GroupKFold F1 (`train+val`) | Ukuran |
| --- | --- | :---: | :---: | :---: |
| `models/svm_all90_final.joblib` | `StandardScaler + SVC(rbf, C=1.0, gamma=0.005, balanced, probability=True)` | 90 | `0,4868 ± 0,0448` | 927,5 KB |
| `models/svm_sa52_final.joblib` | `StandardScaler + SVC(rbf, C=1.0, gamma="scale", balanced, probability=True)` | 52 (SA) | `0,5034 ± 0,0576` | 591,5 KB |
| `models/rf_all90_final.joblib` | `StandardScaler + RF(n=400, max_features=0.3, min_split=5, balanced)` | 90 | `0,4292 ± 0,0358` | 18,4 MB |
| `models/rf_sa52_final.joblib` | `StandardScaler + RF(n=400, max_features=0.3, min_split=5, balanced)` | 52 (SA) | `0,4394 ± 0,0445` | 19,7 MB |

---

## 2. Kontrak Interface & Struktur Artefak

### A. Pemakaian Lewat Wrapper (`progres_anggota_b/src/inference.py`)

Semua berkas `.joblib` di `models/` langsung kompatibel dengan `EmotionPredictor` dan `get_predictor`:

| Fungsi / Method | Format Input | Kegunaan |
| --- | --- | --- |
| `get_predictor(model_path)` | `str` path ke `models/*.joblib` | Memuat model final pilihan (misal `models/svm_sa52_final.joblib`). |
| `predictor.predict_gradio(audio_input)` | `str` (filepath) atau `tuple (sr, y)` | Callback langsung untuk Gradio `gr.Audio` (unggah file atau mikrofon). |
| `predictor.predict_from_file(audio_path)` | `str` path `.wav` / `.flac` | Prediksi dari file audio yang diunggah (Streamlit / FastAPI). |
| `predictor.predict_from_array(y, sr)` | `np.ndarray` + `int sr` | Prediksi dari buffer memori; otomatis konversi `int16` & stereo ke mono. |

### B. Skema Output Wrapper (`dict`) & Isi Bundle `.joblib`

| Key pada Output `predict_*` | Tipe | Contoh Nilai | Kegunaan di UI |
| --- | --- | --- | --- |
| `emotion` | `str` | `"angry"` | Label emosi utama hasil prediksi. |
| `scores` | `dict[str, float]` | `{"angry": 0.4312, ...}` | Probabilitas 8 emosi terurut menurun (siap untuk `gr.Label` / grafik). |
| `score_type` | `str` | `"probability"` | Menandakan skor berasal dari `predict_proba` (jumlah = 1,0). |
| `duration_after_trim_s` | `float` | `2.14` | Durasi ucapan aktif setelah trim hening (detik). |
| `voiced_ratio` | `float` | `0.7821` | Proporsi frame bersuara (0–1); indikator kualitas rekaman. |
| `n_features_used` | `int` | `52` atau `90` | Jumlah fitur aktif yang dipakai model terpilih. |

Setiap berkas `.joblib` menyimpan `dict` dengan key utama: `pipeline`, `model_name`, `normalization`, `selected_indices`, `selected_feature_names`, `params`, `trained_on`, `train_val_actors`, dan `test_actors_untouched`.

---

## 3. Langkah Sederhana Penggunaan

1. **Untuk evaluasi akhir (`test` bersih & bernoise):** Muat berkas `.joblib` langsung dengan `joblib.load`, potong kolom fitur memakai `bundle['selected_indices']`, lalu panggil `bundle['pipeline'].predict(X_test_sub)`.
2. **Untuk demo Front-End (Gradio / Streamlit):** Berikan path `models/svm_sa52_final.joblib` (atau `models/svm_all90_final.joblib`) ke `get_predictor(model_path=...)` sekali saat startup.
3. **Untuk melatih ulang (reproduksi):** Jalankan `python train_final_models.py` dari root repo `AI-Projek/`.

### Contoh A: Evaluasi Batch pada Data `test` Bersih / Bernoise (Anggota C)

```python
import joblib, pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score
import sys; sys.path.insert(0, 'src')
import data_splits as ds

feat_names = pd.read_csv('data/features/feature_names.csv')['name'].tolist()
df = pd.read_csv('data/features/features_peak.csv')
_, _, te = ds.split_frames(df, ds.load_splits('.'))

X_te, y_te = te[feat_names].values, te['emotion'].values

for path in ['models/svm_all90_final.joblib', 'models/svm_sa52_final.joblib']:
    b = joblib.load(path)
    y_pred = b['pipeline'].predict(X_te[:, b['selected_indices']])
    f1 = f1_score(y_te, y_pred, average='macro')
    uar = balanced_accuracy_score(y_te, y_pred)
    print(f"{b['model_name']:10s} (k={b['n_features']}) -> Test Macro-F1: {f1:.4f} | UAR: {uar:.4f}")
```

### Contoh B: Integrasi ke Gradio dengan Pilihan Model (`app.py`)

```python
import sys, gradio as gr
sys.path.insert(0, 'progres_anggota_b/src')
from inference import get_predictor

MODELS = {
    'SVM Subset SA (52 fitur)': get_predictor('models/svm_sa52_final.joblib'),
    'SVM Penuh (90 fitur)': get_predictor('models/svm_all90_final.joblib'),
}

def prediksi_emosi(audio, nama_model):
    try:
        res = MODELS[nama_model].predict_gradio(audio)
        info = f"Emosi: {res['emotion']} | Fitur: {res['n_features_used']} | Durasi: {res['duration_after_trim_s']}s"
        return res['scores'], info
    except ValueError as e:
        return {}, f"Input tidak valid: {e}"

gr.Interface(
    fn=prediksi_emosi,
    inputs=[gr.Audio(sources=['microphone', 'upload'], type='numpy'),
            gr.Dropdown(list(MODELS.keys()), value='SVM Subset SA (52 fitur)', label='Model')],
    outputs=[gr.Label(num_top_classes=8), gr.Textbox(label='Ringkasan')],
).launch()
```

---

## 4. Penanganan Error & Catatan Penting

| Kondisi / Error | Exception | Saran Penanganan |
| --- | --- | --- |
| Submit tanpa audio | `ValueError` | Tampilkan pesan agar pengguna merekam atau mengunggah audio. |
| Audio hening / `< 0,1` detik setelah trim | `ValueError` | Minta pengguna merekam ulang dengan suara lebih jelas. |
| Path berkas model / audio tidak ditemukan | `FileNotFoundError` | Pastikan menjalankan skrip dari root `AI-Projek/` dan file tersedia. |
| Salah slicing fitur saat evaluasi batch | `ValueError` | Selalu gunakan `X[:, bundle['selected_indices']]` sebelum `.predict()`. |

- **Aturan Evaluasi Test:** Partisi `test` (aktor 21–24) dibuka **satu kali saja** untuk pelaporan hasil akhir (bersih & 6 tabel `test_noisy_*_peak.csv`). Jangan gunakan hasil `test` untuk mengubah hyperparameter atau subset fitur.
- **Label Skor di UI:** Karena seluruh model final sudah menghasilkan `score_type = "probability"`, skor di UI dapat ditampilkan sebagai probabilitas kelas emosi (total 100%).
- **Kompatibilitas Versi:** Gunakan `scikit-learn` yang kompatibel (`1.6.x`) saat memuat berkas `.joblib`; jika versi environment berbeda, cukup jalankan ulang `python train_final_models.py` untuk meregenerasi seluruh berkas model.
