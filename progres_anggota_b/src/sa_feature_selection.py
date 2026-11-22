"""
sa_feature_selection.py — Modul Seleksi Fitur Simulated Annealing (SA) untuk Proyek 07.

Penanggung jawab : Anggota B
Versi            : 1.0

Protokol anti-kebocoran:
- Evaluasi objektif di dalam SA menggunakan 4-fold GroupKFold (berdasarkan aktor)
  hanya pada data train. StandardScaler di-fit terpisah di setiap fold train.
- Data validation (aktor 17-20 pada split utama) hanya dipakai untuk menilai subset
  akhir hasil SA; data test (aktor 21-24 pada split utama) TIDAK pernah disentuh.
"""
import time
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SA_VERSION = '1.0'


def _precompute_scaled_folds(X_tr, y_tr, g_tr, n_splits=4):
    """
    Precompute StandardScaler per fold (fit hanya pada fold train).
    Karena StandardScaler bersifat kolom-per-kolom, slicing kolom setelah scaling
    identik persis dengan Pipeline(StandardScaler, SVC) per subset.
    """
    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    for tr_idx, va_idx in gkf.split(X_tr, y_tr, groups=g_tr):
        scaler = StandardScaler()
        X_f_tr = scaler.fit_transform(X_tr[tr_idx])
        X_f_va = scaler.transform(X_tr[va_idx])
        folds.append((X_f_tr, y_tr[tr_idx], X_f_va, y_tr[va_idx]))
    return folds


def _make_svm(svm_params):
    """Buat instance SVC dari dict parameter (baik berawalan 'clf__' maupun tidak)."""
    clean = {}
    for k, v in svm_params.items():
        clean[k.replace('clf__', '')] = v
    clean.setdefault('class_weight', 'balanced')
    clean.setdefault('random_state', 42)
    return SVC(**clean)


def evaluate_subset_cv(mask, folds, svm_params, lambda_penalty=0.02):
    """
    Hitung skor objektif SA pada subset biner `mask`:
        J(s) = mean_groupkfold_macro_f1(s) - lambda_penalty * (|s| / D)
        E(s) = -J(s)  (energi untuk diminimalkan)
    """
    idx = np.flatnonzero(mask)
    D = len(mask)
    f1s = []
    for X_f_tr, y_f_tr, X_f_va, y_f_va in folds:
        clf = _make_svm(svm_params)
        clf.fit(X_f_tr[:, idx], y_f_tr)
        pred = clf.predict(X_f_va[:, idx])
        f1s.append(f1_score(y_f_va, pred, average='macro'))
    mean_f1 = float(np.mean(f1s))
    ratio = len(idx) / float(D)
    obj = mean_f1 - lambda_penalty * ratio
    return -obj, mean_f1  # (energy, cv_macro_f1)


def _propose_neighbor(mask, rng, high_temp=True, min_features=5):
    """Bangkitkan subset tetangga dengan adaptive bit-flip / swap."""
    D = len(mask)
    cand = mask.copy()
    if high_temp:
        # Eksplorasi lebih luas pada suhu tinggi: flip 1..3 bit
        n_flips = int(rng.integers(1, 4))
        flip_idx = rng.choice(D, size=n_flips, replace=False)
        cand[flip_idx] = ~cand[flip_idx]
    else:
        # Eksploitasi lokal pada suhu rendah: 70% flip 1 bit, 30% swap 1 aktif <-> 1 non-aktif
        active = np.flatnonzero(cand)
        inactive = np.flatnonzero(~cand)
        if rng.random() < 0.30 and len(active) > 0 and len(inactive) > 0:
            i_off = int(rng.choice(active))
            i_on = int(rng.choice(inactive))
            cand[i_off] = False
            cand[i_on] = True
        else:
            i = int(rng.integers(0, D))
            cand[i] = ~cand[i]

    # Pastikan jumlah fitur aktif >= min_features
    if cand.sum() < min_features:
        inactive = np.flatnonzero(~cand)
        need = min_features - int(cand.sum())
        add_idx = rng.choice(inactive, size=need, replace=False)
        cand[add_idx] = True
    return cand


def run_simulated_annealing(
    X_tr,
    y_tr,
    g_tr,
    svm_params,
    seed=42,
    n_iter=300,
    t0=0.05,
    t_min=1e-4,
    lambda_penalty=0.02,
    min_features=5,
    init_prob=0.5,
    stagnation_limit=60,
    max_reheats=2,
    n_splits=4,
    verbose=False,
):
    """
    Jalankan Simulated Annealing untuk seleksi fitur pada (X_tr, y_tr, g_tr).

    Returns
    -------
    dict dengan:
        - best_indices : list[int] terurut (0..D-1)
        - best_energy  : float
        - best_cv_f1   : float
        - best_n_features : int
        - history      : list[dict] jejak konvergensi per iterasi
        - cache_hits   : int
        - reheats_done : int
        - elapsed_s    : float
        - config       : dict parameter SA
    """
    t_start = time.time()
    rng = np.random.default_rng(seed)
    D = X_tr.shape[1]
    folds = _precompute_scaled_folds(X_tr, y_tr, g_tr, n_splits=n_splits)

    # Cooling rate geometrik agar suhu turun dari t0 ke t_min dalam n_iter langkah
    cooling_rate = (t_min / t0) ** (1.0 / max(n_iter, 1))

    # Inisialisasi acak terkendali (berbeda per seed)
    cur_mask = rng.random(D) < init_prob
    if cur_mask.sum() < min_features:
        cur_mask[rng.choice(D, size=min_features, replace=False)] = True

    cache = {}
    cache_hits = 0

    def eval_cached(m):
        nonlocal cache_hits
        key = m.tobytes()
        if key in cache:
            cache_hits += 1
            return cache[key]
        res = evaluate_subset_cv(m, folds, svm_params, lambda_penalty=lambda_penalty)
        cache[key] = res
        return res

    cur_E, cur_f1 = eval_cached(cur_mask)
    best_mask = cur_mask.copy()
    best_E, best_f1 = cur_E, cur_f1

    T = float(t0)
    stagnation = 0
    reheats_done = 0
    accepted_worse = 0

    history = [{
        'iter': 0,
        'temp': round(T, 6),
        'cur_cv_f1': round(cur_f1, 4),
        'best_cv_f1': round(best_f1, 4),
        'cur_obj': round(-cur_E, 4),
        'best_obj': round(-best_E, 4),
        'cur_k': int(cur_mask.sum()),
        'best_k': int(best_mask.sum()),
    }]

    for it in range(1, n_iter + 1):
        high_temp = T > 0.30 * t0
        cand_mask = _propose_neighbor(cur_mask, rng, high_temp=high_temp, min_features=min_features)
        cand_E, cand_f1 = eval_cached(cand_mask)

        delta_E = cand_E - cur_E
        if delta_E <= 0.0:
            cur_mask, cur_E, cur_f1 = cand_mask, cand_E, cand_f1
        else:
            prob = float(np.exp(-delta_E / max(T, 1e-12)))
            if rng.random() < prob:
                cur_mask, cur_E, cur_f1 = cand_mask, cand_E, cand_f1
                accepted_worse += 1

        if cur_E < best_E:
            best_mask = cur_mask.copy()
            best_E, best_f1 = cur_E, cur_f1
            stagnation = 0
        else:
            stagnation += 1

        # Reheating jika stagnan
        if stagnation >= stagnation_limit and reheats_done < max_reheats:
            T = max(T * 2.5, 0.35 * t0)
            cur_mask = best_mask.copy()
            cur_E, cur_f1 = best_E, best_f1
            stagnation = 0
            reheats_done += 1
        else:
            T = max(T * cooling_rate, t_min)

        history.append({
            'iter': it,
            'temp': round(T, 6),
            'cur_cv_f1': round(cur_f1, 4),
            'best_cv_f1': round(best_f1, 4),
            'cur_obj': round(-cur_E, 4),
            'best_obj': round(-best_E, 4),
            'cur_k': int(cur_mask.sum()),
            'best_k': int(best_mask.sum()),
        })

        if verbose and (it % 50 == 0 or it == n_iter):
            print(
                f'  [iter {it:3d}/{n_iter}] T={T:.5f} | best_obj={-best_E:.4f} '
                f'| best_cv_f1={best_f1:.4f} | k={int(best_mask.sum()):2d} | reheats={reheats_done}'
            )

    elapsed = time.time() - t_start
    best_indices = [int(i) for i in np.flatnonzero(best_mask)]

    return {
        'best_indices': best_indices,
        'best_energy': round(float(best_E), 6),
        'best_obj': round(float(-best_E), 6),
        'best_cv_f1': round(float(best_f1), 4),
        'best_n_features': len(best_indices),
        'accepted_worse': accepted_worse,
        'cache_hits': cache_hits,
        'reheats_done': reheats_done,
        'elapsed_s': round(elapsed, 2),
        'history': history,
        'config': {
            'seed': seed,
            'n_iter': n_iter,
            't0': t0,
            't_min': t_min,
            'lambda_penalty': lambda_penalty,
            'min_features': min_features,
            'init_prob': init_prob,
            'stagnation_limit': stagnation_limit,
            'max_reheats': max_reheats,
            'n_splits': n_splits,
            'svm_params': svm_params,
        },
    }
