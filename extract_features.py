#!/usr/bin/env python3
"""
extract_features.py — Membuat ulang tabel fitur Proyek 07 dari audio asli RAVDESS.

Penanggung jawab : Anggota A - Ito
Versi            : 1.0

Satu perintah ini menjalankan seluruh alur langkah 4 dan 5 dari nol:
  audio asli 48 kHz -> preprocessing (src/preprocess.py) -> 90 fitur (src/features.py)
  -> features_none.csv, features_peak.csv, feature_names.csv, features_meta.json

Semua parameter dibaca dari config.json (bagian "preprocessing" dan "features"),
sehingga hasilnya identik dengan tabel fitur resmi selama config dan versi library sama.

Contoh (dijalankan dari folder project07/):
  python extract_features.py --config config.json
  python extract_features.py --config config.json --compare data/features
  python extract_features.py --config config.json --actors 1 2 --out-dir data/features_repro_test

Kode keluar (exit code): 0 = sukses (dan identik jika --compare dipakai),
1 = ada file gagal atau hasil tidak identik, 2 = masalah input/konfigurasi.
"""
import os
import sys
import json
import glob
import time
import hashlib
import argparse
import platform
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

SCRIPT_VERSION = '1.0'
INFO_COLS = ['file', 'actor', 'gender', 'emotion', 'intensity']
EMOTIONS = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
            5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}
INTENSITIES = {1: 'normal', 2: 'strong'}
PRE_KEYS = ['target_sr', 'mono', 'trim_top_db', 'trim_frame_length', 'trim_hop_length', 'peak_target']
FEAT_KEYS = ['sr', 'n_fft', 'hop_length', 'n_mfcc', 'n_mels', 'delta_width', 'f0_fmin', 'f0_fmax',
             'f0_frame_length', 'rolloff_percent', 'contrast_fmin', 'contrast_n_bands']

# Diisi di setiap proses pekerja oleh _init_worker
_PP = _FT = _PRE = _FEAT = _RAW = None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Buat ulang tabel fitur Proyek 07 dari audio asli RAVDESS.')
    p.add_argument('--config', default='config.json', help='Lokasi config.json (default: config.json)')
    p.add_argument('--raw-dir', default=None,
                   help='Folder audio asli berisi Actor_XX/ (default: data/raw/ravdess_speech di sebelah config)')
    p.add_argument('--out-dir', default=None,
                   help='Folder output (default: data/features_repro). Tidak menimpa data/features kecuali diminta.')
    p.add_argument('--actors', type=int, nargs='+', default=None,
                   help='Hanya proses aktor tertentu, misalnya --actors 1 2 (untuk uji cepat)')
    p.add_argument('--compare', default=None,
                   help='Folder tabel fitur pembanding, misalnya data/features. Hasil dibandingkan sel per sel.')
    p.add_argument('--n-jobs', type=int, default=1, help='Jumlah proses paralel (default 1)')
    p.add_argument('--rtol', type=float, default=1e-6, help='Toleransi relatif saat membandingkan (default 1e-6)')
    p.add_argument('--atol', type=float, default=1e-9, help='Toleransi absolut saat membandingkan (default 1e-9)')
    return p.parse_args(argv)


def config_hashes(pre_cfg, feat_cfg):
    """Hitung config_hash persis seperti notebook langkah 4 dan 5."""
    pre_hash = hashlib.md5(json.dumps(pre_cfg, sort_keys=True).encode()).hexdigest()[:10]
    feat_hash = hashlib.md5(json.dumps({**feat_cfg, 'pre': pre_hash}, sort_keys=True).encode()).hexdigest()[:10]
    return pre_hash, feat_hash


def parse_filename(rel_path):
    """Actor_01/03-01-05-02-01-01-01.wav -> kolom info."""
    parts = os.path.splitext(os.path.basename(rel_path))[0].split('-')
    if len(parts) != 7:
        raise ValueError(f'Nama file tidak sesuai format RAVDESS: {rel_path}')
    emo, inten, actor = int(parts[2]), int(parts[3]), int(parts[6])
    return {'file': rel_path.replace(os.sep, '/'), 'actor': actor,
            'gender': 'male' if actor % 2 == 1 else 'female',
            'emotion': EMOTIONS[emo], 'intensity': INTENSITIES[inten]}


def list_files(raw_dir, actors=None):
    """Daftar file dengan urutan sama seperti metadata.csv (langkah 1: sorted glob)."""
    paths = sorted(glob.glob(os.path.join(raw_dir, '**', '*.wav'), recursive=True))
    rows = [parse_filename(os.path.relpath(p, raw_dir)) for p in paths]
    if actors:
        rows = [r for r in rows if r['actor'] in set(actors)]
    return pd.DataFrame(rows, columns=INFO_COLS)


def _init_worker(src_dir, pre_cfg, feat_cfg, raw_dir):
    global _PP, _FT, _PRE, _FEAT, _RAW
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import preprocess as pp
    import features as ft
    _PP, _FT, _PRE, _FEAT, _RAW = pp, ft, pre_cfg, feat_cfg, raw_dir


def _process_one(rel_path):
    """Satu file: preprocessing (langkah 4) lalu fitur dua versi (langkah 5)."""
    try:
        y, sr, _ = _PP.preprocess_file(os.path.join(_RAW, rel_path), _PRE, 'none')
        # Langkah 4 menyimpan audio sebagai WAV float32 dan langkah 5 membacanya kembali;
        # float32 tersimpan tanpa kehilangan, jadi cukup dipastikan bertipe float32.
        y = np.asarray(y, dtype=np.float32)
        f0 = _FT.extract_f0_features(y, _FEAT)
        x_none = _FT.extract_features(y, _FEAT, f0_feats=f0)
        y_peak = _PP.normalize(y, 'peak', _PRE['peak_target'])
        x_peak = _FT.extract_features(y_peak, _FEAT, f0_feats=f0)
        return rel_path, x_none, x_peak, None
    except Exception as e:  # dilaporkan, tidak menghentikan proses
        return rel_path, None, None, f'{type(e).__name__}: {e}'


def compare_tables(new_path, ref_path, rtol, atol):
    """Bandingkan dua tabel fitur. Kembalikan dict ringkasan."""
    new = pd.read_csv(new_path, float_precision='round_trip')
    ref = pd.read_csv(ref_path, float_precision='round_trip')
    rep = {'file_baru': new_path, 'file_pembanding': ref_path,
           'baris_baru': len(new), 'baris_pembanding': len(ref)}
    rep['kolom_sama'] = list(new.columns) == list(ref.columns)
    ref = ref[ref['file'].isin(new['file'])]           # mendukung --actors
    rep['baris_dibandingkan'] = len(ref)
    rep['urutan_file_sama'] = list(new['file']) == list(ref['file'])
    ref = ref.set_index('file').loc[new['file']].reset_index()
    info = [c for c in INFO_COLS if c != 'file']
    rep['info_sama'] = bool((new[info].astype(str).values == ref[info].astype(str).values).all())
    fcols = [c for c in new.columns if c not in INFO_COLS]
    a = new[fcols].to_numpy(dtype=float)
    b = ref[fcols].to_numpy(dtype=float)
    diff = np.abs(a - b)
    rep['jumlah_sel'] = int(a.size)
    rep['sel_identik_persis'] = int((a == b).sum())
    rep['selisih_abs_maks'] = float(diff.max()) if diff.size else 0.0
    rel = diff / np.maximum(np.abs(b), 1e-12)
    rep['selisih_rel_maks'] = float(rel.max()) if rel.size else 0.0
    close = np.isclose(a, b, rtol=rtol, atol=atol)
    rep['sel_dalam_toleransi'] = int(close.sum())
    worst = np.argsort(-(diff.max(axis=0)))[:3] if diff.size else []
    rep['fitur_selisih_terbesar'] = {fcols[i]: float(diff[:, i].max()) for i in worst}
    rep['identik'] = bool(rep['kolom_sama'] and rep['info_sama'] and close.all()
                          and rep['baris_dibandingkan'] == len(new))
    return rep


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    cfg_path = os.path.abspath(args.config)
    if not os.path.exists(cfg_path):
        print(f'[ERROR] config tidak ditemukan: {cfg_path}')
        return 2
    base = os.path.dirname(cfg_path)
    with open(cfg_path, encoding='utf-8') as f:
        config = json.load(f)
    for sec in ('preprocessing', 'features'):
        if sec not in config:
            print(f'[ERROR] config.json tidak punya bagian "{sec}". Jalankan notebook langkah 4 dan 5 dulu.')
            return 2

    pre_cfg = {k: config['preprocessing'][k] for k in PRE_KEYS}
    feat_cfg = {k: config['features'][k] for k in FEAT_KEYS}
    pre_hash, feat_hash = config_hashes(pre_cfg, feat_cfg)
    src_dir = os.path.join(base, 'src')
    raw_dir = os.path.abspath(args.raw_dir or os.path.join(base, 'data', 'raw', 'ravdess_speech'))
    out_dir = os.path.abspath(args.out_dir or os.path.join(base, 'data', 'features_repro'))

    for p, nama in [(os.path.join(src_dir, 'preprocess.py'), 'src/preprocess.py'),
                    (os.path.join(src_dir, 'features.py'), 'src/features.py'), (raw_dir, 'folder audio asli')]:
        if not os.path.exists(p):
            print(f'[ERROR] {nama} tidak ditemukan: {p}')
            return 2

    print(f'extract_features.py v{SCRIPT_VERSION}')
    print(f'  config     : {cfg_path}')
    print(f'  audio asli : {raw_dir}')
    print(f'  output     : {out_dir}')
    for nama, h_cfg, h_hit in [('preprocessing', config['preprocessing'].get('config_hash'), pre_hash),
                               ('features', config['features'].get('config_hash'), feat_hash)]:
        tanda = 'cocok' if h_cfg == h_hit else f'BERBEDA dari config.json ({h_cfg})'
        print(f'  config_hash {nama:<13}: {h_hit} ({tanda})')

    files = list_files(raw_dir, args.actors)
    if files.empty:
        print('[ERROR] tidak ada file WAV yang ditemukan.')
        return 2
    print(f'  file       : {len(files)}' + (f' (aktor {args.actors})' if args.actors else ''))

    _init_worker(src_dir, pre_cfg, feat_cfg, raw_dir)
    ft = _FT
    hasil, gagal = {}, {}
    n = len(files)
    if args.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=args.n_jobs, initializer=_init_worker,
                                 initargs=(src_dir, pre_cfg, feat_cfg, raw_dir)) as ex:
            it = ex.map(_process_one, files['file'], chunksize=8)
            for i, (f, xn, xp, err) in enumerate(it, start=1):
                (gagal.__setitem__(f, err) if err else hasil.__setitem__(f, (xn, xp)))
                if i % 100 == 0 or i == n:
                    print(f'  {i}/{n} file | {(time.time() - t0) / 60:.1f} menit', flush=True)
    else:
        for i, f in enumerate(files['file'], start=1):
            f, xn, xp, err = _process_one(f)
            (gagal.__setitem__(f, err) if err else hasil.__setitem__(f, (xn, xp)))
            if i % 100 == 0 or i == n:
                print(f'  {i}/{n} file | {(time.time() - t0) / 60:.1f} menit', flush=True)

    os.makedirs(out_dir, exist_ok=True)
    ok = files[files['file'].isin(hasil)].reset_index(drop=True)
    paths = {}
    for k, versi in enumerate(['none', 'peak']):
        X = pd.DataFrame([hasil[f][k] for f in ok['file']], columns=ft.FEATURE_NAMES)
        paths[versi] = os.path.join(out_dir, f'features_{versi}.csv')
        pd.concat([ok, X], axis=1).to_csv(paths[versi], index=False)
    pd.DataFrame([{'index': i, 'name': nm, 'group': g, 'description': d, 'scale_dependent': s}
                  for i, (nm, g, d, s) in enumerate(ft.FEATURE_TABLE)]).to_csv(
        os.path.join(out_dir, 'feature_names.csv'), index=False)

    import librosa
    import soundfile
    meta = {'config_hash': feat_hash, 'preprocessing_hash': pre_hash,
            'created': datetime.now().strftime('%Y-%m-%d %H:%M'), 'n_files': len(ok), 'n_failed': len(gagal),
            'actors': args.actors or 'semua', 'script_version': SCRIPT_VERSION,
            'library': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                        'librosa': librosa.__version__, 'soundfile': soundfile.__version__},
            'failed': gagal}
    with open(os.path.join(out_dir, 'features_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f'Tersimpan di {out_dir}: features_none.csv, features_peak.csv, feature_names.csv, features_meta.json')
    print(f'Berhasil {len(ok)} file, gagal {len(gagal)} file, waktu {(time.time() - t0) / 60:.1f} menit')
    for f, e in list(gagal.items())[:10]:
        print(f'  GAGAL {f}: {e}')

    kode = 1 if gagal else 0
    if args.compare:
        ref_dir = os.path.abspath(args.compare)
        laporan = {'rtol': args.rtol, 'atol': args.atol}
        for versi in ['none', 'peak']:
            ref = os.path.join(ref_dir, f'features_{versi}.csv')
            if not os.path.exists(ref):
                print(f'[ERROR] pembanding tidak ditemukan: {ref}')
                return 2
            r = compare_tables(paths[versi], ref, args.rtol, args.atol)
            laporan[versi] = r
            print(f"Perbandingan {versi}: {'IDENTIK' if r['identik'] else 'BERBEDA'} | "
                  f"{r['sel_identik_persis']}/{r['jumlah_sel']} sel sama persis | "
                  f"selisih abs maks {r['selisih_abs_maks']:.3g} | rel maks {r['selisih_rel_maks']:.3g}")
            if not r['identik']:
                kode = 1
        ref_names = os.path.join(ref_dir, 'feature_names.csv')
        if os.path.exists(ref_names):
            same = pd.read_csv(ref_names).equals(pd.read_csv(os.path.join(out_dir, 'feature_names.csv')))
            laporan['feature_names_sama'] = bool(same)
            print(f"feature_names.csv: {'sama' if same else 'BERBEDA'}")
            if not same:
                kode = 1
        with open(os.path.join(out_dir, 'compare_report.json'), 'w', encoding='utf-8') as f:
            json.dump(laporan, f, indent=2)
        print('Laporan perbandingan:', os.path.join(out_dir, 'compare_report.json'))
    return kode


if __name__ == '__main__':
    sys.exit(main())
