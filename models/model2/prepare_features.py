#!/usr/bin/env python3
"""
prepare_features.py — Extraction des features MFCC et split train/val/test
============================================================================

Fonctionnalités du script :
---------------------------
  1. Détecte les fichiers audio (Bee, NoBee, Missing Queen) sous `./data/`.
  2. Charge chaque clip audio en canal mono (`librosa.load(..., mono=True)`)
     avec une longueur fixe pour garantir des dimensions homogènes.
  3. Extrait les coefficients MFCC en parallèle (multiprocessing).
  4. Effectue le découpage de la classe 'Bee' :
     - Train : 80%
     - Validation : 10%
     - Test : 10%
     Les classes 'NoBee' et 'Missing Queen' sont réservées exclusivement
     au jeu de test pour l'évaluation.
  5. Normalise (z-score) toutes les caractéristiques en utilisant uniquement les
     statistiques calculées sur le jeu d'entraînement (Bee-train).
  6. Sauvegarde les matrices de features (`.npy`), les paramètres de
     normalisation et un fichier `manifest.json` dans `./processed_data/`.

Usage :
    python prepare_features.py
    python prepare_features.py --n-mfcc 20 --n-jobs 8
    python prepare_features.py --limit 200   # test rapide sur un sous-ensemble
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import librosa
except ImportError:
    print("ERREUR : librosa n'est pas installé. Faites `pip install librosa`.")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # repli minimal si tqdm absent
        total = kwargs.get("total")
        desc = kwargs.get("desc", "")
        for i, item in enumerate(iterable, 1):
            if total and (i % max(1, total // 20) == 0 or i == total):
                print(f"  {desc} : {i}/{total}", flush=True)
            yield item

try:
    from explore_data import discover_dataset_structure
except ImportError:
    print(
        "ERREUR : impossible d'importer `discover_dataset_structure` depuis "
        "explore_data.py. Assurez-vous que explore_data.py se trouve dans le "
        "même dossier que prepare_features.py."
    )
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_features")


# ---------------------------------------------------------------------------
# Extraction MFCC
# ---------------------------------------------------------------------------

def _extract_worker(path_str: str, sr: int, n_samples: int, n_mfcc: int,
                     n_fft: int, hop_length: int) -> Optional[np.ndarray]:
    """Charge un fichier audio (mono forcé, longueur fixe) et calcule ses
    MFCC. Renvoie None si le fichier est corrompu/illisible. Fonction
    top-level requise pour être picklable par ProcessPoolExecutor."""
    try:
        y, _ = librosa.load(path_str, sr=sr, mono=True)
        if y.size == 0:
            return None
        y = librosa.util.fix_length(y, size=n_samples)
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
        )
        return mfcc.astype(np.float32)
    except Exception:
        return None


def extract_batch(
    paths: List[Path], sr: int, n_samples: int, n_mfcc: int, n_fft: int,
    hop_length: int, n_jobs: int, desc: str,
) -> Tuple[np.ndarray, List[Path], int]:
    """Extrait les MFCC pour une liste de fichiers, en parallèle si n_jobs>1.
    Renvoie (array empilé, liste des chemins valides dans le même ordre que
    l'array, nombre de fichiers ignorés car corrompus/illisibles)."""
    if not paths:
        return np.zeros((0, n_mfcc, 0), dtype=np.float32), [], 0

    order = {p: i for i, p in enumerate(paths)}
    results: List[Tuple[Path, np.ndarray]] = []

    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(_extract_worker, str(p), sr, n_samples,
                                 n_mfcc, n_fft, hop_length): p
                for p in paths
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
                p = futures[fut]
                try:
                    mfcc = fut.result()
                except Exception:
                    mfcc = None
                if mfcc is not None:
                    results.append((p, mfcc))
    else:
        for p in tqdm(paths, total=len(paths), desc=desc):
            mfcc = _extract_worker(str(p), sr, n_samples, n_mfcc, n_fft, hop_length)
            if mfcc is not None:
                results.append((p, mfcc))

    
    results.sort(key=lambda item: order[item[0]])

    n_failed = len(paths) - len(results)
    if n_failed:
        logger.warning(f"[{desc}] {n_failed} fichier(s) ignoré(s) (corrompu/illisible).")

    if not results:
        return np.zeros((0, n_mfcc, 0), dtype=np.float32), [], n_failed

    # Vérification de cohérence des formes
    shapes = {m.shape for _, m in results}
    if len(shapes) > 1:
        target_shape = max(shapes, key=lambda s: sum(
            1 for _, m in results if m.shape == s))
        logger.warning(
            f"[{desc}] Formes MFCC hétérogènes détectées {shapes}. "
            f"Conservation uniquement de la forme majoritaire {target_shape}."
        )
        results = [(p, m) for p, m in results if m.shape == target_shape]
        n_failed = len(paths) - len(results)

    valid_paths = [p for p, _ in results]
    array = np.stack([m for _, m in results]).astype(np.float32)
    return array, valid_paths, n_failed


# ---------------------------------------------------------------------------
# Split train/val/test (classe Bee uniquement)
# ---------------------------------------------------------------------------

def split_indices(n: int, train_ratio: float, val_ratio: float,
                   seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_val = min(n_val, n - n_train)  # sécurité anti dépassement par arrondi
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def compute_normalization_stats(train_array: np.ndarray, eps: float = 1e-8
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Calcule moyenne/écart-type par coefficient MFCC (axe 1), sur l'ensemble
    des échantillons et des frames temporelles du train Bee UNIQUEMENT."""
    mean = train_array.mean(axis=(0, 2))          # shape (n_mfcc,)
    std = train_array.std(axis=(0, 2)) + eps       # shape (n_mfcc,)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_normalization(array: np.ndarray, mean: np.ndarray, std: np.ndarray
                         ) -> np.ndarray:
    if array.size == 0:
        return array
    return (array - mean[None, :, None]) / std[None, :, None]


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extraction MFCC + split train/val/test (Modèle 2)."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./processed_data"))
    parser.add_argument("--sr", type=int, default=44100,
                         help="Sample rate cible (Hz). Les clips sont "
                              "resamplés si nécessaire (défaut : 44100).")
    parser.add_argument("--duration", type=float, default=2.0,
                         help="Durée cible en secondes ; les clips sont "
                              "tronqués/complétés par zéro-padding pour "
                              "garantir une longueur fixe (défaut : 2.0).")
    parser.add_argument("--n-mfcc", type=int, default=20)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=None,
                         help="Nombre de process pour l'extraction parallèle "
                              "(défaut : nb de coeurs CPU - 1).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Limite le nombre de fichiers traités PAR "
                              "CLASSE (utile pour tester rapidement le "
                              "pipeline avant de lancer sur les 13 792 "
                              "fichiers complets).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()

    if abs((args.train_ratio + args.val_ratio) - 1.0) < 1e-9:
        test_ratio_display = 0.0
    else:
        test_ratio_display = 1.0 - args.train_ratio - args.val_ratio
    if not (0 < args.train_ratio < 1) or not (0 <= args.val_ratio < 1) or test_ratio_display <= 0:
        logger.error("train-ratio/val-ratio invalides (le test-ratio implicite "
                      "doit être > 0). Reçu : train=%.2f val=%.2f -> test=%.2f",
                      args.train_ratio, args.val_ratio, test_ratio_display)
        return 1

    n_jobs = args.n_jobs
    if n_jobs is None:
        import os
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    n_samples = int(round(args.sr * args.duration))
    logger.info(f"Config : sr={args.sr} Hz | durée={args.duration}s "
                f"({n_samples} échantillons) | n_mfcc={args.n_mfcc} | "
                f"n_fft={args.n_fft} | hop_length={args.hop_length} | "
                f"n_jobs={n_jobs}")

    class_files = discover_dataset_structure(args.data_dir)
    if not class_files:
        logger.error("Aucune classe détectée sous %s. Abandon.", args.data_dir)
        return 1

    if args.limit:
        logger.warning(f"Mode --limit={args.limit} actif : seul un "
                        f"sous-ensemble de fichiers sera traité par classe.")
        class_files = {
            c: (files if len(files) <= args.limit
                else list(np.random.default_rng(args.seed).choice(
                    files, size=args.limit, replace=False)))
            for c, files in class_files.items()
        }

    extract_kwargs = dict(sr=args.sr, n_samples=n_samples, n_mfcc=args.n_mfcc,
                           n_fft=args.n_fft, hop_length=args.hop_length,
                           n_jobs=n_jobs)

    # --- Bee : extraction complète puis split ---------------------------------
    bee_files = class_files.get("Bee", [])
    logger.info(f"Extraction MFCC — classe 'Bee' ({len(bee_files)} fichiers)...")
    bee_array, bee_paths, bee_failed = extract_batch(
        bee_files, desc="Bee", **extract_kwargs
    )
    if bee_array.shape[0] == 0:
        logger.error("Aucun fichier 'Bee' n'a pu être traité. Abandon.")
        return 1

    train_idx, val_idx, test_idx = split_indices(
        bee_array.shape[0], args.train_ratio, args.val_ratio, args.seed
    )
    bee_train, bee_val, bee_test = bee_array[train_idx], bee_array[val_idx], bee_array[test_idx]
    paths_train = [str(bee_paths[i]) for i in train_idx]
    paths_val = [str(bee_paths[i]) for i in val_idx]
    paths_test = [str(bee_paths[i]) for i in test_idx]

    logger.info(f"Split Bee -> train={bee_train.shape[0]} | "
                f"val={bee_val.shape[0]} | test={bee_test.shape[0]}")

    # --- NoBee / Missing Queen : intégralement réservés à l'évaluation --------
    eval_arrays: Dict[str, Tuple[np.ndarray, List[str], int]] = {}
    for class_name in ("NoBee", "Missing Queen"):
        files = class_files.get(class_name, [])
        logger.info(f"Extraction MFCC — classe '{class_name}' ({len(files)} fichiers)...")
        arr, paths, failed = extract_batch(files, desc=class_name, **extract_kwargs)
        eval_arrays[class_name] = (arr, [str(p) for p in paths], failed)

    # --- Normalisation (stats calculées UNIQUEMENT sur Bee-train) -------------
    mean, std = compute_normalization_stats(bee_train)
    logger.info("Statistiques de normalisation calculées sur Bee-train uniquement "
                "(pas de fuite de données vers val/test/anomalies).")

    bee_train_n = apply_normalization(bee_train, mean, std)
    bee_val_n = apply_normalization(bee_val, mean, std)
    bee_test_n = apply_normalization(bee_test, mean, std)
    eval_arrays_n = {
        name: apply_normalization(arr, mean, std)
        for name, (arr, _, _) in eval_arrays.items()
    }

    # --- Sauvegarde -------------------------------------------------------------
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_bee_train.npy", bee_train_n)
    np.save(out_dir / "X_bee_val.npy", bee_val_n)
    np.save(out_dir / "X_bee_test.npy", bee_test_n)
    np.save(out_dir / "X_nobee.npy", eval_arrays_n["NoBee"])
    np.save(out_dir / "X_missing_queen.npy", eval_arrays_n["Missing Queen"])
    np.save(out_dir / "mfcc_mean.npy", mean)
    np.save(out_dir / "mfcc_std.npy", std)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "params": {
            "sample_rate": args.sr,
            "duration_s": args.duration,
            "n_samples": n_samples,
            "n_mfcc": args.n_mfcc,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
        },
        "shapes": {
            "X_bee_train": list(bee_train_n.shape),
            "X_bee_val": list(bee_val_n.shape),
            "X_bee_test": list(bee_test_n.shape),
            "X_nobee": list(eval_arrays_n["NoBee"].shape),
            "X_missing_queen": list(eval_arrays_n["Missing Queen"].shape),
        },
        "n_corrupted": {
            "Bee": bee_failed,
            "NoBee": eval_arrays["NoBee"][2],
            "Missing Queen": eval_arrays["Missing Queen"][2],
        },
        "normalization": {
            "method": "z-score par coefficient MFCC, stats calculées sur Bee-train",
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "file_lists": {
            "bee_train": paths_train,
            "bee_val": paths_val,
            "bee_test": paths_test,
            "nobee": eval_arrays["NoBee"][1],
            "missing_queen": eval_arrays["Missing Queen"][1],
        },
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    logger.info(f"Terminé en {elapsed:.1f}s. Fichiers sauvegardés dans {out_dir.resolve()}")

    print("\n" + "=" * 72)
    print(" RÉSUMÉ — Préparation des features MFCC")
    print("=" * 72)
    print(f"  Forme des matrices MFCC : (n_mfcc={args.n_mfcc}, "
          f"n_frames={bee_train_n.shape[-1]})")
    print(f"  Bee  : train={bee_train_n.shape[0]:>5} | val={bee_val_n.shape[0]:>5} "
          f"| test={bee_test_n.shape[0]:>5}  ({bee_failed} corrompu(s))")
    print(f"  NoBee (éval. uniquement)          : {eval_arrays_n['NoBee'].shape[0]:>5}  "
          f"({eval_arrays['NoBee'][2]} corrompu(s))")
    print(f"  Missing Queen (éval. uniquement)  : {eval_arrays_n['Missing Queen'].shape[0]:>5}  "
          f"({eval_arrays['Missing Queen'][2]} corrompu(s))")
    print(f"  Fichiers écrits dans : {out_dir.resolve()}")
    print("=" * 72 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())