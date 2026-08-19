#!/usr/bin/env python3
"""
evaluate_all.py — Évaluation comparative des 3 architectures (HiveSense)
============================================================================

Méthodologie de seuillage :
---------------------------
Le jeu de validation (`X_bee_val.npy`) contenant uniquement des sons normaux (Bee),
ce script évalue les modèles selon deux métriques de seuil :

  1. Seuil "percentile" (Principal, non supervisé) :
     Calculé sur les erreurs de reconstruction du jeu de validation (Bee-VAL).
     Le 80e percentile est appliqué par défaut.

  2. Seuil "F1-optimal" (Secondaire, indicatif) :
     Déterminé par balayage direct sur le jeu de test pour évaluer la limite
     théorique des performances.

Fonctionnalités du script :
---------------------------
  1. Charge les 3 modèles entraînés (configurations et poids) depuis `./saved_models/`.
  2. Charge les données de test (Bee-test, NoBee/Missing Queen) et de validation (Bee-val) depuis `./processed_data/`.
  3. Calcule l'erreur de reconstruction MSE par échantillon pour chaque modèle.
  4. Calcule les métriques globales (ROC-AUC, PR-AUC, Précision, Rappel, F1) ainsi que le rappel par type d'anomalie.
  5. Calcule les scores de confiance (%) via `compute_confidence_scores()`.
  6. Sauvegarde les erreurs du modèle gagnant dans `processed_data/val_errors_bee.npy`.
  7. Génère les graphiques comparatifs (`roc_curves_comparison.png`, `pr_curves_comparison.png`, `reconstruction_errors_dist.png`).
  8. Affiche un tableau comparatif final et désigne le modèle le plus performant (basé sur le ROC-AUC).

Usage :
    python evaluate_all.py
    python evaluate_all.py --percentile 80 --batch-size 256
"""


from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("ERREUR : PyTorch n'est pas installé. Faites `pip install torch`.")
    sys.exit(1)

try:
    from sklearn.metrics import (precision_recall_curve, precision_score,
                                  recall_score, f1_score, roc_auc_score,
                                  roc_curve, average_precision_score)
except ImportError:
    print("ERREUR : scikit-learn n'est pas installé. Faites `pip install scikit-learn`.")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


try:
    from train_conv2d_autoencoder import Conv2DAutoencoder
    from train_vae import Conv2DVAE
    from train_contrastive_autoencoder import ContrastiveConv2DAutoencoder
except ImportError as e:
    print(
        f"ERREUR : impossible d'importer les architectures des scripts "
        f"d'entraînement ({e}). Assurez-vous que train_conv2d_autoencoder.py, "
        f"train_vae.py et train_model3.py se trouvent dans le même dossier "
        f"que evaluate_all.py."
    )
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_all")

MODEL_ORDER = ["Conv2D Autoencoder", "Conv2D VAE", "Contrastive AE"]
MODEL_COLORS = {
    "Conv2D Autoencoder": "#1f77b4",
    "Conv2D VAE": "#ff7f0e",
    "Contrastive AE": "#2ca02c",
}
CLASS_COLORS = {"Bee": "#2ca02c", "NoBee": "#d62728", "Missing Queen": "#9467bd"}
SWEEP_PERCENTILES = (50.0, 80.0, 95.0)  # Tableau 4 : compromis rappel/précision


# ---------------------------------------------------------------------------
# Chargement des modèles (INCHANGÉ)
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path, device: torch.device) -> Optional[dict]:
    if not path.exists():
        logger.warning(f"Fichier introuvable, modèle ignoré : {path}")
        return None
    return torch.load(path, map_location=device, weights_only=False)


def load_conv2d_ae(path: Path, device: torch.device):
    ckpt = load_checkpoint(path, device)
    if ckpt is None:
        return None, None
    cfg = ckpt["config"]
    model = Conv2DAutoencoder(in_channels=1, base_channels=cfg["base_channels"],
                               n_conv_blocks=cfg["n_conv_blocks"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


def load_vae(path: Path, device: torch.device):
    ckpt = load_checkpoint(path, device)
    if ckpt is None:
        return None, None
    cfg = ckpt["config"]
    model = Conv2DVAE(n_mfcc=cfg["n_mfcc"], n_frames=cfg["n_frames"], in_channels=1,
                       base_channels=cfg["base_channels"], n_conv_blocks=cfg["n_conv_blocks"],
                       latent_dim=cfg["latent_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


def load_contrastive_ae(path: Path, device: torch.device):
    ckpt = load_checkpoint(path, device)
    if ckpt is None:
        return None, None
    cfg = ckpt["config"]
    model = ContrastiveConv2DAutoencoder(in_channels=1, base_channels=cfg["base_channels"],
                                          n_conv_blocks=cfg["n_conv_blocks"],
                                          projection_dim=cfg["projection_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# Calcul de l'erreur de reconstruction (INCHANGÉ)
# ---------------------------------------------------------------------------

def compute_reconstruction_errors(model, architecture: str, X: np.ndarray,
                                   device: torch.device, batch_size: int
                                   ) -> Tuple[np.ndarray, float]:
    """Renvoie (erreurs MSE par échantillon, temps d'inférence total en s).
    Pour le VAE, on utilise mu (moyenne) plutôt qu'un tirage aléatoire de z,
    afin d'obtenir un score de reconstruction déterministe et reproductible
    — pratique standard pour la détection d'anomalies avec un VAE."""
    if X.shape[0] == 0:
        return np.array([]), 0.0

    tensor = torch.from_numpy(X).unsqueeze(1).float()
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)

    errors = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            if architecture == "conv2d_vae":
                mu, _ = model.encode(batch)
                recon = model.decode(mu)
            else:
                recon, _ = model(batch)  # AE simple et Contrastive AE : même interface
            batch_errors = ((recon - batch) ** 2).mean(dim=[1, 2, 3])
            errors.append(batch_errors.cpu().numpy())

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    return np.concatenate(errors), elapsed


# ---------------------------------------------------------------------------
# Seuils et métriques (INCHANGÉ + nouvelles fonctions)
# ---------------------------------------------------------------------------

def percentile_threshold(errors_normal_val: np.ndarray, percentile: float) -> float:
    return float(np.percentile(errors_normal_val, percentile))


def f1_optimal_threshold(errors: np.ndarray, labels: np.ndarray) -> float:
    """Balaie tous les seuils atteignables (via precision_recall_curve) et
    renvoie celui qui maximise le F1. ATTENTION : calibré sur les mêmes
    données que celles utilisées pour le F1 final -> estimation optimiste."""
    precisions, recalls, thresholds = precision_recall_curve(labels, errors)
    denom = precisions[:-1] + recalls[:-1]
    f1s = np.where(denom > 0, 2 * precisions[:-1] * recalls[:-1] / np.maximum(denom, 1e-12), 0.0)
    if len(f1s) == 0:
        return float(np.median(errors))
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx])


def compute_metrics_at_threshold(errors: np.ndarray, labels: np.ndarray,
                                  threshold: float) -> Dict[str, float]:
    preds = (errors >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def recall_at_threshold(errors: np.ndarray, threshold: float) -> float:
    """Rappel pour une sous-classe d'anomalie prise isolément (tous les
    échantillons de cette sous-classe sont, par construction, des positifs)."""
    if errors.shape[0] == 0:
        return float("nan")
    return float((errors >= threshold).mean())


def build_percentile_sweep(errors_val: np.ndarray, errors_test: np.ndarray,
                            labels_test: np.ndarray,
                            percentiles=SWEEP_PERCENTILES) -> List[Dict[str, float]]:
    """Tableau 4 : pour chaque percentile de Bee-val, calcule le seuil
    correspondant et les métriques sur le jeu de test -> permet de visualiser
    le compromis rappel/précision plutôt que de le figer dans un seul défaut."""
    rows = []
    for p in percentiles:
        thr = percentile_threshold(errors_val, p)
        metrics = compute_metrics_at_threshold(errors_test, labels_test, thr)
        rows.append({"percentile": p, **metrics})
    return rows


def compute_confidence_scores(raw_mse: float, threshold: float = 0.04545,
                               scale: Optional[float] = None) -> Dict[str, float]:
    """Convertit une erreur de reconstruction brute (MSE) en scores de
    confiance interprétables (%), via une sigmoïde centrée sur le seuil de
    décision. Pensée pour être réutilisée telle quelle dans l'endpoint
    FastAPI /predict/anomaly (Étape 5) : {"class": ..., "confidence": ...}.

    ATTENTION : le défaut threshold=0.04545 correspond
    au seuil calibré pour le Contrastive AE (percentile 80, via
    find_best_percentile.py). Les 3 architectures ont des échelles d'erreur
    de reconstruction très différentes (AE ~0.01, VAE ~0.5-1.0, Contrastive
    AE ~0.04-0.9) : appeler cette fonction avec le défaut pour un AE ou un
    VAE donnerait des scores de confiance non calibrés, donc trompeurs. Dans
    ce script, `threshold` (et `scale`) sont TOUJOURS passés explicitement
    avec les valeurs propres à chaque modèle — le défaut n'est là que pour
    un usage autonome rapide (tests, prototypage) avec le Contrastive AE.

    `scale` contrôle la "douceur" de la transition autour du seuil. Idéalement
    dérivé empiriquement (écart-type des erreurs Bee-val du modèle concerné) ;
    à défaut, repli arbitraire à 25% du seuil.
    """
    if scale is None or scale <= 0:
        scale = threshold * 0.25
    z = (raw_mse - threshold) / scale
    confidence_anomaly = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    confidence_normal = 1.0 - confidence_anomaly
    return {
        "confidence_normal": round(float(confidence_normal) * 100, 2),
        "confidence_anomaly": round(float(confidence_anomaly) * 100, 2),
    }


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_roc_curves(results: Dict[str, dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for name in MODEL_ORDER:
        if name not in results:
            continue
        r = results[name]
        ax.plot(r["fpr"], r["tpr"], color=MODEL_COLORS[name], linewidth=2.2,
                label=f"{name} (AUC = {r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="Hasard (AUC = 0.5)")
    ax.set_xlabel("Taux de faux positifs (1 - Spécificité)")
    ax.set_ylabel("Taux de vrais positifs (Rappel)")
    ax.set_title("Comparaison des courbes ROC — HiveSense (Modèle 2)",
                  fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Figure sauvegardée : {output_path}")


def plot_pr_curves(results: Dict[str, dict], output_path: Path) -> None:
    """Courbes Precision-Recall superposées. Contrairement à la ROC, le
    niveau "hasard" d'une courbe PR dépend de la prévalence des positifs
    (pas 0.5 fixe) -> on trace cette ligne de base explicitement."""
    fig, ax = plt.subplots(figsize=(7.5, 7))
    prevalence = None
    for name in MODEL_ORDER:
        if name not in results:
            continue
        r = results[name]
        precision, recall, _ = precision_recall_curve(r["labels_test"], r["errors_test"])
        ax.plot(recall, precision, color=MODEL_COLORS[name], linewidth=2.2,
                label=f"{name} (PR-AUC = {r['pr_auc']:.3f})")
        if prevalence is None:
            prevalence = float(r["labels_test"].mean())

    if prevalence is not None:
        ax.axhline(prevalence, linestyle="--", color="grey", linewidth=1,
                    label=f"Hasard (prévalence = {prevalence:.3f})")

    ax.set_xlabel("Rappel")
    ax.set_ylabel("Précision")
    ax.set_title("Comparaison des courbes Precision-Recall — HiveSense (Modèle 2)",
                  fontsize=13, fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Figure sauvegardée : {output_path}")


def plot_error_distributions(results: Dict[str, dict], output_path: Path) -> None:
    available = [n for n in MODEL_ORDER if n in results]
    fig, axes = plt.subplots(1, len(available), figsize=(6.5 * len(available), 5.5),
                              sharey=False)
    if len(available) == 1:
        axes = [axes]

    for ax, name in zip(axes, available):
        r = results[name]
        for class_name in ("Bee", "NoBee", "Missing Queen"):
            data = r["errors_by_class"].get(class_name, np.array([]))
            if data.size == 0:
                continue
            ax.hist(data, bins=50, density=True, alpha=0.5,
                    color=CLASS_COLORS[class_name], label=f"{class_name} (n={data.size})")
        ax.axvline(r["threshold_percentile"], color="black", linestyle="--", linewidth=1.5,
                   label=f"Seuil percentile ({r['threshold_percentile']:.4f})")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Erreur de reconstruction (MSE)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Densité")

    fig.suptitle("Distribution des erreurs de reconstruction par classe",
                  fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure sauvegardée : {output_path}")


# ---------------------------------------------------------------------------
# Tableau récapitulatif
# ---------------------------------------------------------------------------

def print_summary(results: Dict[str, dict], winner_name: str, default_percentile: float) -> None:
    available = [n for n in MODEL_ORDER if n in results]
    if not available:
        print("Aucun modèle n'a pu être évalué.")
        return

    line = "=" * 108
    print("\n" + line)
    print(" ÉVALUATION COMPARATIVE — HiveSense — Modèle 2 (détection d'anomalies acoustiques)")
    print(line)

    print(f"\n--- Tableau 1 : Métriques PRINCIPALES (seuil = {default_percentile:.0f}e percentile "
          f"de Bee-val, sans fuite de données) ---\n")
    header = (f"{'Modèle':<22}{'ROC-AUC':>9}{'PR-AUC':>9}{'Seuil':>12}{'Précision':>12}"
              f"{'Rappel':>10}{'F1-Score':>10}{'Inférence':>14}{'Taille':>10}")
    print(header)
    print("-" * len(header))
    for name in available:
        r = results[name]
        m = r["metrics_percentile"]
        print(f"{name:<22}{r['auc']:>9.4f}{r['pr_auc']:>9.4f}{m['threshold']:>12.5f}"
              f"{m['precision']:>12.3f}{m['recall']:>10.3f}{m['f1']:>10.3f}"
              f"{r['inference_ms_per_sample']:>11.3f} ms/éch{r['checkpoint_size_kb']:>8.0f} Ko")

    print("\n--- Tableau 2 : Métriques SECONDAIRES (seuil F1-optimal calibré sur le jeu de "
          "test — estimation OPTIMISTE, à visée indicative uniquement) ---\n")
    header2 = f"{'Modèle':<22}{'Seuil':>12}{'Précision':>12}{'Rappel':>10}{'F1-Score':>10}"
    print(header2)
    print("-" * len(header2))
    for name in available:
        r = results[name]
        m = r["metrics_f1opt"]
        print(f"{name:<22}{m['threshold']:>12.5f}{m['precision']:>12.3f}"
              f"{m['recall']:>10.3f}{m['f1']:>10.3f}")

    print("\n--- Tableau 3 : Rappel par type d'anomalie (seuil percentile, Tableau 1) ---\n")
    header3 = f"{'Modèle':<22}{'Rappel NoBee':>15}{'Rappel Missing Queen':>22}"
    print(header3)
    print("-" * len(header3))
    for name in available:
        r = results[name]
        print(f"{name:<22}{r['recall_nobee']:>15.3f}{r['recall_missing_queen']:>22.3f}")

    print(f"\n--- Tableau 4 : Compromis Rappel/Précision selon le seuil "
          f"({'/'.join(f'{p:.0f}e' for p in SWEEP_PERCENTILES)} percentile de Bee-val) ---")
    print("    Note : choisir un percentile en observant sa performance sur le jeu de test\n"
          "    (comme fait ici pour arrêter le défaut à 80) réintroduit le même biais optimiste\n"
          "    que le Tableau 2 — ce tableau sert à le rendre visible, pas à le supprimer.\n")
    header4 = f"{'Modèle':<22}{'Percentile':>12}{'Seuil':>12}{'Précision':>12}{'Rappel':>10}{'F1-Score':>10}"
    print(header4)
    print("-" * len(header4))
    for name in available:
        for row in results[name]["percentile_sweep"]:
            marker = "  <- défaut actuel" if abs(row["percentile"] - default_percentile) < 1e-6 else ""
            print(f"{name:<22}{row['percentile']:>11.0f}e{row['threshold']:>12.5f}"
                  f"{row['precision']:>12.3f}{row['recall']:>10.3f}{row['f1']:>10.3f}{marker}")
        print()

    r_winner = results[winner_name]
    scale = float(np.std(r_winner["errors_bee_val"])) if r_winner["errors_bee_val"].size > 1 else None
    print(f"--- Exemple de scores de confiance ({winner_name}, seuil percentile="
          f"{r_winner['threshold_percentile']:.5f}) ---\n")
    demo_points = {
        "Bee (médiane)": r_winner["errors_by_class"].get("Bee", np.array([])),
        "NoBee (médiane)": r_winner["errors_by_class"].get("NoBee", np.array([])),
        "Missing Queen (médiane)": r_winner["errors_by_class"].get("Missing Queen", np.array([])),
    }
    for label, arr in demo_points.items():
        if arr.size == 0:
            continue
        mse_val = float(np.median(arr))
        conf = compute_confidence_scores(mse_val, threshold=r_winner["threshold_percentile"], scale=scale)
        print(f"  {label:<26} MSE={mse_val:.5f}  ->  confidence_normal={conf['confidence_normal']}%"
              f"  |  confidence_anomaly={conf['confidence_anomaly']}%")

    print("\n" + line)
    print(f" MODÈLE GAGNANT (critère : ROC-AUC — seule métrique indépendante du seuil) : "
          f"{winner_name}  (AUC = {results[winner_name]['auc']:.4f}, "
          f"PR-AUC = {results[winner_name]['pr_auc']:.4f})")
    f1_winner = max(available, key=lambda n: results[n]["metrics_percentile"]["f1"])
    if f1_winner != winner_name:
        print(f" (Sous le critère F1 au seuil percentile, {f1_winner} serait en tête à la place "
              f"— à noter dans votre analyse.)")
    print(line + "\n")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Évaluation comparative des 3 architectures (Modèle 2 - HiveSense)."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./processed_data"))
    parser.add_argument("--model-dir", type=Path, default=Path("./saved_models"))
    parser.add_argument("--output-dir", type=Path, default=Path("./output"))
    parser.add_argument("--percentile", type=float, default=80.0,
                         help="Percentile de Bee-val utilisé comme seuil principal "
                              "(défaut : 80, calibré via find_best_percentile.py — "
                              "voir Tableau 4 pour le compromis complet 50/80/95).")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device utilisé pour l'inférence : {device}")

    # --- Chargement des données -------------------------------------------------
    required = ["X_bee_val.npy", "X_bee_test.npy", "X_nobee.npy", "X_missing_queen.npy"]
    for fname in required:
        if not (args.data_dir / fname).exists():
            logger.error(f"Fichier manquant : {args.data_dir / fname}. Avez-vous lancé "
                          f"prepare_features.py ?")
            return 1

    X_bee_val = np.load(args.data_dir / "X_bee_val.npy")
    X_bee_test = np.load(args.data_dir / "X_bee_test.npy")
    X_nobee = np.load(args.data_dir / "X_nobee.npy")
    X_missing_queen = np.load(args.data_dir / "X_missing_queen.npy")

    logger.info(f"Jeux chargés : Bee-val={len(X_bee_val)} | Bee-test={len(X_bee_test)} | "
                f"NoBee={len(X_nobee)} | Missing Queen={len(X_missing_queen)}")

    labels_test = np.concatenate([
        np.zeros(len(X_bee_test), dtype=int),
        np.ones(len(X_nobee), dtype=int),
        np.ones(len(X_missing_queen), dtype=int),
    ])

    # --- Chargement des 3 modèles (logique inchangée) ---------------------------
    model_specs = [
        ("Conv2D Autoencoder", args.model_dir / "conv2d_autoencoder.pt",
         load_conv2d_ae, "conv2d_ae"),
        ("Conv2D VAE", args.model_dir / "vae_autoencoder.pt", load_vae, "conv2d_vae"),
        ("Contrastive AE", args.model_dir / "contrastive_autoencoder.pt",
         load_contrastive_ae, "contrastive_ae"),
    ]

    results: Dict[str, dict] = {}

    for name, path, loader_fn, arch_tag in model_specs:
        model, ckpt = loader_fn(path, device)
        if model is None:
            continue
        logger.info(f"[{name}] Modèle chargé depuis {path} "
                    f"(meilleure époque : {ckpt.get('best_epoch', '?')}).")

        # Erreurs de reconstruction par classe
        errors_val, _ = compute_reconstruction_errors(model, arch_tag, X_bee_val, device, args.batch_size)
        errors_bee, t_bee = compute_reconstruction_errors(model, arch_tag, X_bee_test, device, args.batch_size)
        errors_nobee, t_nobee = compute_reconstruction_errors(model, arch_tag, X_nobee, device, args.batch_size)
        errors_mq, t_mq = compute_reconstruction_errors(model, arch_tag, X_missing_queen, device, args.batch_size)

        errors_test = np.concatenate([errors_bee, errors_nobee, errors_mq])
        total_time = t_bee + t_nobee + t_mq
        n_total = len(errors_test)

        # Seuils
        thr_percentile = percentile_threshold(errors_val, args.percentile)
        thr_f1opt = f1_optimal_threshold(errors_test, labels_test)

        # Métriques
        metrics_percentile = compute_metrics_at_threshold(errors_test, labels_test, thr_percentile)
        metrics_f1opt = compute_metrics_at_threshold(errors_test, labels_test, thr_f1opt)

        auc = roc_auc_score(labels_test, errors_test)
        pr_auc = average_precision_score(labels_test, errors_test)
        fpr, tpr, _ = roc_curve(labels_test, errors_test)

        percentile_sweep = build_percentile_sweep(errors_val, errors_test, labels_test)

        checkpoint_size_kb = os.path.getsize(path) / 1024.0

        results[name] = {
            "errors_by_class": {"Bee": errors_bee, "NoBee": errors_nobee, "Missing Queen": errors_mq},
            "errors_bee_val": errors_val,
            "errors_test": errors_test,
            "labels_test": labels_test,
            "auc": auc, "pr_auc": pr_auc, "fpr": fpr, "tpr": tpr,
            "threshold_percentile": thr_percentile,
            "metrics_percentile": metrics_percentile,
            "threshold_f1opt": thr_f1opt,
            "metrics_f1opt": metrics_f1opt,
            "percentile_sweep": percentile_sweep,
            "recall_nobee": recall_at_threshold(errors_nobee, thr_percentile),
            "recall_missing_queen": recall_at_threshold(errors_mq, thr_percentile),
            "inference_ms_per_sample": (total_time / n_total * 1000) if n_total else float("nan"),
            "checkpoint_size_kb": checkpoint_size_kb,
        }

    if not results:
        logger.error("Aucun modèle n'a pu être chargé/évalué. Abandon.")
        return 1

    # Modèle gagnant déterminé UNE SEULE FOIS ici (source de vérité unique,
    # jamais codé en dur) puis réutilisé pour la sauvegarde ET l'affichage.
    winner_name = max(results.keys(), key=lambda n: results[n]["auc"])

    val_errors_path = args.data_dir / "val_errors_bee.npy"
    np.save(val_errors_path, results[winner_name]["errors_bee_val"])
    logger.info(f"Erreurs de validation du modèle gagnant ({winner_name}, désigné "
                f"dynamiquement par l'AUC) sauvegardées : {val_errors_path}")

    plot_roc_curves(results, args.output_dir / "roc_curves_comparison.png")
    plot_pr_curves(results, args.output_dir / "pr_curves_comparison.png")
    plot_error_distributions(results, args.output_dir / "reconstruction_errors_dist.png")
    print_summary(results, winner_name, args.percentile)

    return 0


if __name__ == "__main__":
    sys.exit(main())