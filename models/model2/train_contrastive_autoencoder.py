#!/usr/bin/env python3
"""
train_contrastive_autoencoder.py — Entraînement du Contrastive Conv2D Autoencoder (PyTorch)
=============================================================================

Architecture 3 du  Modèle 2 (détection d'anomalies acoustiques).

Approche auto-supervisée :
--------------------------
Afin d'éviter toute fuite de données avec les classes d'anomalies réservées
exclusivement à l'évaluation (NoBee / Missing Queen), ce modèle utilise un
apprentissage contrastif auto-supervisé (inspiré de SimCLR) basé uniquement
sur la classe normale (Bee) :

  - Deux vues augmentées (SpecAugment : masquage temps/fréquence + bruit + gain)
    sont générées pour chaque clip audio.
  - Le décodeur reconstruit le clip d'origine à partir de chaque vue bruitée
    (Denoising Autoencoder).
  - Une perte contrastive (NT-Xent) rapproche dans l'espace latent les deux
    vues d'un même clip tout en éloignant les représentations des autres clips.

Caractéristiques de l'architecture :
-------------------------------------
Ce modèle est entièrement convolutif avec un padding dynamique. Il s'adapte
automatiquement aux dimensions des matrices MFCC sans reconfiguration des
couches.

Fonctionnalités du script :
---------------------------
  1. Charge `X_bee_train.npy` et `X_bee_val.npy` depuis `./processed_data/`.
  2. Définit le Contrastive Conv2D Autoencoder (encodeur partagé, tête de
     reconstruction et tête de projection contrastive).
  3. Calcule une perte combinée : MSE (reconstruction) + NT-Xent (contrastive).
  4. Exécute l'entraînement avec accélération CUDA/AMP (`GradScaler`),
     Early Stopping (patience=15) et ajustement automatique du learning rate
     (`ReduceLROnPlateau`).
  5. Sauvegarde le modèle (`saved_models/contrastive_autoencoder.pt`) ainsi que
     les courbes d'apprentissage (`output/loss_curve_contrastive.png`).

Usage :
    python train_contrastive_autoencoder.py
    python train_contrastive_autoencoder.py --contrastive-weight 0.1 --temperature 0.5
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("ERREUR : PyTorch n'est pas installé. Faites `pip install torch`.")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_contrastive_autoencoder")


# ---------------------------------------------------------------------------
# Architecture : Contrastive Conv2D Autoencoder
# ---------------------------------------------------------------------------

class ContrastiveConv2DAutoencoder(nn.Module):
    """Autoencodeur convolutif 2D + tête de projection contrastive.

    - encode()/decode() : identiques en esprit au Conv2DAutoencoder de
      l'Architecture 1 (padding dynamique, robuste à toute forme MFCC).
    - project() : global average pooling (indépendant de H'/W') + petit MLP
      -> embedding L2-normalisé utilisé pour la loss NT-Xent.
    - forward(x) : interface standard (reconstruction, features latentes),
      compatible avec le pipeline d'évaluation des Architectures 1 et 2 pour
      l'Étape 3 (score d'anomalie = erreur de reconstruction).
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 16,
                 n_conv_blocks: int = 3, projection_dim: int = 32):
        super().__init__()
        self.n_conv_blocks = n_conv_blocks
        self.downsample_factor = 2 ** n_conv_blocks

        c = base_channels
        enc_layers = []
        prev_c = in_channels
        for i in range(n_conv_blocks):
            out_c = c * (2 ** i)
            enc_layers += [
                nn.Conv2d(prev_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ]
            prev_c = out_c
        self.encoder = nn.Sequential(*enc_layers)
        self.bottleneck_channels = prev_c

        dec_layers = []
        for i in reversed(range(n_conv_blocks)):
            out_c = in_channels if i == 0 else c * (2 ** (i - 1))
            dec_layers.append(
                nn.ConvTranspose2d(prev_c, out_c, kernel_size=3, stride=2,
                                    padding=1, output_padding=1)
            )
            if i != 0:
                dec_layers += [nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
            prev_c = out_c
        self.decoder = nn.Sequential(*dec_layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Sequential(
            nn.Linear(self.bottleneck_channels, self.bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.bottleneck_channels, projection_dim),
        )

    def _pad_to_multiple(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        h, w = x.shape[-2], x.shape[-1]
        f = self.downsample_factor
        pad_h = (f - h % f) % f
        pad_w = (f - w % f) % f
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, h, w

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x_padded, h, w = self._pad_to_multiple(x)
        z = self.encoder(x_padded)
        return z, (h, w)

    def decode(self, z: torch.Tensor, orig_hw: Tuple[int, int]) -> torch.Tensor:
        h, w = orig_hw
        out = self.decoder(z)
        return out[:, :, :h, :w]

    def project(self, z: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(z).flatten(1)
        return F.normalize(self.projector(pooled), dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z, orig_hw = self.encode(x)
        recon = self.decode(z, orig_hw)
        return recon, z


# ---------------------------------------------------------------------------
# Augmentations (SpecAugment-style) 
# ---------------------------------------------------------------------------

def augment_mfcc(x: torch.Tensor, time_mask_frac: float = 0.15,
                  freq_mask_frac: float = 0.15, noise_std: float = 0.05,
                  gain_std: float = 0.05) -> torch.Tensor:
    """Applique un masquage temporel + fréquentiel (SpecAugment), du bruit
    gaussien additif et un jitter de gain à un batch de MFCC normalisées
    (B, 1, n_mfcc, n_frames). Renvoie une copie augmentée (n'altère pas x)."""
    x_aug = x.clone()
    B, _, H, W = x_aug.shape

    if time_mask_frac > 0 and W > 1:
        mask_w = max(1, int(W * time_mask_frac))
        for b in range(B):
            w0 = random.randint(0, max(0, W - mask_w))
            x_aug[b, :, :, w0:w0 + mask_w] = 0.0

    if freq_mask_frac > 0 and H > 1:
        mask_h = max(1, int(H * freq_mask_frac))
        for b in range(B):
            h0 = random.randint(0, max(0, H - mask_h))
            x_aug[b, :, h0:h0 + mask_h, :] = 0.0

    if noise_std > 0:
        x_aug = x_aug + torch.randn_like(x_aug) * noise_std

    if gain_std > 0:
        gain = 1.0 + torch.randn(B, 1, 1, 1, device=x_aug.device) * gain_std
        x_aug = x_aug * gain

    return x_aug


# ---------------------------------------------------------------------------
# Loss contrastive : NT-Xent (SimCLR)
# ---------------------------------------------------------------------------

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                  temperature: float = 0.5) -> torch.Tensor:
    """Normalized Temperature-scaled Cross Entropy loss.
    z1, z2 : (B, D) embeddings L2-normalisés des deux vues augmentées.
    Pour chaque vue, la paire positive est l'autre vue du même clip ; toutes
    les autres vues du batch (2B-2 au total) servent de négatifs."""
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                       # (2B, D)
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.t()) / temperature                # (2B, 2B)

    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))

    targets = torch.arange(2 * B, device=z.device)
    targets = (targets + B) % (2 * B)  # positif de i : i+B (mod 2B)

    return F.cross_entropy(sim, targets)


@dataclass
class ContrastiveConfig:
    base_channels: int
    n_conv_blocks: int
    projection_dim: int
    contrastive_weight: float
    temperature: float
    time_mask_frac: float
    freq_mask_frac: float
    noise_std: float
    gain_std: float
    batch_size: int
    lr: float
    weight_decay: float
    epochs: int
    patience: int
    seed: int


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------

def load_bee_datasets(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    train_path = data_dir / "X_bee_train.npy"
    val_path = data_dir / "X_bee_val.npy"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Fichiers introuvables dans {data_dir}. Avez-vous bien lancé "
            f"prepare_features.py au préalable ? (attendu : X_bee_train.npy, "
            f"X_bee_val.npy)"
        )
    return np.load(train_path), np.load(val_path)


def make_dataloader(X: np.ndarray, batch_size: int, shuffle: bool,
                     drop_last: bool, device: torch.device) -> DataLoader:
    tensor = torch.from_numpy(X).unsqueeze(1).float()
    dataset = TensorDataset(tensor)
    pin_memory = device.type == "cuda"
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                       drop_last=drop_last, pin_memory=pin_memory)


# ---------------------------------------------------------------------------
# Boucle d'entraînement
# ---------------------------------------------------------------------------

def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
              contrastive_weight: float, temperature: float,
              aug_kwargs: dict, optimizer=None, scaler=None
              ) -> Tuple[float, float, float]:
    """Renvoie (loss_totale_moy, recon_loss_moy, contrastive_loss_moy)."""
    is_train = optimizer is not None
    model.train(mode=is_train)
    totals = {"loss": 0.0, "recon": 0.0, "contrastive": 0.0}
    n_samples = 0

    for (batch,) in loader:
        batch = batch.to(device, non_blocking=True)
        use_amp = scaler is not None and device.type == "cuda"

        with torch.set_grad_enabled(is_train):
            view1 = augment_mfcc(batch, **aug_kwargs)
            view2 = augment_mfcc(batch, **aug_kwargs)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                z1, orig_hw = model.encode(view1)
                z2, _ = model.encode(view2)
                recon1 = model.decode(z1, orig_hw)
                recon2 = model.decode(z2, orig_hw)
                proj1 = model.project(z1)
                proj2 = model.project(z2)

            # Loss calculée en float32 pour la stabilité numérique sous AMP
            batch_f = batch.float()
            recon_loss = 0.5 * (
                F.mse_loss(recon1.float(), batch_f) +
                F.mse_loss(recon2.float(), batch_f)
            )
            contrastive_loss = nt_xent_loss(proj1.float(), proj2.float(), temperature)
            loss = recon_loss + contrastive_weight * contrastive_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        bs = batch.size(0)
        totals["loss"] += loss.item() * bs
        totals["recon"] += recon_loss.item() * bs
        totals["contrastive"] += contrastive_loss.item() * bs
        n_samples += bs

    n_samples = max(1, n_samples)
    return (totals["loss"] / n_samples, totals["recon"] / n_samples,
            totals["contrastive"] / n_samples)


def plot_loss_curves(history: dict, best_epoch: int, contrastive_weight: float,
                      output_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    ax1.plot(epochs, history["train_loss"], label="Train Loss (total)", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss (total)", linewidth=2)
    ax1.axvline(best_epoch, color="grey", linestyle="--", alpha=0.7,
                label=f"Meilleur modèle (époque {best_epoch})")
    ax1.set_ylabel("Loss totale (MSE + λ·NT-Xent)")
    ax1.set_title("Contrastive Conv2D Autoencoder — Loss totale (Train vs Val)",
                   fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["val_recon"], label="Val — Reconstruction (MSE)", linewidth=2)
    ax2.plot(epochs, [contrastive_weight * c for c in history["val_contrastive"]],
              label=f"Val — λ·NT-Xent (λ={contrastive_weight})", linewidth=2)
    ax2.set_xlabel("Époque")
    ax2.set_ylabel("Composante de loss")
    ax2.set_title("Décomposition de la Val Loss", fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Courbes de loss sauvegardées : {output_path}")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraînement du Contrastive Conv2D Autoencoder (MFCC, classe Bee)."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./processed_data"))
    parser.add_argument("--model-dir", type=Path, default=Path("./saved_models"))
    parser.add_argument("--output-dir", type=Path, default=Path("./output"))
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--n-conv-blocks", type=int, default=3)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--contrastive-weight", type=float, default=0.1,
                         help="Poids λ de la loss NT-Xent dans la loss totale "
                              "(défaut : 0.1).")
    parser.add_argument("--temperature", type=float, default=0.5,
                         help="Température de la loss NT-Xent (défaut : 0.5).")
    parser.add_argument("--time-mask-frac", type=float, default=0.15)
    parser.add_argument("--freq-mask-frac", type=float, default=0.15)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--gain-std", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64,
                         help="La loss contrastive bénéficie de batchs plus "
                              "grands (plus de négatifs) — évitez de "
                              "descendre sous ~16 si possible.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"GPU détecté : {torch.cuda.get_device_name(0)} — "
                     f"entraînement accéléré (CUDA + AMP).")
    else:
        logger.warning("Aucun GPU CUDA détecté : entraînement sur CPU (plus lent).")

    try:
        X_train, X_val = load_bee_datasets(args.data_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    n_mfcc, n_frames = X_train.shape[1], X_train.shape[2]
    batch_size = args.batch_size
    if len(X_train) < batch_size:
        logger.warning(
            f"batch_size={batch_size} > taille du train ({len(X_train)}). "
            f"Réduction automatique à {len(X_train)} pour éviter un DataLoader vide."
        )
        batch_size = len(X_train)

    logger.info(f"Données chargées : train={X_train.shape[0]} | val={X_val.shape[0]} "
                f"| forme MFCC=({n_mfcc}, {n_frames}) | batch_size={batch_size}")

    train_loader = make_dataloader(X_train, batch_size, shuffle=True,
                                    drop_last=True, device=device)
    val_loader = make_dataloader(X_val, batch_size, shuffle=False,
                                  drop_last=False, device=device)
    if len(train_loader) == 0:
        logger.error("Le DataLoader d'entraînement est vide (batch_size trop "
                      "grand avec drop_last=True). Réduisez --batch-size.")
        return 1

    model = ContrastiveConv2DAutoencoder(
        in_channels=1, base_channels=args.base_channels,
        n_conv_blocks=args.n_conv_blocks, projection_dim=args.projection_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"ContrastiveConv2DAutoencoder instancié ({n_params:,} paramètres, "
                f"contrastive_weight={args.contrastive_weight}, "
                f"temperature={args.temperature}).".replace(",", " "))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    aug_kwargs = dict(
        time_mask_frac=args.time_mask_frac, freq_mask_frac=args.freq_mask_frac,
        noise_std=args.noise_std, gain_std=args.gain_std,
    )

    history = {"train_loss": [], "val_loss": [], "val_recon": [], "val_contrastive": []}
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    config = ContrastiveConfig(
        base_channels=args.base_channels, n_conv_blocks=args.n_conv_blocks,
        projection_dim=args.projection_dim,
        contrastive_weight=args.contrastive_weight, temperature=args.temperature,
        time_mask_frac=args.time_mask_frac, freq_mask_frac=args.freq_mask_frac,
        noise_std=args.noise_std, gain_std=args.gain_std,
        batch_size=batch_size, lr=args.lr, weight_decay=args.weight_decay,
        epochs=args.epochs, patience=args.patience, seed=args.seed,
    )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "contrastive_autoencoder.pt"

    logger.info(f"Début de l'entraînement ({args.epochs} époques max, "
                f"early stopping patience={args.patience})...")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        train_loss, train_recon, train_contrastive = run_epoch(
            model, train_loader, device, args.contrastive_weight,
            args.temperature, aug_kwargs, optimizer, scaler
        )
        val_loss, val_recon, val_contrastive = run_epoch(
            model, val_loader, device, args.contrastive_weight,
            args.temperature, aug_kwargs, optimizer=None, scaler=None
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_recon"].append(val_recon)
        history["val_contrastive"].append(val_contrastive)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": asdict(config),
                "best_val_loss": best_val_loss,
                "best_val_recon_loss": val_recon,
                "best_val_contrastive_loss": val_contrastive,
                "best_epoch": best_epoch,
                "architecture": "contrastive_conv2d_autoencoder",
            }, model_path)
        else:
            epochs_without_improvement += 1

        marker = " *" if improved else ""
        logger.info(
            f"Époque {epoch:>3}/{args.epochs} | "
            f"train_loss={train_loss:.6f} (recon={train_recon:.6f}, "
            f"ntxent={train_contrastive:.4f}) | "
            f"val_loss={val_loss:.6f} (recon={val_recon:.6f}, "
            f"ntxent={val_contrastive:.4f}) | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"{time.time() - t_epoch:.1f}s{marker}"
        )

        if epochs_without_improvement >= args.patience:
            logger.info(f"Early stopping déclenché à l'époque {epoch} "
                        f"(aucune amélioration depuis {args.patience} époques).")
            break

    total_time = time.time() - t0
    logger.info(f"Entraînement terminé en {total_time / 60:.1f} min. "
                f"Meilleur val_loss={best_val_loss:.6f} (époque {best_epoch}).")
    logger.info(f"Meilleur modèle sauvegardé : {model_path.resolve()}")

    plot_loss_curves(history, best_epoch, args.contrastive_weight,
                      args.output_dir / "loss_curve_model3.png")

    print("\n" + "=" * 72)
    print(" RÉSUMÉ — Entraînement Contrastive Conv2D Autoencoder")
    print("=" * 72)
    print(f"  Device utilisé        : {device} {'(+AMP)' if use_amp else ''}")
    print(f"  Contrastive weight/T  : {args.contrastive_weight} / {args.temperature}")
    print(f"  Époques exécutées     : {len(history['train_loss'])} / {args.epochs}")
    print(f"  Meilleure val_loss    : {best_val_loss:.6f}  (époque {best_epoch})")
    print(f"  Modèle sauvegardé     : {model_path.resolve()}")
    print(f"  Courbe de loss        : {(args.output_dir / 'loss_curve_model3.png').resolve()}")
    print("=" * 72 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())