#!/usr/bin/env python3
"""
train_conv2d_autoencoder.py — Entraînement du Conv2D Autoencoder (PyTorch)
=============================================================================

Architecture 1 du  Modèle 2 (détection d'anomalies acoustiques).

Principe du modèle :
--------------------
L'autoencodeur est entraîné de manière non supervisée exclusivement sur les
données normales (Bee). En apprenant à reconstruire les spectrogrammes MFCC
standards, il produira une erreur de reconstruction significativement plus
élevée sur les sons anormaux (NoBee / Missing Queen) lors de l'évaluation.

Fonctionnalités du script :
---------------------------
  1. Charge `X_bee_train.npy` et `X_bee_val.npy` depuis `./processed_data/`.
  2. Définit une architecture Conv2D Autoencoder entièrement convolutive avec
     padding dynamique s'adaptant automatiquement aux dimensions des MFCC.
  3. Exécute l'entraînement avec support GPU (CUDA et précision mixte AMP)
     et bascule automatiquement sur CPU si aucun GPU n'est disponible.
  4. Trace et sauvegarde les courbes de perte (`output/loss_curve_conv2d_autoencoder.png`).
  5. Sauvegarde le meilleur modèle (selon la perte de validation) et sa
     configuration dans `saved_models/conv2d_autoencoder.pt`.

Usage :
    python train_conv2d_autoencoder.py
    python train_conv2d_autoencoder.py --epochs 100 --batch-size 64 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

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
logger = logging.getLogger("train_conv2d_autoencoder")


# ---------------------------------------------------------------------------
# Architecture : Conv2D Autoencoder entièrement convolutif
# ---------------------------------------------------------------------------

class Conv2DAutoencoder(nn.Module):
    """Autoencodeur convolutif 2D pour matrices MFCC (1, n_mfcc, n_frames).

    Encodeur : 3 blocs Conv2d(stride=2) -> compression x8 en H et en W.
    Décodeur : 3 blocs ConvTranspose2d(stride=2) miroir.

    La forme d'entrée n'est PAS supposée être un multiple de 8 : le forward()
    applique un padding dynamique (zéro-padding à droite/en bas) avant
    l'encodeur, puis recadre la sortie du décodeur à la taille d'origine.
    Cela rend le modèle robuste à toute variation du nombre de frames MFCC
    (dépend de hop_length / durée exacte des clips) sans recalcul manuel.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 16,
                 n_conv_blocks: int = 3):
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

    def _pad_to_multiple(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        h, w = x.shape[-2], x.shape[-1]
        f = self.downsample_factor
        pad_h = (f - h % f) % f
        pad_w = (f - w % f) % f
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
        return x, h, w

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_padded, h, w = self._pad_to_multiple(x)
        z = self.encoder(x_padded)
        out = self.decoder(z)
        out = out[:, :, :h, :w]  # recadrage à la taille d'origine exacte
        return out, z


@dataclass
class TrainConfig:
    n_mfcc: int
    n_frames: int
    base_channels: int
    n_conv_blocks: int
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
    X_train = np.load(train_path)
    X_val = np.load(val_path)
    return X_train, X_val


def make_dataloader(X: np.ndarray, batch_size: int, shuffle: bool,
                     device: torch.device) -> DataLoader:
    # (N, n_mfcc, n_frames) -> (N, 1, n_mfcc, n_frames)
    tensor = torch.from_numpy(X).unsqueeze(1).float()
    dataset = TensorDataset(tensor)
    pin_memory = device.type == "cuda"
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                       pin_memory=pin_memory, drop_last=False)


# ---------------------------------------------------------------------------
# Boucle d'entraînement
# ---------------------------------------------------------------------------

def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
              optimizer=None, scaler: "torch.cuda.amp.GradScaler | None" = None
              ) -> float:
    """Exécute une époque. Si optimizer est fourni -> mode train, sinon eval."""
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    n_samples = 0

    for (batch,) in loader:
        batch = batch.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            use_amp = scaler is not None and device.type == "cuda"
            with torch.autocast(device_type=device.type, enabled=use_amp):
                reconstruction, _ = model(batch)
                loss = F.mse_loss(reconstruction, batch)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item() * batch.size(0)
        n_samples += batch.size(0)

    return total_loss / max(1, n_samples)


def plot_loss_curves(train_losses: List[float], val_losses: List[float],
                      best_epoch: int, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss (MSE)", linewidth=2)
    ax.plot(epochs, val_losses, label="Val Loss (MSE)", linewidth=2)
    ax.axvline(best_epoch, color="grey", linestyle="--", alpha=0.7,
                label=f"Meilleur modèle (époque {best_epoch})")
    ax.set_xlabel("Époque")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Conv2D Autoencoder — Courbes de Loss (Train vs Val)",
                  fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
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
        description="Entraînement du Conv2D Autoencoder (MFCC, classe Bee)."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./processed_data"))
    parser.add_argument("--model-dir", type=Path, default=Path("./saved_models"))
    parser.add_argument("--output-dir", type=Path, default=Path("./output"))
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--n-conv-blocks", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15,
                         help="Early stopping : nb d'époques sans amélioration "
                              "de la val loss avant arrêt (défaut : 15).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true",
                         help="Désactive la précision mixte (AMP) même si un "
                              "GPU CUDA est disponible.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"GPU détecté : {torch.cuda.get_device_name(0)} — "
                     f"entraînement accéléré (CUDA + AMP).")
    else:
        logger.warning("Aucun GPU CUDA détecté : entraînement sur CPU (plus lent).")

    # --- Données --------------------------------------------------------------
    try:
        X_train, X_val = load_bee_datasets(args.data_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    n_mfcc, n_frames = X_train.shape[1], X_train.shape[2]
    logger.info(f"Données chargées : train={X_train.shape[0]} | val={X_val.shape[0]} "
                f"| forme MFCC=({n_mfcc}, {n_frames})")

    train_loader = make_dataloader(X_train, args.batch_size, shuffle=True, device=device)
    val_loader = make_dataloader(X_val, args.batch_size, shuffle=False, device=device)

    # --- Modèle -----------------------------------------------------------------
    model = Conv2DAutoencoder(
        in_channels=1, base_channels=args.base_channels,
        n_conv_blocks=args.n_conv_blocks,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Conv2DAutoencoder instancié ({n_params:,} paramètres).".replace(",", " "))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None

    # --- Boucle d'entraînement avec early stopping ------------------------------
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    config = TrainConfig(
        n_mfcc=n_mfcc, n_frames=n_frames, base_channels=args.base_channels,
        n_conv_blocks=args.n_conv_blocks, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs,
        patience=args.patience, seed=args.seed,
    )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "conv2d_autoencoder.pt"

    logger.info(f"Début de l'entraînement ({args.epochs} époques max, "
                f"early stopping patience={args.patience})...")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        train_loss = run_epoch(model, train_loader, device, optimizer, scaler)
        val_loss = run_epoch(model, val_loader, device, optimizer=None, scaler=None)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": asdict(config),
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "architecture": "conv2d_autoencoder",
            }, model_path)
        else:
            epochs_without_improvement += 1

        marker = " *" if improved else ""
        logger.info(
            f"Époque {epoch:>3}/{args.epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
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

    # --- Courbes de loss ----------------------------------------------------------
    plot_loss_curves(train_losses, val_losses, best_epoch,
                      args.output_dir / "loss_curve_conv2d_autoencoder.png")

    print("\n" + "=" * 72)
    print(" RÉSUMÉ — Entraînement Conv2D Autoencoder")
    print("=" * 72)
    print(f"  Device utilisé        : {device} {'(+AMP)' if use_amp else ''}")
    print(f"  Époques exécutées     : {len(train_losses)} / {args.epochs}")
    print(f"  Meilleure val_loss    : {best_val_loss:.6f}  (époque {best_epoch})")
    print(f"  Modèle sauvegardé     : {model_path.resolve()}")
    print(f"  Courbe de loss        : {(args.output_dir / 'loss_curve_conv2d_autoencoder.png').resolve()}")
    print("=" * 72 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())