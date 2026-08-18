#!/usr/bin/env python3
"""
train_vae.py — Entraînement du Conv2D Variational Autoencoder (PyTorch)
============================================================================

Architecture 2 du Modèle 2 (détection d'anomalies acoustiques).

Caractéristiques de l'architecture :
-------------------------------------
Ce VAE utilise un espace latent dense contrôlé par des paramètres de moyenne 
(mu) et de variance (logvar) via l'astuce de reparamétrisation. Cette structure 
impose une dimension d'entrée fixe (n_mfcc, n_frames), correspondant aux 
matrice de caractéristiques générées par `prepare_features.py`.

Fonctionnalités du script :
---------------------------
  1. Charge `X_bee_train.npy` et `X_bee_val.npy` depuis `./processed_data/`.
  2. Définit l'architecture Conv2D VAE (Encodeur -> (mu, logvar) -> Reparamétrisation -> Décodeur).
  3. Calcule la perte globale : Erreur de reconstruction (MSE) + Divergence KL (pondérée par beta).
  4. Exécute l'entraînement avec support GPU (CUDA et précision mixte AMP), 
     scheduler de taux d'apprentissage et Early Stopping.
  5. Sauvegarde le meilleur modèle dans `saved_models/vae_autoencoder.pt` ainsi 
     que les courbes de perte (totale, reconstruction et KL) dans `output/loss_curve_vae.png`.

Usage :
    python train_vae.py
    python train_vae.py --latent-dim 64 --beta 0.001 --epochs 100
"""


from __future__ import annotations

import argparse
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
logger = logging.getLogger("train_vae")


# ---------------------------------------------------------------------------
# Architecture : Conv2D VAE
# ---------------------------------------------------------------------------

class Conv2DVAE(nn.Module):
    """VAE convolutif 2D pour matrices MFCC de forme fixe (1, n_mfcc, n_frames).

    Encodeur : n_conv_blocks blocs Conv2d(stride=2) -> flatten -> Linear(mu),
    Linear(logvar).
    Décodeur : Linear -> reshape -> n_conv_blocks blocs ConvTranspose2d
    (miroir de l'encodeur) -> recadrage à la taille d'origine.
    """

    def __init__(self, n_mfcc: int, n_frames: int, in_channels: int = 1,
                 base_channels: int = 16, n_conv_blocks: int = 3,
                 latent_dim: int = 64):
        super().__init__()
        self.orig_h, self.orig_w = n_mfcc, n_frames
        self.latent_dim = latent_dim
        self.n_conv_blocks = n_conv_blocks

        df = 2 ** n_conv_blocks
        self.pad_h = (df - n_mfcc % df) % df
        self.pad_w = (df - n_frames % df) % df
        self.padded_h = n_mfcc + self.pad_h
        self.padded_w = n_frames + self.pad_w
        self.bottleneck_h = self.padded_h // df
        self.bottleneck_w = self.padded_w // df

        # --- Encodeur ---------------------------------------------------------
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
        self.flatten_dim = self.bottleneck_channels * self.bottleneck_h * self.bottleneck_w

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # --- Décodeur -----------------------------------------------------------
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

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

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, self.bottleneck_channels, self.bottleneck_h, self.bottleneck_w)
        out = self.decoder(h)
        return out[:, :, :self.orig_h, :self.orig_w]  # recadrage taille d'origine

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.shape[-2:] != (self.orig_h, self.orig_w):
            raise ValueError(
                f"Conv2DVAE attend des entrées de forme "
                f"({self.orig_h}, {self.orig_w}) mais a reçu {tuple(x.shape[-2:])}. "
                f"Contrairement au Conv2DAutoencoder, ce VAE a un goulot "
                f"d'étranglement dense fixé à la construction — reconstruisez "
                f"le modèle avec les bonnes dimensions si vos données changent."
            )
        if self.pad_h or self.pad_w:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h))
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor,
             logvar: torch.Tensor, beta: float
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Loss = Reconstruction (MSE, moyennée sur tous les éléments) +
    beta * KL Divergence (sommée sur le vecteur latent, moyennée sur le
    batch). Renvoie (loss_totale, recon_loss, kl_loss) pour diagnostic."""
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


@dataclass
class VAEConfig:
    n_mfcc: int
    n_frames: int
    base_channels: int
    n_conv_blocks: int
    latent_dim: int
    beta: float
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
                     device: torch.device) -> DataLoader:
    tensor = torch.from_numpy(X).unsqueeze(1).float()
    dataset = TensorDataset(tensor)
    pin_memory = device.type == "cuda"
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                       pin_memory=pin_memory, drop_last=False)


# ---------------------------------------------------------------------------
# Boucle d'entraînement
# ---------------------------------------------------------------------------

def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
              beta: float, optimizer=None, scaler=None
              ) -> Tuple[float, float, float]:
    """Exécute une époque. Si optimizer est fourni -> mode train, sinon eval.
    Renvoie (loss_totale_moy, recon_loss_moy, kl_loss_moy)."""
    is_train = optimizer is not None
    model.train(mode=is_train)
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    n_samples = 0

    for (batch,) in loader:
        batch = batch.to(device, non_blocking=True)
        use_amp = scaler is not None and device.type == "cuda"

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                recon, mu, logvar = model(batch)

            # La loss (en particulier exp(logvar) pour la KL) est calculée en
            # float32 pour la stabilité numérique, même si le forward a
            # tourné en float16 sous AMP.
            loss, recon_loss, kl_loss = vae_loss(
                recon.float(), batch.float(), mu.float(), logvar.float(), beta
            )

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
        totals["kl"] += kl_loss.item() * bs
        n_samples += bs

    n_samples = max(1, n_samples)
    return totals["loss"] / n_samples, totals["recon"] / n_samples, totals["kl"] / n_samples


def plot_loss_curves(history: dict, best_epoch: int, beta: float,
                      output_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    ax1.plot(epochs, history["train_loss"], label="Train Loss (total)", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss (total)", linewidth=2)
    ax1.axvline(best_epoch, color="grey", linestyle="--", alpha=0.7,
                label=f"Meilleur modèle (époque {best_epoch})")
    ax1.set_ylabel("Loss totale (MSE + β·KL)")
    ax1.set_title("Conv2D VAE — Loss totale (Train vs Val)",
                   fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["val_recon"], label="Val — Reconstruction (MSE)", linewidth=2)
    ax2.plot(epochs, [beta * k for k in history["val_kl"]],
              label=f"Val — β·KL (β={beta})", linewidth=2)
    ax2.set_xlabel("Époque")
    ax2.set_ylabel("Composante de loss")
    ax2.set_title("Décomposition de la Val Loss (diagnostic posterior collapse)",
                   fontsize=12)
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
        description="Entraînement du Conv2D VAE (MFCC, classe Bee)."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./processed_data"))
    parser.add_argument("--model-dir", type=Path, default=Path("./saved_models"))
    parser.add_argument("--output-dir", type=Path, default=Path("./output"))
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--n-conv-blocks", type=int, default=3)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.001,
                         help="Poids de la KL Divergence dans la loss totale "
                              "(défaut : 0.001).")
    parser.add_argument("--batch-size", type=int, default=64)
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
    logger.info(f"Données chargées : train={X_train.shape[0]} | val={X_val.shape[0]} "
                f"| forme MFCC=({n_mfcc}, {n_frames}) | latent_dim={args.latent_dim} "
                f"| beta={args.beta}")

    train_loader = make_dataloader(X_train, args.batch_size, shuffle=True, device=device)
    val_loader = make_dataloader(X_val, args.batch_size, shuffle=False, device=device)

    model = Conv2DVAE(
        n_mfcc=n_mfcc, n_frames=n_frames, in_channels=1,
        base_channels=args.base_channels, n_conv_blocks=args.n_conv_blocks,
        latent_dim=args.latent_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Conv2DVAE instancié ({n_params:,} paramètres, "
                f"bottleneck spatial={model.bottleneck_h}x{model.bottleneck_w}, "
                f"flatten_dim={model.flatten_dim}).".replace(",", " "))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None

    history = {"train_loss": [], "val_loss": [], "val_recon": [], "val_kl": []}
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    config = VAEConfig(
        n_mfcc=n_mfcc, n_frames=n_frames, base_channels=args.base_channels,
        n_conv_blocks=args.n_conv_blocks, latent_dim=args.latent_dim,
        beta=args.beta, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, epochs=args.epochs,
        patience=args.patience, seed=args.seed,
    )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "vae_autoencoder.pt"

    logger.info(f"Début de l'entraînement ({args.epochs} époques max, "
                f"early stopping patience={args.patience})...")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        train_loss, train_recon, train_kl = run_epoch(
            model, train_loader, device, args.beta, optimizer, scaler
        )
        val_loss, val_recon, val_kl = run_epoch(
            model, val_loader, device, args.beta, optimizer=None, scaler=None
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_recon"].append(val_recon)
        history["val_kl"].append(val_kl)

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
                "best_val_kl_loss": val_kl,
                "best_epoch": best_epoch,
                "architecture": "conv2d_vae",
            }, model_path)
        else:
            epochs_without_improvement += 1

        marker = " *" if improved else ""
        logger.info(
            f"Époque {epoch:>3}/{args.epochs} | "
            f"train_loss={train_loss:.6f} (recon={train_recon:.6f}, kl={train_kl:.4f}) | "
            f"val_loss={val_loss:.6f} (recon={val_recon:.6f}, kl={val_kl:.4f}) | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"{time.time() - t_epoch:.1f}s{marker}"
        )

        if val_kl < 1e-3 and epoch > 5:
            logger.warning(
                "KL divergence proche de 0 : signe possible de 'posterior "
                "collapse' (le décodeur ignore z). Envisagez de réduire beta "
                "ou d'utiliser un warm-up de beta."
            )

        if epochs_without_improvement >= args.patience:
            logger.info(f"Early stopping déclenché à l'époque {epoch} "
                        f"(aucune amélioration depuis {args.patience} époques).")
            break

    total_time = time.time() - t0
    logger.info(f"Entraînement terminé en {total_time / 60:.1f} min. "
                f"Meilleur val_loss={best_val_loss:.6f} (époque {best_epoch}).")
    logger.info(f"Meilleur modèle sauvegardé : {model_path.resolve()}")

    plot_loss_curves(history, best_epoch, args.beta,
                      args.output_dir / "loss_curve_vae.png")

    print("\n" + "=" * 72)
    print(" RÉSUMÉ — Entraînement Conv2D VAE")
    print("=" * 72)
    print(f"  Device utilisé        : {device} {'(+AMP)' if use_amp else ''}")
    print(f"  Latent dim / beta     : {args.latent_dim} / {args.beta}")
    print(f"  Époques exécutées     : {len(history['train_loss'])} / {args.epochs}")
    print(f"  Meilleure val_loss    : {best_val_loss:.6f}  (époque {best_epoch})")
    print(f"  Modèle sauvegardé     : {model_path.resolve()}")
    print(f"  Courbe de loss        : {(args.output_dir / 'loss_curve_vae.png').resolve()}")
    print("=" * 72 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())