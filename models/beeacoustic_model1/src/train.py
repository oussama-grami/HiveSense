import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import get_dataloaders, CLASS_NAMES
from model import BeeModel

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
CSV_PATH    = '../all_segments.csv'
MODELS_DIR  = '../models'
BATCH_SIZE  = 32
NUM_EPOCHS  = 20
LR          = 3e-4
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── ENTRAÎNEMENT D'UNE EPOCH ─────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} "
                  f"| Loss: {loss.item():.4f}")

    return total_loss / len(loader), correct / total

# ─── VALIDATION ───────────────────────────────────────────────────────────────
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs        = model(inputs)
            loss           = criterion(outputs, labels)

            total_loss += loss.item()
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    return total_loss / len(loader), correct / total

# ─── TRACER LES COURBES ───────────────────────────────────────────────────────
def plot_curves(train_losses, val_losses, train_accs, val_accs, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label='Train Loss', color='#1F4E79')
    ax1.plot(val_losses,   label='Val Loss',   color='#B12A2A')
    ax1.set_title('Courbes de perte')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_accs, label='Train Accuracy', color='#1F4E79')
    ax2.plot(val_accs,   label='Val Accuracy',   color='#2E7D32')
    ax2.set_title('Courbes de précision')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=100)
    plt.close()
    print(f"Courbes sauvegardées.")

# ─── ENTRAÎNEMENT PRINCIPAL ───────────────────────────────────────────────────
def train():
    print(f"Device : {DEVICE}")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Données
    print("\nChargement des données...")
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        CSV_PATH, batch_size=BATCH_SIZE
    )

    # Modèle
    model = BeeModel(num_classes=4).to(DEVICE)
    print(f"\nModèle chargé — {sum(p.numel() for p in model.parameters()):,} paramètres")

    # Loss avec class weighting pour corriger le déséquilibre
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # Optimiseur
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # Scheduler : réduit le LR si la val loss ne s'améliore plus
    scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=3,
    factor=0.5
)


    # Historique
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_loss = float('inf')
    best_epoch    = 0

    print(f"\nDébut entraînement — {NUM_EPOCHS} epochs\n")
    print("=" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = validate(
            model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc:.4f}")

        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optimizer'  : optimizer.state_dict(),
                'val_loss'   : val_loss,
                'val_acc'    : val_acc,
            }, os.path.join(MODELS_DIR, 'best_model.pt'))
            print(f"  ✓ Meilleur modèle sauvegardé (epoch {epoch})")

    print("\n" + "=" * 60)
    print(f"Entraînement terminé.")
    print(f"Meilleur modèle : epoch {best_epoch} | Val Loss : {best_val_loss:.4f}")

    # Tracer les courbes
    plot_curves(train_losses, val_losses, train_accs, val_accs, MODELS_DIR)

    return model, test_loader, class_weights

if __name__ == '__main__':
    train()