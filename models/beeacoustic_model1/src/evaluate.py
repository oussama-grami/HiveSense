import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import BeeModel
from dataset import get_dataloaders

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
MODEL_PATH  = '../models/best_model.pt'
CSV_PATH    = '../all_segments.csv'
RESULTS_DIR = '../models'
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = [
    'Queen Not Present',
    'Queen Present - Newly Accepted',
    'Queen Present - Rejected',
    'Queen Present - Original'
]

# ─── CHARGEMENT ───────────────────────────────────────────────────────────────
def load_model():
    model      = BeeModel(num_classes=4).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"Modèle chargé — epoch {checkpoint['epoch']}")
    print(f"Val Acc  : {checkpoint['val_acc']:.4f}")
    print(f"Val Loss : {checkpoint['val_loss']:.4f}")
    return model

# ─── PRÉDICTIONS ──────────────────────────────────────────────────────────────
def get_predictions(model, test_loader):
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs  = inputs.to(DEVICE)
            outputs = model(inputs)
            probs   = torch.softmax(outputs, dim=1)
            preds   = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return (np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs))

# ─── RAPPORT DE CLASSIFICATION ────────────────────────────────────────────────
def print_classification_report(labels, preds):
    print("\n" + "="*60)
    print("RAPPORT DE CLASSIFICATION")
    print("="*60)
    print(classification_report(labels, preds,
                                 target_names=CLASS_NAMES, digits=3))
    wf1 = f1_score(labels, preds, average='weighted')
    mf1 = f1_score(labels, preds, average='macro')
    print(f"Weighted F1 : {wf1:.4f}")
    print(f"Macro F1    : {mf1:.4f}")
    return wf1, mf1

# ─── MATRICE DE CONFUSION ─────────────────────────────────────────────────────
def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)

    # Matrice en valeurs absolues
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title('Matrice de confusion — Valeurs absolues')
    plt.ylabel('Réel')
    plt.xlabel('Prédit')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=100)
    plt.close()

    # Matrice normalisée (pourcentages)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title('Matrice de confusion — Normalisée (%)')
    plt.ylabel('Réel')
    plt.xlabel('Prédit')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_normalized.png'), dpi=100)
    plt.close()
    print("Matrices de confusion sauvegardées.")

# ─── COURBES ROC ──────────────────────────────────────────────────────────────
def plot_roc_curves(labels, probs):
    labels_bin = label_binarize(labels, classes=[0, 1, 2, 3])
    colors     = ['#1F4E79', '#2E7D32', '#B45309', '#B12A2A']

    plt.figure(figsize=(9, 7))
    auc_scores = []

    for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        auc         = roc_auc_score(labels_bin[:, i], probs[:, i])
        auc_scores.append(auc)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{cls} (AUC = {auc:.3f})')

    plt.plot([0,1], [0,1], 'k--', lw=1, label='Aléatoire')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbes ROC par classe')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curves.png'), dpi=100)
    plt.close()

    weighted_auc = roc_auc_score(labels_bin, probs,
                                  multi_class='ovr', average='weighted')
    print(f"AUC-ROC par classe :")
    for cls, auc in zip(CLASS_NAMES, auc_scores):
        print(f"  {cls:<35} : {auc:.4f}")
    print(f"AUC-ROC weighted   : {weighted_auc:.4f}")
    print("Courbes ROC sauvegardées.")
    return weighted_auc

# ─── SAUVEGARDE DES RÉSULTATS ─────────────────────────────────────────────────
def save_results(wf1, mf1, auc):
    results = {
        'weighted_f1' : round(wf1, 4),
        'macro_f1'    : round(mf1, 4),
        'auc_roc'     : round(auc, 4),
    }
    import json
    with open(os.path.join(RESULTS_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nRésultats sauvegardés dans models/evaluation_results.json")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def evaluate():
    print(f"Device : {DEVICE}")

    model = load_model()
    _, _, test_loader, _ = get_dataloaders(CSV_PATH)

    print("\nCalcul des prédictions sur le jeu de test...")
    preds, labels, probs = get_predictions(model, test_loader)

    wf1, mf1 = print_classification_report(labels, preds)
    plot_confusion_matrix(labels, preds)
    auc = plot_roc_curves(labels, probs)
    save_results(wf1, mf1, auc)

    print("\n" + "="*60)
    print("RÉSUMÉ FINAL")
    print("="*60)
    print(f"Weighted F1 : {wf1:.4f}")
    print(f"Macro F1    : {mf1:.4f}")
    print(f"AUC-ROC     : {auc:.4f}")
    print("="*60)

if __name__ == '__main__':
    evaluate()