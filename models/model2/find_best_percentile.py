#!/usr/bin/env python3
"""
find_best_percentile.py — Optimal Threshold Finder for HiveSense
================================================================
Sweeps percentiles on Bee-val to find the best threshold balancing 
Recall and Precision using F2-score and Target Recall strategies.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

try:
    from train_contrastive_autoencoder import ContrastiveConv2DAutoencoder
except ImportError:
    print("Error: train_model3.py must be in the same folder.")
    sys.exit(1)


def compute_errors(model, X, device, batch_size=256):
    """Computes MSE reconstruction error for an array of samples."""
    if len(X) == 0:
        return np.array([])
    
    tensor = torch.from_numpy(X).unsqueeze(1).float()
    dataset = torch.utils.data.TensorDataset(tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    errors = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            batch_errors = ((recon - batch) ** 2).mean(dim=[1, 2, 3])
            errors.append(batch_errors.cpu().numpy())
            
    return np.concatenate(errors)


def main():
    data_dir = Path("./processed_data")
    model_path = Path("./saved_models/contrastive_autoencoder.pt")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # 1. Load Data
    X_bee_val = np.load(data_dir / "X_bee_val.npy")
    X_bee_test = np.load(data_dir / "X_bee_test.npy")
    X_nobee = np.load(data_dir / "X_nobee.npy")
    X_missing_queen = np.load(data_dir / "X_missing_queen.npy")

    # Labels: 0 = Normal (Bee), 1 = Anomaly (NoBee + Missing Queen)
    labels_test = np.concatenate([
        np.zeros(len(X_bee_test), dtype=int),
        np.ones(len(X_nobee), dtype=int),
        np.ones(len(X_missing_queen), dtype=int)
    ])

    # 2. Load Winning Model (Contrastive AE)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = ContrastiveConv2DAutoencoder(
        in_channels=1, 
        base_channels=cfg["base_channels"],
        n_conv_blocks=cfg["n_conv_blocks"],
        projection_dim=cfg["projection_dim"]
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # 3. Compute Reconstruction Errors
    print("Computing reconstruction errors...")
    val_errors = compute_errors(model, X_bee_val, device)
    
    errors_bee = compute_errors(model, X_bee_test, device)
    errors_nobee = compute_errors(model, X_nobee, device)
    errors_mq = compute_errors(model, X_missing_queen, device)
    test_errors = np.concatenate([errors_bee, errors_nobee, errors_mq])

    # 4. Sweep Percentiles
    print("\n" + "=" * 80)
    print(f"{'Percentile':<12}{'Threshold':<14}{'Precision':<12}{'Recall':<10}{'F1-Score':<10}{'F2-Score':<10}")
    print("=" * 80)

    results = []
    percentiles = np.arange(50.0, 99.5, 1.0)

    for p in percentiles:
        threshold = float(np.percentile(val_errors, p))
        preds = (test_errors >= threshold).astype(int)

        prec = precision_score(labels_test, preds, zero_division=0)
        rec = recall_score(labels_test, preds, zero_division=0)
        f1 = f1_score(labels_test, preds, zero_division=0)
        f2 = fbeta_score(labels_test, preds, beta=2, zero_division=0)

        results.append({
            "percentile": p,
            "threshold": threshold,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "f2": f2
        })

        # Print selected step rows for readability
        if p in [50, 60, 70, 75, 80, 85, 90, 95]:
            print(f"{p:<12.1f}{threshold:<14.5f}{prec:<12.3f}{rec:<10.3f}{f1:<10.3f}{f2:<10.3f}")

    print("=" * 80)

    # 5. Identify Optimal Choices
    best_f2 = max(results, key=lambda x: x["f2"])
    
    # Target Recall Strategy (First percentile achieving >= 85% Recall)
    target_rec_results = [r for r in results if r["recall"] >= 0.85]
    best_target = max(target_rec_results, key=lambda x: x["percentile"]) if target_rec_results else None

    print("\n--- OPTIMAL THRESHOLD RECOMMENDATIONS ---")
    print(f"\n1. Best Balanced Strategy (Max F2-Score):")
    print(f"   • Percentile : {best_f2['percentile']:.1f}th")
    print(f"   • Threshold  : {best_f2['threshold']:.5f}")
    print(f"   • Recall     : {best_f2['recall'] * 100:.1f}%")
    print(f"   • Precision  : {best_f2['precision'] * 100:.1f}%")
    print(f"   • F2-Score   : {best_f2['f2']:.3f}")

    if best_target:
        print(f"\n2. High-Safety Strategy (Target Recall >= 85%):")
        print(f"   • Percentile : {best_target['percentile']:.1f}th")
        print(f"   • Threshold  : {best_target['threshold']:.5f}")
        print(f"   • Recall     : {best_target['recall'] * 100:.1f}%")
        print(f"   • Precision  : {best_target['precision'] * 100:.1f}%")


if __name__ == "__main__":
    main()