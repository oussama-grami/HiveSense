import os
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
SR          = 16000   # YAMNet attend 16kHz
WINDOW_SIZE = 5       # secondes par fenêtre
HOP_SIZE    = 5       # décalage entre fenêtres (secondes)
N_MFCC      = 40
N_FFT       = 512
HOP_LENGTH  = 256

CLASS_NAMES = {
    0: 'Queen Not Present',
    1: 'Queen Present - Newly Accepted',
    2: 'Queen Present - Rejected',
    3: 'Queen Present - Original'
}

# ─── DÉCOUPER UN AUDIO EN FENÊTRES ───────────────────────────────────────────
def get_windows(audio, sr=SR, window_size=WINDOW_SIZE, hop_size=HOP_SIZE):
    """
    Découpe un audio en fenêtres de window_size secondes.
    Retourne une liste de segments numpy.
    """
    window_samples = sr * window_size
    hop_samples    = sr * hop_size
    windows = []

    start = 0
    while start + window_samples <= len(audio):
        windows.append(audio[start:start + window_samples])
        start += hop_samples

    return windows

# ─── EXTRACTION DES MFCC ─────────────────────────────────────────────────────
def extract_mfcc(audio, sr=SR, n_mfcc=N_MFCC):
    """
    Extrait les MFCC d'un segment audio.
    Retourne une matrice normalisée (n_mfcc, time_steps).
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    # Normalisation
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
    return mfcc.astype(np.float32)

# ─── CONSTRUIRE LE DATAFRAME AVEC FENÊTRES ───────────────────────────────────
def build_windows_dataframe(csv_path):
    """
    Pour chaque fichier audio, découpe en fenêtres de 5s
    et retourne un dataframe avec une ligne par fenêtre.
    """
    df = pd.read_csv(csv_path)
    rows = []
    total = len(df)

    print(f"Découpage de {total} fichiers en fenêtres de {WINDOW_SIZE}s...")

    for idx, row in df.iterrows():
        if idx % 500 == 0:
            print(f"  {idx}/{total}...")

        try:
            audio, _ = librosa.load(row['file_path'], sr=SR)
            windows  = get_windows(audio)

            for w_idx, window in enumerate(windows):
                rows.append({
                    'file_path' : row['file_path'],
                    'window_idx': w_idx,
                    'label'     : int(row['label']),
                    'class_name': row['class_name'],
                })
        except Exception as e:
            print(f"Erreur sur {row['file_path']}: {e}")

    df_windows = pd.DataFrame(rows)
    print(f"\nTotal fenêtres : {len(df_windows)}")
    print(df_windows['class_name'].value_counts())
    return df_windows

# ─── DATASET PYTORCH ─────────────────────────────────────────────────────────
class BeeDataset(Dataset):
    def __init__(self, dataframe, augment=False):
        self.data    = dataframe.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row    = self.data.iloc[idx]
        label  = int(row['label'])

        try:
            audio, _ = librosa.load(row['file_path'], sr=SR)
            windows  = get_windows(audio)
            window   = windows[int(row['window_idx'])]

            # Augmentation : ajout de bruit léger
            if self.augment and np.random.rand() > 0.5:
                noise  = np.random.randn(len(window)) * 0.005
                window = window + noise

            mfcc = extract_mfcc(window)

        except:
            mfcc = np.zeros((N_MFCC, 313), dtype=np.float32)

        # (1, n_mfcc, time_steps)
        tensor = torch.tensor(mfcc).unsqueeze(0)
        return tensor, torch.tensor(label, dtype=torch.long)

# ─── SPLIT ───────────────────────────────────────────────────────────────────
def prepare_splits(df_windows, test_size=0.15, val_size=0.15, random_state=42):
    df_trainval, df_test = train_test_split(
        df_windows,
        test_size=test_size,
        stratify=df_windows['label'],
        random_state=random_state
    )
    val_adj = val_size / (1 - test_size)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_adj,
        stratify=df_trainval['label'],
        random_state=random_state
    )
    print(f"Train : {len(df_train)} fenêtres")
    print(f"Val   : {len(df_val)} fenêtres")
    print(f"Test  : {len(df_test)} fenêtres")
    return df_train, df_val, df_test

# ─── DATALOADERS ─────────────────────────────────────────────────────────────
def get_dataloaders(csv_path, batch_size=32, force_rebuild=False):
    
    windows_cache = csv_path.replace('.csv', '_windows.csv')

    # Si le cache existe déjà, on le charge directement
    if os.path.exists(windows_cache) and not force_rebuild:
        print(f"Cache trouvé — chargement de {windows_cache}")
        df_windows = pd.read_csv(windows_cache)
        print(f"Total fenêtres : {len(df_windows)}")
        print(df_windows['class_name'].value_counts())
    else:
        # Première fois : on découpe et on sauvegarde
        print("Première exécution — découpage des audios...")
        df_windows = build_windows_dataframe(csv_path)
        df_windows.to_csv(windows_cache, index=False)
        print(f"Cache sauvegardé : {windows_cache}")

    # Split
    df_train, df_val, df_test = prepare_splits(df_windows)

    # Class weights
    class_counts  = df_train['label'].value_counts().sort_index()
    class_weights = 1.0 / np.sqrt(class_counts)
    class_weights = class_weights / class_weights.sum()
    weights_tensor = torch.tensor(class_weights.values, dtype=torch.float32)

    print("\nPoids des classes :")
    for label, w in zip(class_counts.index, class_weights):
        print(f"  {CLASS_NAMES[label]}: {w:.4f}")

    # Datasets
    train_ds = BeeDataset(df_train, augment=True)
    val_ds   = BeeDataset(df_val,   augment=False)
    test_ds  = BeeDataset(df_test,  augment=False)

    train_loader = DataLoader(train_ds, batch_size=32,
                          shuffle=True,  num_workers=2,
                          pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=32,
                          shuffle=False, num_workers=2,
                          pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=32,
                          shuffle=False, num_workers=2,
                          pin_memory=True)  

    return train_loader, val_loader, test_loader, weights_tensor