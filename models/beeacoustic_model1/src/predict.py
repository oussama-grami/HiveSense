import os
import sys
import torch
import numpy as np
import librosa
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import BeeModel

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pt')
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SR          = 16000
WINDOW_SIZE = 5
N_MFCC      = 40
N_FFT       = 512
HOP_LENGTH  = 256

CLASS_NAMES = {
    0: 'Queen Not Present',
    1: 'Queen Present - Newly Accepted',
    2: 'Queen Present - Rejected',
    3: 'Queen Present - Original'
}

RECOMMENDATIONS = {
    0: 'URGENT : La reine est absente. Inspectez la ruche immédiatement.',
    1: 'Nouvelle reine acceptée. Surveillez l\'évolution.',
    2: 'Reine rejetée. Intervention recommandée.',
    3: 'État normal. Aucune action requise.'
}

# ─── CHARGEMENT DU MODÈLE ─────────────────────────────────────────────────────
def load_model():
    model      = BeeModel(num_classes=4).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

# ─── PRÉTRAITEMENT ────────────────────────────────────────────────────────────
def preprocess(file_path):
    audio, _       = librosa.load(file_path, sr=SR)
    window_samples = SR * WINDOW_SIZE
    windows        = []
    start          = 0

    while start + window_samples <= len(audio):
        windows.append(audio[start:start + window_samples])
        start += window_samples

    if not windows:
        pad    = np.pad(audio, (0, window_samples - len(audio)))
        windows.append(pad)

    tensors = []
    for window in windows:
        mfcc = librosa.feature.mfcc(y=window, sr=SR, n_mfcc=N_MFCC,
                                     n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        tensors.append(torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0))

    return torch.stack(tensors)

# ─── PRÉDICTION ───────────────────────────────────────────────────────────────
def predict(file_path, verbose=True):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    model   = load_model()
    tensors = preprocess(file_path).to(DEVICE)

    with torch.no_grad():
        outputs    = model(tensors)
        probs      = torch.softmax(outputs, dim=1)
        mean_probs = probs.mean(dim=0).cpu().numpy()

    pred_label = int(mean_probs.argmax())
    confidence = float(mean_probs.max())

    result = {
        'file'          : os.path.basename(file_path),
        'prediction'    : CLASS_NAMES[pred_label],
        'label'         : pred_label,
        'confidence'    : round(confidence, 4),
        'recommendation': RECOMMENDATIONS[pred_label],
        'probabilities' : {CLASS_NAMES[i]: round(float(mean_probs[i]), 4)
                           for i in range(4)},
        'windows_count' : len(tensors)
    }

    if verbose:
        print(f"\nFichier      : {result['file']}")
        print(f"Prédiction   : {result['prediction']}")
        print(f"Confiance    : {result['confidence']:.1%}")
        print(f"Conseil      : {result['recommendation']}")
        print(f"Fenêtres     : {result['windows_count']}")
        print(f"\nProbabilités :")
        for cls, prob in result['probabilities'].items():
            bar = '█' * int(prob * 20)
            print(f"  {cls:<35} {prob:.3f}  {bar}")

    return result

# ─── LIGNE DE COMMANDE ────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='BeeAcoustic — Prédiction état de la reine')
    parser.add_argument('file', help='Chemin vers le fichier audio WAV')
    args = parser.parse_args()
    predict(args.file)