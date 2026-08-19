import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import librosa
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
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
    1: 'Nouvelle reine acceptée. Surveillez l\'évolution dans les prochains jours.',
    2: 'Reine rejetée par la colonie. Intervention recommandée.',
    3: 'État normal. Aucune action requise.'
}

# ─── CHARGER LE MODÈLE ────────────────────────────────────────────────────────
app   = Flask(__name__)
CORS(app)
model = BeeModel(num_classes=4).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state'])
model.eval()
print(f"Modèle chargé — Val Acc : {checkpoint['val_acc']:.4f}")

# ─── PRÉTRAITEMENT ────────────────────────────────────────────────────────────
def preprocess_audio(file_path):
    """
    Charge un fichier audio, découpe en fenêtres de 5s
    et retourne les MFCC de chaque fenêtre.
    """
    audio, _ = librosa.load(file_path, sr=SR)
    window_samples = SR * WINDOW_SIZE
    windows = []
    start   = 0
    while start + window_samples <= len(audio):
        windows.append(audio[start:start + window_samples])
        start += window_samples

    if not windows:
        windows.append(audio[:window_samples] if len(audio) >= window_samples
                       else np.pad(audio, (0, window_samples - len(audio))))

    tensors = []
    for window in windows:
        mfcc = librosa.feature.mfcc(y=window, sr=SR, n_mfcc=N_MFCC,
                                     n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        tensors.append(torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0))

    return torch.stack(tensors)

# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status'  : 'ok',
        'model'   : 'BeeModel v1.0',
        'device'  : str(DEVICE),
        'val_acc' : round(float(checkpoint['val_acc']), 4)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Reçoit un fichier audio WAV et retourne la prédiction.
    Usage : POST /predict avec form-data 'file' = fichier WAV
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni. Utilisez le champ "file".'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Fichier vide.'}), 400

    # Sauvegarder temporairement le fichier
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    try:
        # Prétraitement
        tensors = preprocess_audio(tmp_path).to(DEVICE)

        # Inférence sur toutes les fenêtres
        with torch.no_grad():
            outputs = model(tensors)
            probs   = torch.softmax(outputs, dim=1)

        # Moyenne des probabilités sur toutes les fenêtres
        mean_probs = probs.mean(dim=0).cpu().numpy()
        pred_label = int(mean_probs.argmax())
        confidence = float(mean_probs.max())

        # Résultat par fenêtre
        window_preds = [
            {
                'window'    : i,
                'class'     : CLASS_NAMES[int(p.argmax())],
                'confidence': round(float(p.max()), 4)
            }
            for i, p in enumerate(probs.cpu())
        ]

        return jsonify({
            'prediction'    : CLASS_NAMES[pred_label],
            'label'         : pred_label,
            'confidence'    : round(confidence, 4),
            'recommendation': RECOMMENDATIONS[pred_label],
            'probabilities' : {
                CLASS_NAMES[i]: round(float(mean_probs[i]), 4)
                for i in range(4)
            },
            'windows_count' : len(tensors),
            'window_details': window_preds
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        os.unlink(tmp_path)

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Reçoit plusieurs fichiers audio et retourne une prédiction pour chacun.
    Usage : POST /predict/batch avec form-data 'files' = liste de fichiers WAV
    """
    if 'files' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni.'}), 400

    files   = request.files.getlist('files')
    results = []

    for file in files:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)
        try:
            tensors    = preprocess_audio(tmp_path).to(DEVICE)
            with torch.no_grad():
                outputs    = model(tensors)
                probs      = torch.softmax(outputs, dim=1)
            mean_probs = probs.mean(dim=0).cpu().numpy()
            pred_label = int(mean_probs.argmax())
            results.append({
                'filename'  : file.filename,
                'prediction': CLASS_NAMES[pred_label],
                'label'     : pred_label,
                'confidence': round(float(mean_probs.max()), 4),
            })
        except Exception as e:
            results.append({'filename': file.filename, 'error': str(e)})
        finally:
            os.unlink(tmp_path)

    return jsonify({'results': results, 'total': len(results)})

# ─── LANCEMENT ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"API BeeAcoustic démarrée sur http://localhost:5000")
    print(f"Routes disponibles :")
    print(f"  GET  /health          — vérifier l'état de l'API")
    print(f"  POST /predict         — prédire sur un fichier audio")
    print(f"  POST /predict/batch   — prédire sur plusieurs fichiers")
    app.run(debug=False, host='0.0.0.0', port=5000)