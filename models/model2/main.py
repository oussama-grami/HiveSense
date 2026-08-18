#!/usr/bin/env python3
"""
main.py — API HiveSense (Modèle 2 : détection d'anomalies acoustiques)
============================================================================

Fonctionnement et caractéristiques du pipeline :
------------------------------------------------
  1. Preprocessing & Normalisation :
     Les caractéristiques MFCC extraites de l'audio sont normalisées (z-score
     via `mfcc_mean.npy` / `mfcc_std.npy`) avant inférence pour correspondre
     strictement à la calibration du modèle.

  2. Gestion du Sample Rate :
     Le pipeline ré-échantillonne automatiquement à 44100 Hz. Si le sample rate
     d'origine est sous le seuil minimal (ex. < 8000 Hz), un avertissement est
     renvoyé dans le champ `warning` de la réponse au lieu de rejeter la requête.

Décisions d'architecture & déploiement :
-----------------------------------------
  - Seuil dynamique : Calculé au démarrage depuis `val_errors_bee.npy` selon
    le percentile défini par `HIVESENSE_PERCENTILE` (défaut : 80.0).
  - Exécution synchrone (`def` vs `async def`) : L'endpoint `/predict/anomaly`
    est défini en synchrone (`def`) afin que FastAPI l'exécute dans un threadpool
    dédié, évitant de bloquer la boucle d'événements asyncio pendant l'inférence
    PyTorch.
  - Fail-Fast Boot : Le service refuse de démarrer en cas d'absence d'un fichier
    de modèle ou de données requis.
  - Import du modèle : `ContrastiveConv2DAutoencoder` est importé directement
    depuis `train_model3.py` comme source unique de vérité.

Variables d'environnement (optionnelles, avec valeurs par défaut) :
    HIVESENSE_MODEL_DIR       (défaut : ./saved_models)
    HIVESENSE_DATA_DIR        (défaut : ./processed_data)
    HIVESENSE_PERCENTILE      (défaut : 80.0)
    HIVESENSE_SR              (défaut : 44100)
    HIVESENSE_DURATION        (défaut : 2.0)
    HIVESENSE_N_MFCC          (défaut : 20)
    HIVESENSE_N_FFT           (défaut : 2048)
    HIVESENSE_HOP_LENGTH      (défaut : 512)
    HIVESENSE_MIN_SAMPLE_RATE (défaut : 8000)

Lancement (développement) :
    uvicorn main:app --reload --port 8000

Lancement (production) :
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
"""

from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import torch
except ImportError:
    raise SystemExit("PyTorch n'est pas installé. Faites `pip install torch`.")

try:
    import librosa
    import soundfile as sf
except ImportError:
    raise SystemExit("librosa/soundfile ne sont pas installés. "
                      "Faites `pip install librosa soundfile`.")

try:
    from fastapi import FastAPI, File, HTTPException, Request, UploadFile
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError:
    raise SystemExit("FastAPI n'est pas installé. Faites "
                      "`pip install fastapi 'uvicorn[standard]' python-multipart`.")

try:
    from train_contrastive_autoencoder import ContrastiveConv2DAutoencoder
except ImportError as e:
    raise SystemExit(
        f"Impossible d'importer ContrastiveConv2DAutoencoder depuis "
        f"train_contrastive_autoencoder.py ({e}). Ce fichier doit se trouver dans le même "
        f"dossier que main.py."
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hivesense-api")


# ---------------------------------------------------------------------------
# Configuration 
# ---------------------------------------------------------------------------

MODEL_DIR = Path(os.environ.get("HIVESENSE_MODEL_DIR", "./saved_models"))
DATA_DIR = Path(os.environ.get("HIVESENSE_DATA_DIR", "./processed_data"))

MODEL_PATH = MODEL_DIR / "contrastive_autoencoder.pt"
VAL_ERRORS_PATH = DATA_DIR / "val_errors_bee.npy"
MFCC_MEAN_PATH = DATA_DIR / "mfcc_mean.npy"
MFCC_STD_PATH = DATA_DIR / "mfcc_std.npy"

PERCENTILE = float(os.environ.get("HIVESENSE_PERCENTILE", "80.0"))

# Paramètres audio 
SR = int(os.environ.get("HIVESENSE_SR", "44100"))
DURATION = float(os.environ.get("HIVESENSE_DURATION", "2.0"))
N_MFCC = int(os.environ.get("HIVESENSE_N_MFCC", "20"))
N_FFT = int(os.environ.get("HIVESENSE_N_FFT", "2048"))
HOP_LENGTH = int(os.environ.get("HIVESENSE_HOP_LENGTH", "512"))
N_SAMPLES = int(round(SR * DURATION))

MIN_SAMPLE_RATE = int(os.environ.get("HIVESENSE_MIN_SAMPLE_RATE", "8000"))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 Mo — un clip de 2s ne pèse que ~350 Ko


# ---------------------------------------------------------------------------
# État applicatif 
# ---------------------------------------------------------------------------

class ModelState:
    def __init__(self) -> None:
        self.model: Optional[torch.nn.Module] = None
        self.device: torch.device = torch.device("cpu")
        self.threshold: Optional[float] = None
        self.scale: Optional[float] = None
        self.mfcc_mean: Optional[np.ndarray] = None
        self.mfcc_std: Optional[np.ndarray] = None
        self.percentile: float = PERCENTILE
        self.checkpoint_info: dict = {}


state = ModelState()


def load_resources() -> None:
    """Charge le modèle + les artefacts de calibration. Lève une exception si
    un fichier requis manque -> le service ne démarre pas (fail-fast)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device d'inférence : {device}")

    required = [MODEL_PATH, VAL_ERRORS_PATH, MFCC_MEAN_PATH, MFCC_STD_PATH]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            "Fichiers requis introuvables : " + ", ".join(str(p) for p in missing) +
            ". Avez-vous lancé prepare_features.py, train_model3.py, puis "
            "evaluate_all.py (qui génère val_errors_bee.npy) au préalable ?"
        )

    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = ContrastiveConv2DAutoencoder(
        in_channels=1, base_channels=cfg["base_channels"],
        n_conv_blocks=cfg["n_conv_blocks"], projection_dim=cfg["projection_dim"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    val_errors = np.load(VAL_ERRORS_PATH)
    if val_errors.size == 0:
        raise RuntimeError(f"{VAL_ERRORS_PATH} est vide — recalibration du seuil impossible.")
    threshold = float(np.percentile(val_errors, PERCENTILE))
    scale = float(np.std(val_errors)) if val_errors.size > 1 else threshold * 0.25

    mfcc_mean = np.load(MFCC_MEAN_PATH).astype(np.float32)
    mfcc_std = np.load(MFCC_STD_PATH).astype(np.float32)
    if mfcc_mean.shape[0] != N_MFCC or mfcc_std.shape[0] != N_MFCC:
        raise RuntimeError(
            f"Incohérence de configuration : mfcc_mean/std ont {mfcc_mean.shape[0]} "
            f"coefficients mais N_MFCC={N_MFCC}. Vérifiez HIVESENSE_N_MFCC."
        )

    state.model = model
    state.device = device
    state.threshold = threshold
    state.scale = scale
    state.mfcc_mean = mfcc_mean
    state.mfcc_std = mfcc_std
    state.checkpoint_info = {
        "architecture": ckpt.get("architecture", "contrastive_conv2d_autoencoder"),
        "best_epoch": ckpt.get("best_epoch"),
        "best_val_loss": ckpt.get("best_val_loss"),
    }

    logger.info(f"Modèle chargé : {MODEL_PATH} (meilleure époque : {ckpt.get('best_epoch')}).")
    logger.info(f"Seuil recalculé dynamiquement depuis {VAL_ERRORS_PATH.name} "
                f"({val_errors.size} échantillons) : percentile={PERCENTILE:.0f} "
                f"-> threshold={threshold:.5f}, scale={scale:.5f}.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_resources()
    except Exception:
        logger.exception("Échec du chargement des ressources au démarrage — arrêt du service.")
        raise
    yield
    logger.info("Arrêt du service HiveSense.")


app = FastAPI(
    title="HiveSense — Modèle 2 : Détection d'anomalies acoustiques",
    description="Classifie un clip audio de ruche en Bee (normal) ou Anomalie "
                "(NoBee / Missing Queen) via un Contrastive Autoencoder.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Filet de sécurité générique : n'expose jamais une trace interne au
    client pour une erreur imprévue."""
    logger.exception(f"Erreur non gérée sur {request.url.path}")
    return JSONResponse(status_code=500, content={"detail": "Erreur interne du serveur."})


# ---------------------------------------------------------------------------
# Prétraitement audio 
# ---------------------------------------------------------------------------

def check_original_sample_rate(audio_bytes: bytes) -> Optional[str]:
    """Lit uniquement les métadonnées (pas le signal complet) pour repérer un
    sample rate source anormalement bas AVANT ré-échantillonnage. Ne bloque
    PAS la requête (prepare_features.py ré-échantillonne systématiquement,
    donc ce n'est pas une erreur de forme) mais retourne un avertissement :
    un enregistrement à 8kHz ré-échantillonné à 44.1kHz n'a aucune énergie
    au-dessus de 4kHz, ce qui peut rendre les MFCC hors-distribution."""
    try:
        info = sf.info(io.BytesIO(audio_bytes))
    except Exception:
        return None  # l'erreur réelle sera de toute façon levée au décodage complet
    if info.samplerate < MIN_SAMPLE_RATE:
        return (f"Sample rate source ({info.samplerate} Hz) très inférieur à celui "
                f"d'entraînement ({SR} Hz) — le signal ré-échantillonné peut être "
                f"hors-distribution ; résultat à interpréter avec prudence.")
    return None


def extract_mfcc_from_bytes(audio_bytes: bytes) -> np.ndarray:
    """Reproduit EXACTEMENT le pipeline de prepare_features.py : mono forcé,
    longueur fixe, mêmes paramètres MFCC. Toute divergence ici fausserait
    silencieusement le score d'anomalie."""
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=SR, mono=True)
    if y.size == 0:
        raise ValueError("audio vide après décodage (0 échantillon)")
    y = librosa.util.fix_length(y, size=N_SAMPLES)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return mfcc.astype(np.float32)


def normalize_mfcc(mfcc: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (mfcc - mean[:, None]) / std[:, None]


def compute_confidence_scores(raw_mse: float, threshold: float,
                               scale: Optional[float] = None) -> Dict[str, float]:
    """Identique à la fonction du même nom dans evaluate_all.py (dupliquée
    ici pour ne pas importer sklearn/matplotlib dans ce service API — voir
    note de dette technique en tête de fichier). Gardez les deux versions
    synchronisées si vous modifiez l'une des deux."""
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
# Schémas de réponse
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    filename: str
    status: str  # "NORMAL" | "ANOMALY"
    reconstruction_mse: float
    threshold_used: float
    confidence_normal: float
    confidence_anomaly: float
    warning: Optional[str] = None  


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    architecture: Optional[str] = None
    threshold: Optional[float] = None
    percentile: float
    device: Optional[str] = None
    best_epoch: Optional[int] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"service": "HiveSense — Modèle 2", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if state.model is not None else "unhealthy",
        model_loaded=state.model is not None,
        architecture=state.checkpoint_info.get("architecture"),
        threshold=state.threshold,
        percentile=state.percentile,
        device=str(state.device) if state.model is not None else None,
        best_epoch=state.checkpoint_info.get("best_epoch"),
    )


@app.post("/predict/anomaly", response_model=PredictionResponse)
def predict_anomaly(file: UploadFile = File(...)) -> PredictionResponse:
    if state.model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé — service indisponible.")

    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers .wav sont acceptés.")

    audio_bytes = file.file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Fichier vide.")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Fichier trop volumineux ({len(audio_bytes) / 1024:.0f} Ko, "
                    f"max {MAX_UPLOAD_BYTES // 1024} Ko).",
        )

    warning = check_original_sample_rate(audio_bytes)

    try:
        mfcc = extract_mfcc_from_bytes(audio_bytes)
    except Exception as e:
        logger.warning(f"Échec de lecture audio pour '{file.filename}' : {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Fichier audio illisible ou corrompu : {e}",
        )

    mfcc_norm = normalize_mfcc(mfcc, state.mfcc_mean, state.mfcc_std)
    tensor = torch.from_numpy(mfcc_norm).unsqueeze(0).unsqueeze(0).float().to(state.device)

    with torch.no_grad():
        recon, _ = state.model(tensor)
        mse = ((recon - tensor) ** 2).mean().item()

    confidence = compute_confidence_scores(mse, state.threshold, state.scale)
    
    status_label = "ANOMALY" if mse >= state.threshold else "NORMAL"

    return PredictionResponse(
        filename=file.filename,
        status=status_label,
        reconstruction_mse=round(mse, 6),
        threshold_used=round(state.threshold, 6),
        confidence_normal=confidence["confidence_normal"],
        confidence_anomaly=confidence["confidence_anomaly"],
        warning=warning,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)