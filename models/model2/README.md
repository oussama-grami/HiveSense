# HiveSense — Model 2: Acoustic Anomaly Detection

Unsupervised acoustic anomaly detection for beehive monitoring. Trained
**only on normal (`Bee`) audio**, the model outputs a **binary** decision —
`NORMAL` or `ANOMALY` — with a confidence score, based on reconstruction
error. `NoBee` (silent/inactive hive) and `Missing Queen` (acoustic anomaly
indicating queen absence) are the two real-world anomaly types present in
the *data* (used to train and evaluate), but the model itself does **not**
classify which anomaly type it's seeing — it only flags deviation from
normal.

> Scope note: this README covers Model 2 only (feature extraction, the three
> candidate architectures, evaluation, and the serving API). Flutter
> integration and Model 1 (queen-state classifier) are out of scope here.

---

## 1. Overview

**Goal:** flag anomalous hive sounds without needing labeled anomaly
examples at training time. The core assumption: an autoencoder trained
exclusively to reconstruct *normal* Bee audio will reconstruct it well, but
will show high reconstruction error on anything that deviates from that
distribution — because it has never learned to reconstruct those patterns.

**Output:** binary `NORMAL` / `ANOMALY` classification + a confidence score,
derived from thresholding the reconstruction MSE. `Bee`/`NoBee`/`Missing Queen` are dataset labels used
for training (`Bee` only) and evaluation (all three).

**Approach:** three autoencoder architectures were trained (on `Bee` audio
only) and compared on a held-out set containing all three classes; the best
one is served in production.

**Dataset:** [Beehive Buzz Anomalies](https://www.kaggle.com/datasets/yevheniiklymenko/beehive-buzz-anomalies)
(Kaggle), a processed/segmented version of the TBON dataset (NU-Hive + OSBH).

| Class | Meaning | Count | Used for training? |
|---|---|---:|---|
| `Bee` | Normal, healthy colony | 5,473 | Yes (80/10/10 train/val/test split) |
| `NoBee` | Silent / inactive hive | 3,458 | No — held out entirely for evaluation |
| `Missing Queen` | Acoustic anomaly (queen absent) | 4,861 | No — held out entirely for evaluation |

All clips: 2.00s, 44.1 kHz (mono/stereo mixed, forced to mono at
preprocessing). Total: 13,792 files.
### Dataset & Audio Feature Visualization

<p align="center">
  <img src="output/distribution_classes.png" width="45%" alt="Class Distribution" />
  <img src="output/mel_spectrogram_bee.png" width="45%" alt="Bee Mel Spectrogram" />
</p>
Pipeline structure loosely inspired by
[tymons/buzz-based-anomaly](https://github.com/tymons/buzz-based-anomaly),
adapted for this dataset's layout and this project's requirements.

---

## 2. File Structure & Pipeline

| File | Role | Key output(s) |
|---|---|---|
| `explore_data.py` | Auto-detects class folders under `./data/`, counts files per class, samples audio metadata (sample rate, duration, channels), plots class distribution + an example Mel spectrogram | `output/distribution_classes.png`, `output/mel_spectrogram_bee.png` |
| `prepare_features.py` | Loads audio (mono-forced, fixed-length), extracts MFCC (20 coeffs), splits `Bee` 80/10/10, reserves **all** `NoBee`/`Missing Queen` for evaluation, z-score normalizes using **Bee-train stats only** | `processed_data/X_bee_{train,val,test}.npy`, `X_nobee.npy`, `X_missing_queen.npy`, `mfcc_mean.npy`, `mfcc_std.npy`, `manifest.json` |
| `train_conv2d_autoencoder.py` | **Architecture 1** — fully-convolutional Conv2D Autoencoder, dynamic padding (shape-agnostic) | `saved_models/conv2d_autoencoder.pt`, `output/loss_curve_conv2d_autoencoder.png` |
| `train_vae.py` | **Architecture 2** — Conv2D VAE (fixed input shape, dense μ/logvar bottleneck), MSE + β·KL loss | `saved_models/vae_autoencoder.pt`, `output/loss_curve_vae.png` |
| `train_contrastive_autoencoder.py` | **Architecture 3, winner** — Contrastive Autoencoder (self-supervised, SimCLR-style: denoising reconstruction + NT-Xent on augmented Bee-only views — see § 6 for why this deviates from a "real negatives" contrastive design) | `saved_models/contrastive_autoencoder.pt`, `output/loss_curve_contrastive_autoencoder.png` |
| `evaluate_all.py` | Loads all 3 checkpoints, computes reconstruction MSE on the shared held-out set, calibrates thresholds, computes ROC-AUC/PR-AUC/Precision/Recall/F1, designates the winning architecture | `output/roc_curves_comparison.png`, `output/pr_curves_comparison.png`, `output/reconstruction_errors_dist.png`, `processed_data/val_errors_bee.npy` |
| `main.py` | FastAPI service serving the winning model (Contrastive AE) for real-time inference | REST API — see § 4 |

**Pipeline order:**

```
explore_data.py → prepare_features.py → {train_conv2d_autoencoder, train_vae, train_contrastive_autoencoder}.py → evaluate_all.py → main.py
```


---

## 3. Model Evaluation & Results

Evaluated on the full held-out set: `Bee-val`=547, `Bee-test`=548,
`NoBee`=3,458, `Missing Queen`=4,861. Threshold = 80th percentile of
`Bee-val` reconstruction error (leak-free — calibrated without ever looking
at anomaly labels; see § 6 for why 80 was chosen over the initial default
of 95).

| Architecture | ROC-AUC | PR-AUC | Threshold (P80) | Precision | Recall | F1 | Recall (NoBee) | Recall (Missing Queen) | Inference (ms/sample) | Checkpoint Size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Conv2D Autoencoder | 0.7223 | 0.9766 | 0.01054 | 0.977 | 0.578 | 0.726 | 0.515 | 0.622 | 0.035 | 197 KB |
| Conv2D VAE | 0.8421 | 0.9837 | 0.47499 | 0.983 | 0.810 | 0.888 | 0.685 | 0.898 | 0.030 | 3,384 KB |
| **Contrastive AE** 🏆 | **0.8929** | **0.9919** | 0.04545 | 0.983 | 0.853 | 0.913 | 0.741 | 0.933 | 0.029 | 223 KB |
### Comparative Evaluation Visualizations

<p align="center">
  <img src="output/roc_curves_comparison.png" height="280" alt="ROC Curves" />
  <img src="output/reconstruction_errors_dist.png" height="280" alt="Reconstruction Error Distribution" />
</p>

*The ROC curves clearly demonstrate the superiority of the **Contrastive Autoencoder** (AUC = 0.8929) across all threshold levels, while the error distribution shows a clear separation between normal `Bee` sounds and anomalous audio (`Missing Queen` & `NoBee`).*
#### Winning Model Training Curve (Contrastive AE)

![Contrastive AE Loss Curve](output/loss_curve_contrastive_autoencoder.png)

**Winner: Contrastive Autoencoder** — best ROC-AUC (the threshold-independent,
fairest comparison metric across architectures with very different error
scales), best recall at a fixed, leak-free threshold, and among the smallest
+ fastest checkpoints. It's the model served by `main.py`.

### Key finding: training loss is not a proxy for anomaly-detection quality

Conv2D Autoencoder had by far the **lowest training loss** (0.009686 —
roughly 50x lower than the other two) but the **worst evaluation performance
on every metric** (ROC-AUC 0.72 vs. 0.84–0.89 for the others). A model that
becomes extremely good at reconstructing exactly the training distribution
doesn't necessarily learn representations general enough to separate normal
from anomalous inputs — it may be overfitting to reconstruct `Bee` audio
"too precisely," including quirks that don't generalize. Don't select an
architecture on training loss alone.

### Interpretation caveat: PR-AUC and Precision are inflated by class imbalance

Anomalies make up **93.8%** of the test set (8,319 of 8,867 samples) — a
direct consequence of reserving *all* `NoBee`/`Missing Queen` for evaluation
while only 10% of `Bee` is. This compresses PR-AUC into a narrow, less
discriminating band (0.977–0.994 across all three models) and makes
precision look uniformly strong (0.977–0.983) regardless of architecture
quality. **ROC-AUC is unaffected by this and is the more informative metric
here** — it shows a much clearer, 17-point spread between the best and worst
architecture.

### Secondary threshold: F1-optimal (calibrated on the test set — optimistic)

| Architecture | Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Conv2D Autoencoder | 0.00620 | 0.938 | 1.000 | 0.968 |
| Conv2D VAE | 0.35242 | 0.939 | 1.000 | 0.968 |
| Contrastive AE | 0.03378 | 0.940 | 0.998 | 0.968 |

This threshold is searched directly on the test set, so it's an optimistic
upper bound rather than a generalization estimate (same caveat as any
test-calibrated threshold) — kept here for reference only, not used in
production. Notably all three converge to F1 ≈ 0.968, which the leak-free
Table above does not — a good illustration of how much a threshold choice
alone can mask real architecture differences.

### Confidence score examples (Contrastive AE, production threshold = 0.04545)

| Input | Reconstruction MSE | `confidence_normal` | `confidence_anomaly` |
|---|---:|---:|---:|
| Bee (median) | 0.03922 | 68.33% | 31.67% |
| NoBee (median) | 0.05815 | 17.27% | 82.73% |
| Missing Queen (median) | 0.07266 | 3.37% | 96.63% |

---

## 4. FastAPI Endpoints

Base URL: `http://<host>:8000` (default `uvicorn` port).

### `GET /health`

Returns service status and the currently active threshold (recomputed at
startup from `val_errors_bee.npy`, never hardcoded).

**Example response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "architecture": "contrastive_conv2d_autoencoder",
  "threshold": 0.04545,
  "percentile": 80.0,
  "device": "cuda",
  "best_epoch": 100
}
```

### `POST /predict/anomaly`

Accepts a single `.wav` file upload (multipart/form-data) and returns the
predicted class with confidence scores.

**Example request:**
```bash
curl -X POST "http://localhost:8000/predict/anomaly" \
  -F "file=@hive_clip.wav"
```

**Example response (200 OK):**
```json
{
  "filename": "hive_clip.wav",
  "status": "ANOMALY",
  "reconstruction_mse": 0.05815,
  "threshold_used": 0.04545,
  "confidence_normal": 17.27,
  "confidence_anomaly": 82.73,
  "warning": null
}
```

`status` is `"NORMAL"` or `"ANOMALY"` — binary only (see § 1). `warning` is
populated (non-`null`) when the source audio has an unusually low sample
rate, without rejecting the request.

**Error responses:**

| Code | Condition |
|---:|---|
| `400` | Empty file, non-`.wav` extension, or file exceeds 10 MB |
| `422` | Audio file unreadable / corrupted |
| `503` | Model not loaded (should not occur — startup fails fast if loading fails) |
| `500` | Unexpected server error (generic message, no internal details leaked) |

Interactive docs (Swagger UI) available at `/docs` once the server is running.

---

## 5. Quickstart / Execution Workflow

Run from within `models/model2/`. Assumes the Kaggle dataset has already
been downloaded and unzipped into `./data/`.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Explore the dataset (sanity check: class counts, sample rate, duration)
python explore_data.py

# 3. Extract MFCC features, split Bee 80/10/10, reserve NoBee/Missing Queen for eval
python prepare_features.py
# -> populates ./processed_data/

# 4. Train all 3 candidate architectures
python train_conv2d_autoencoder.py
python train_vae.py
python train_contrastive_autoencoder.py
# -> populates ./saved_models/ and ./output/*.png (loss curves)

# 5. Compare all 3 architectures on the held-out test set
python evaluate_all.py
# -> prints the comparative table (§3), designates the winning model,
#    generates ROC/PR/error-distribution plots, and saves
#    processed_data/val_errors_bee.npy (calibration data for the API)

# 6. Serve the winning model (Contrastive AE) via the REST API
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4   # production
# uvicorn main:app --reload                                 # dev only

# 7. Verify
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict/anomaly -F "file=@some_clip.wav"
```

**Useful flags:**
- `python prepare_features.py --limit 200` — quick pipeline test on a small subset before running on all 13,792 files.
- `python evaluate_all.py --percentile 95` — re-evaluate at a stricter (fewer false positives) threshold.
- `HIVESENSE_PERCENTILE=90 uvicorn main:app ...` — change the serving threshold without retraining.

---

## 6. Known Limitations & Design Notes

Documented here for reproducibility and to avoid re-discovering the same
trade-offs later:

- **Contrastive AE uses self-supervised augmentation, not real anomalies.**
  The original framing called for `NoBee`/`Missing Queen` as "contrastive
  data," but those classes are reserved entirely for evaluation to keep the
  comparison across all 3 architectures leak-free. Architecture 3 instead
  uses SimCLR-style augmented views of `Bee` clips only.
- **Two threshold philosophies coexist:** a leak-free percentile threshold
  (calibrated only on `Bee-val`, never sees anomaly labels — used in
  production) and an F1-optimal threshold (searched directly on the test
  set — optimistic, reference only, § 3).
- **Percentile 80 (vs. the initial default of 95)** was chosen by observing
  test-set performance during a threshold sweep (`find_best_percentile.py`)
  — this reintroduces a mild version of the same optimism bias as the
  F1-optimal threshold. Treat reported precision/recall at P80 as an
  upper-bound estimate, not a strict generalization guarantee.
- **`main.py` imports `ContrastiveConv2DAutoencoder` from
  `train_contrastive_autoencoder.py`**, which pulls in `matplotlib` as a
  transitive dependency (used there for loss-curve plotting) — unnecessary
  weight for a production API process. Extracting the model classes into a
  dependency-light `model_architectures.py` would be the clean fix if this
  is containerized for production.
- **VAE reconstruction uses `mu`, not a sampled `z`**, everywhere
  reconstruction error is computed — necessary for deterministic,
  reproducible anomaly scores.