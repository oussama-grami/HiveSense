# 🐝 HiveSense — Surveillance Acoustique Intelligente des Ruches

> Système IoT + Deep Learning pour la détection précoce de l'état de la reine et des anomalies acoustiques dans les ruches d'abeilles.

![Flutter](https://img.shields.io/badge/Flutter-3.47-blue?logo=flutter)
![Python](https://img.shields.io/badge/Python-3.11-yellow?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-API-lightgrey?logo=flask)
![Firebase](https://img.shields.io/badge/Firebase-Auth-orange?logo=firebase)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Contexte

HiveSense est un projet de **recherche appliquée** développé dans le cadre d'un stage de fin d'études à **Think&Solution** (INSAT, Génie Logiciel, 2025–2026).

L'apiculture moderne fait face à deux événements critiques difficiles à détecter manuellement :
- 🚨 **La perte de reine** — la colonie devient orpheline et menace de s'effondrer.
- 🐝 **L'essaimage** — une partie de la colonie quitte définitivement la ruche.

HiveSense exploite les **signatures acoustiques** du bourdonnement des abeilles pour détecter ces états automatiquement, via un dispositif IoT autonome (ESP32 + microphone MEMS + 4G + énergie solaire) et une application mobile Flutter.

---

## 🏗️ Architecture du système

```
[Ruche]
   └── Microphone INMP441 (I2S)
          ↓
[Device IoT — ESP32 T-Call A7670E]
   ├── Capture audio 16 kHz
   ├── FFT + filtre 150–600 Hz
   ├── Extraction MFCC
   ├── Inférence TinyML (modèle embarqué)
   └── Transmission 4G / MQTT (si anomalie détectée)
          ↓
[Serveur Cloud — Python]
   ├── Modèle 1 : CNN → état de la reine (4 classes)
   ├── Modèle 2 : Autoencodeur → anomalie acoustique (3 classes)
   └── API Flask / FastAPI
          ↓
[Application Mobile — Flutter]
   ├── Dashboard multi-ruches temps réel
   ├── Analyse audio à la demande
   ├── Alertes et notifications
   └── Authentification Firebase (OTP SMS)
```

---

## 🤖 Modèles d'Intelligence Artificielle

### Modèle 1 — Détection de l'état de la reine

| Paramètre | Valeur |
|---|---|
| **Dataset** | SBCM — [Kaggle](https://www.kaggle.com/datasets/annajyang/beehive-sounds) |
| **Volume** | 7 100 fichiers → 85 200 fenêtres de 5s |
| **Architecture** | CNN 3 blocs Conv2D + BatchNorm + AdaptiveAvgPool |
| **Paramètres** | 961 540 |
| **GPU** | NVIDIA GeForce GTX 1650 (CUDA 12.4) |
| **Accuracy** | **91.5%** |
| **Weighted F1** | **0.9153** |
| **AUC-ROC** | **0.9893** |

**Classes de sortie :**
- `Queen Not Present` — reine absente (intervention urgente)
- `Queen Present - Newly Accepted` — nouvelle reine acceptée
- `Queen Present - Rejected` — reine rejetée
- `Queen Present - Original` — état normal

---

### Modèle 2 — Détection d'anomalies acoustiques

| Paramètre | Valeur |
|---|---|
| **Dataset** | Beehive Buzz Anomalies — [Kaggle](https://www.kaggle.com/datasets/yevheniiklymenko/beehive-buzz-anomalies) |
| **Volume** | 13 792 clips de 2s, 44 100 Hz |
| **Approche** | Détection d'anomalies par autoencodeur (non supervisée) |
| **Architectures testées** | Conv2D Autoencoder · VAE · Contrastive Autoencoder |
| **Features** | MFCC (n_mfcc=20, n_fft=2048, hop_length=512) |

**Classes de sortie :**
- `Bee` — activité normale
- `NoBee` — absence d'abeilles
- `Missing Queen` — anomalie acoustique (reine manquante)

---

## 📱 Application Mobile Flutter

- **Framework :** Flutter 3.47 (Dart) — iOS & Android
- **Authentification :** Firebase Phone Auth (OTP SMS)
- **State management :** Provider (HiveProvider + AuthProvider)
- **Navigation :** go_router avec GoRouterRefreshStream
- **Graphiques :** fl_chart
- **API :** dio + http

**Écrans disponibles :**
| Écran | Fonctionnalité |
|---|---|
| Dashboard | Vue d'ensemble toutes ruches, alertes, statuts |
| Détail ruche | Capteurs, résultats IA, historique graphique |
| Analyse audio | Upload WAV → prédiction Modèle 1 en temps réel |
| Statistiques | KPI, graphiques multi-lignes, heatmap activité |
| Alertes | Timeline, filtres par type, marquage lu |
| Paramètres | Seuils, URLs API, test connexion réel |
| Profil | Photo, informations, statistiques, déconnexion |

---

## 🔧 Matériel IoT

| Composant | Rôle |
|---|---|
| LilyGO T-Call A7670E | ESP32 + modem 4G LTE intégré |
| Microphone MEMS INMP441 | Capture audio I2S, 20 Hz–20 kHz |
| Panneau solaire Soshine 6W | Alimentation autonome |
| Batterie Li-ion 18650 | Stockage d'énergie tampon |
| Antenne GPS (U.FL) | Géolocalisation des ruches |
| Antenne 4G LTE (U.FL) | Connectivité réseau cellulaire |

**Connexions INMP441 → ESP32 :**
```
VDD → 3V3 | GND → GND | SCK → IO26 | WS → IO25 | SD → IO33 | L/R → GND
```

---

## 🚀 Installation et lancement

### Prérequis
- Python 3.11+
- Flutter 3.47+
- CUDA 12.4 (optionnel, pour GPU)

### Modèle 1 — API Flask

```bash
# Créer l'environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install torch torchaudio librosa numpy pandas scikit-learn flask flask-cors

# Lancer l'API
cd beeacoustic_model1
python api/app.py
```

L'API sera disponible sur `http://localhost:5000`

**Endpoints :**
```
GET  /health           → statut de l'API + métriques du modèle
POST /predict          → prédiction sur un fichier WAV
POST /predict/batch    → prédiction sur plusieurs fichiers
```

### Application Flutter

```bash
cd beeacoustic_app
flutter pub get
flutter run -d chrome    # Développement web
flutter run -d android   # Déploiement Android
```

---

## 📊 Résultats du Modèle 1

| Classe | Précision | Recall | F1-Score |
|---|---|---|---|
| Queen Not Present | 0.881 | **0.943** | 0.911 |
| Newly Accepted | 0.827 | 0.873 | 0.849 |
| Rejected | 0.898 | 0.853 | 0.875 |
| Original | 0.958 | 0.945 | **0.952** |
| **Weighted avg** | **0.916** | **0.915** | **0.915** |

---

## 🔬 Contributions de recherche

- **R1** — Vérification primaire du dataset Nolasco 2019 : le chiffre de "5 336 clips Swarm" n'a pas pu être tracé à une source primaire confirmée (Zenodo DOI: 10.5281/zenodo.1321278 contient 78 enregistrements Bee/NoBee uniquement).
- **R2** — Identification du biais dataset/performance : les résultats 93–99% de la littérature proviennent exclusivement de données privées non reproductibles.
- **R3** — Aucun dataset public ne contient de classe essaimage validée avec des données terrain suffisantes.
- **R4** — Analyse chiffrée originale : Cloud pur = jusqu'à 4 920 Go/mois pour 60 ruches → inviable. Hybride ≈ 25 Go/mois → viable.
- **R5** — Justification quantitative de l'architecture Hybride à dominante Edge pour le contexte rural tunisien.

---

## 📁 Structure du projet

```
HiveSense/
├── beeacoustic_model1/          # Modèle 1 — CNN
│   ├── src/
│   │   ├── dataset.py           # Préparation données + fenêtres 5s
│   │   ├── model.py             # Architecture CNN PyTorch
│   │   ├── train.py             # Entraînement GPU
│   │   ├── evaluate.py          # Métriques + courbes ROC
│   │   └── predict.py           # Inférence ligne de commande
│   ├── api/
│   │   └── app.py               # API Flask
│   └── models/
│       └── best_model.pt        # Modèle sauvegardé
│
├── beeacoustic_model2/          # Modèle 2 — Autoencodeurs
│   ├── explore_data.py          # Exploration dataset
│   ├── prepare_features.py      # Extraction MFCC + split
│   ├── train_conv2d_autoencoder.py
│   ├── train_vae.py
│   ├── train_model3.py          # Contrastive Autoencoder
│   └── evaluate_all.py          # Évaluation comparative
│
└── beeacoustic_app/             # Application Flutter
    ├── lib/
    │   ├── core/                # Thème, constantes, router
    │   ├── models/              # HiveModel, SensorData, AIPrediction
    │   ├── providers/           # HiveProvider, AuthProvider
    │   ├── services/            # ApiService
    │   └── screens/             # 7 écrans principaux
    └── pubspec.yaml
```

---

## 👥 Équipe

| Membre | Rôle |
|---|---|
| **Oussema Guerami** | Modèle 1 (CNN)  |
| **Chadha Grami** | Modèle 2 (Autoencodeurs) |

**Encadrant :** Yasser Ben Nejma — Think&Solution  
**Établissement :** INSAT — Institut National des Sciences Appliquées et de Technologie  
**Université :** Université de Carthage  
**Année :** 2025–2026

---

## 📚 Références principales

- Dimitrios et al., *Performance Evaluation of Classification Algorithms to Detect Bee Swarming Events Using Sound*, Signals, 2022.
- Kontogiannis, *Beehive Smart Detector Device*, Sensors, 2024.
- Šerić et al., *Buzzing with Intelligence: A Systematic Review of Smart Beehive Technologies*, Sensors, 2025.
- Chen et al., *A Simple Framework for Contrastive Learning (SimCLR)*, ICML, 2020.
- Kingma & Welling, *Auto-Encoding Variational Bayes*, ICLR, 2014.

---

## 📄 Licence

Ce projet est développé dans le cadre d'un stage académique à Think&Solution. Tous droits réservés.
