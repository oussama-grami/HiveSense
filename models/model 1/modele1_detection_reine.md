# Modèle 1 — Détection de l'état de la reine

## Rôle

Analyser le son d'une ruche et déterminer dans quel état se trouve la reine. C'est le modèle le plus critique pour l'apiculteur : une reine absente ou rejetée met la survie de la colonie en danger.

## Classes de sortie

| Classe | Ce que ça signifie |
|---|---|
| Queen Not Present | Reine absente — intervention urgente requise |
| Queen Present — Newly Accepted | Nouvelle reine introduite et acceptée |
| Queen Present — Rejected | Reine présente mais rejetée par la colonie |
| Queen Present — Original | Reine originale présente — état normal |

## Dataset

**Smart Bee Colony Monitor (SBCM)**

- **Lien :** https://www.kaggle.com/datasets/annajyang/beehive-sounds
- **Volume :** 7 100 clips audio de 60 secondes
- **Matériel de collecte :** ESP32 + microphone INMP441 + capteur BME280 (identique à notre matériel)
- **Provenance :** colonies d'abeilles européennes en Californie (2022)
- **Licence :** publique, téléchargeable librement

## Modèle utilisé

**AI-Belha-Classifier** — NOSInovacao / Hugging Face

- **Lien :** https://huggingface.co/NOSInovacao/AI-Belha-Classifier
- **Architecture :** YAMNet (MobileNet pour audio) + tête de classification fine-tunée
- **Approche :** Transfer learning — YAMNet pré-entraîné sur AudioSet (521 classes), tête FC re-entraînée sur les 4 classes de la reine
- **Framework :** TensorFlow
- **Licence :** MIT

## Performances actuelles (à améliorer)

| Classe | F1-Score |
|---|---|
| Queen Not Present | 0.58 |
| Queen Present — Newly Accepted | 0.82 |
| Queen Present — Rejected | 0.72 |
| Queen Present — Original | 0.51 |
| **Weighted F1 global** | **0.72** |

## Notre travail

- Fine-tuning pour améliorer les classes faibles (Queen Not Present et Queen Present Original)
- Data augmentation : time stretch, pitch shift, ajout de bruit ambiant
- Rééquilibrage des classes (class weighting)
- **Objectif :** dépasser Weighted F1 = 0.82
