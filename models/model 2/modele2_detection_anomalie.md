# Modèle 2 — Détection d'anomalies acoustiques

## Rôle

Détecter si le son de la ruche est normal (colonie active), absent (colonie morte ou abandonnée), ou anormal (reine manquante). Contrairement au Modèle 1 qui identifie précisément l'état de la reine, ce modèle sert de **premier filtre d'alerte** : il détecte que quelque chose ne va pas, même sans savoir exactement quoi.

## Classes de sortie

| Classe | Ce que ça signifie |
|---|---|
| Bee | Son normal — colonie active et saine |
| NoBee | Absence de son — colonie inactive, abandonnée ou morte |
| Missing Queen | Anomalie acoustique indiquant l'absence de la reine |

## Dataset

**Beehive Buzz Anomalies**

- **Lien :** https://www.kaggle.com/datasets/yevheniiklymenko/beehive-buzz-anomalies
- **Description :** version traitée et segmentée du dataset TBON (To Bee or Not to Bee), combinant les données NU-Hive (8 ruches) et OSBH (6 ruches)
- **Source originale :** Zenodo DOI 10.5281/zenodo.1321278
- **Licence :** Creative Commons Attribution 4.0

## Modèle utilisé

**buzz-based-anomaly** — tymons / GitHub

- **Lien :** https://github.com/tymons/buzz-based-anomaly
- **Approche :** Détection d'anomalies par autoencoder — le modèle apprend à reconstruire les sons normaux (Bee). Une erreur de reconstruction élevée signale une anomalie (NoBee ou Missing Queen).
- **Architectures disponibles :** Autoencoder Conv2D, Variational Autoencoder (VAE), Contrastive Autoencoder
- **Features audio supportées :** MFCC, Mel-Spectrogramme, Spectrogramme, Périodogramme
- **Framework :** PyTorch

## Pourquoi un autoencoder plutôt qu'un classifieur classique ?

Un autoencoder n'a pas besoin d'exemples équilibrés de chaque classe anormale. Il apprend uniquement ce qu'est un son "normal" et signale tout ce qui s'en écarte — ce qui est adapté à notre situation où les anomalies sont rares et peu représentées dans les données.

## Notre travail

- Adapter le pipeline de données du repo au dataset Beehive Buzz Anomalies (structure différente du NU-Hive original ciblé par le repo)
- Comparer expérimentalement les 3 architectures (Autoencoder classique vs VAE vs Contrastive) sur les mêmes données
- Définir les seuils optimaux de détection par classe (courbes ROC + AUC)
- Évaluer la généralisation inter-ruches (test sur ruches non vues à l'entraînement)
