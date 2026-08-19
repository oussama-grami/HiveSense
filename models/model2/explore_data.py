#!/usr/bin/env python3
"""
explore_data.py — Exploration du dataset "Beehive Buzz Anomalies"
====================================================================

Ce script :
  1. Parcourt automatiquement ./data/ et détecte l'organisation des classes
     (Bee / NoBee / Missing Queen), quelle que soit la casse ou la convention
     de nommage des dossiers (Missing Queen, missing_queen, MissingQueen, ...).
  2. Compte le nombre de fichiers audio (.wav / .mp3) par classe.
  3. Échantillonne des fichiers par classe et en extrait sample rate, durée
     et nombre de canaux via soundfile (rapide, métadonnées seules), avec
     repli sur librosa si un fichier n'est pas lisible directement.
  4. Génère deux graphiques dans ./output/ :
       - distribution_classes.png
       - mel_spectrogram_bee.png
  5. Affiche un résumé structuré dans la console.

Usage :
    python explore_data.py
    python explore_data.py --data-dir ./data --output-dir ./output --sample-size 40
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import soundfile as sf

try:
    import librosa
    import librosa.display
except ImportError:
    print("ERREUR : librosa n'est pas installé. Faites `pip install librosa`.")
    sys.exit(1)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg")

# Alias connus pour chaque classe canonique. La comparaison se fait sur une
# version "normalisée" (minuscules, sans espaces/tirets/underscores), donc
# "Missing Queen", "missing_queen", "MissingQueen" matchent tous "missingqueen".
CLASS_ALIASES: Dict[str, List[str]] = {
    "Bee": ["bee", "bees", "normal", "healthy", "queenpresent", "queen"],
    "NoBee": ["nobee", "no bee", "absent", "empty", "nohive"],
    "Missing Queen": [
        "missingqueen", "missing queen", "queenless", "noqueen",
        "no queen", "queenlost", "missingqueenbee",
    ],
}

# Ordre d'affichage forcé (au lieu de l'ordre alphabétique / de découverte)
CLASS_ORDER = ["Bee", "NoBee", "Missing Queen"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("explore_data")


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------

@dataclass
class AudioInfo:
    path: Path
    sample_rate: int
    duration_s: float
    channels: int


@dataclass
class ClassStats:
    name: str
    n_files: int = 0
    infos: List[AudioInfo] = None
    n_corrupted: int = 0

    def __post_init__(self):
        if self.infos is None:
            self.infos = []


# ---------------------------------------------------------------------------
# Détection de la structure des classes
# ---------------------------------------------------------------------------

def normalize(name: str) -> str:
    """Normalise un nom de dossier/fichier pour la comparaison d'alias."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def match_class(name: str) -> Optional[str]:
    """Tente de faire correspondre un nom (dossier ou fichier) à une classe
    canonique connue. Renvoie None si aucune correspondance."""
    norm = normalize(name)
    if not norm:
        return None
    for canonical, aliases in CLASS_ALIASES.items():
        for alias in aliases:
            if normalize(alias) == norm:
                return canonical
    return None


def match_class_substring(name: str) -> Optional[str]:
    """Version plus permissive : cherche si un alias apparaît comme
    sous-chaîne du nom normalisé (utile pour matcher des noms de fichiers
    comme 'bee_hive3_20210512.wav')."""
    norm = normalize(name)
    if not norm:
        return None
    # On teste les alias les plus longs/spécifiques en premier pour éviter
    # que "bee" ne matche accidentellement "missingqueenbee".
    candidates = []
    for canonical, aliases in CLASS_ALIASES.items():
        for alias in aliases:
            norm_alias = normalize(alias)
            if norm_alias and norm_alias in norm:
                candidates.append((len(norm_alias), canonical))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def discover_class_directories(root: Path) -> Dict[str, List[Path]]:
    """Parcourt récursivement `root` et repère tous les dossiers dont le nom
    correspond à une classe connue. Renvoie {classe: [dossiers trouvés]}."""
    matches: Dict[str, List[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_dir():
            canonical = match_class(path.name)
            if canonical:
                matches[canonical].append(path)
    return matches


def collect_audio_files(directories: List[Path]) -> List[Path]:
    """Récupère tous les fichiers audio (récursivement) sous une liste de
    dossiers, en évitant les doublons."""
    files = set()
    for d in directories:
        for ext in AUDIO_EXTENSIONS:
            files.update(d.rglob(f"*{ext}"))
            files.update(d.rglob(f"*{ext.upper()}"))
    return sorted(files)


def discover_by_filename(root: Path) -> Dict[str, List[Path]]:
    """Repli si aucune structure de dossiers par classe n'est trouvée :
    on classe les fichiers audio en se basant sur leur nom de fichier."""
    result: Dict[str, List[Path]] = defaultdict(list)
    unmatched = 0
    for ext in AUDIO_EXTENSIONS:
        for path in list(root.rglob(f"*{ext}")) + list(root.rglob(f"*{ext.upper()}")):
            canonical = match_class_substring(path.stem)
            if canonical:
                result[canonical].append(path)
            else:
                unmatched += 1
    if unmatched:
        logger.warning(
            f"{unmatched} fichier(s) audio n'ont pas pu être rattachés à une "
            f"classe via leur nom de fichier."
        )
    return result


def discover_dataset_structure(data_dir: Path) -> Dict[str, List[Path]]:
    """Stratégie de découverte en cascade :
    1) dossiers nommés d'après les classes (n'importe où sous data_dir)
    2) à défaut, classification par nom de fichier
    """
    logger.info(f"Recherche de la structure du dataset dans : {data_dir.resolve()}")

    class_dirs = discover_class_directories(data_dir)
    if class_dirs:
        logger.info("Structure détectée : dossiers organisés par classe.")
        for canonical, dirs in class_dirs.items():
            rel_dirs = [str(d.relative_to(data_dir)) for d in dirs]
            logger.info(f"  - Classe '{canonical}' trouvée dans : {rel_dirs}")
        class_files = {c: collect_audio_files(d) for c, d in class_dirs.items()}
    else:
        logger.warning(
            "Aucun dossier ne correspond aux noms de classes attendus. "
            "Tentative de classification par nom de fichier..."
        )
        class_files = discover_by_filename(data_dir)

    if not class_files:
        logger.error(
            "Impossible de détecter automatiquement la structure du dataset. "
            "Vérifiez que ./data/ contient bien des sous-dossiers ou des noms "
            "de fichiers correspondant à Bee / NoBee / Missing Queen."
        )
    return class_files


# ---------------------------------------------------------------------------
# Analyse audio
# ---------------------------------------------------------------------------

def get_audio_info(path: Path) -> Optional[AudioInfo]:
    """Extrait sample rate / durée / nb de canaux d'un fichier audio.
    Utilise soundfile (rapide, ne décode pas tout le signal) avec repli sur
    librosa pour les formats non supportés (ex : certains .mp3). Renvoie None
    si le fichier est corrompu ou illisible."""
    try:
        info = sf.info(str(path))
        if info.samplerate <= 0 or info.frames <= 0:
            raise ValueError("métadonnées invalides (sample rate ou durée nulle)")
        duration = info.frames / info.samplerate
        return AudioInfo(path=path, sample_rate=info.samplerate,
                          duration_s=duration, channels=info.channels)
    except Exception as e_sf:
        try:
            y, sr = librosa.load(str(path), sr=None, mono=False)
            channels = 1 if y.ndim == 1 else y.shape[0]
            duration = librosa.get_duration(y=y, sr=sr)
            return AudioInfo(path=path, sample_rate=sr, duration_s=duration,
                              channels=channels)
        except Exception as e_librosa:
            logger.warning(
                f"Fichier corrompu ou illisible, ignoré : {path.name} "
                f"(soundfile: {e_sf} | librosa: {e_librosa})"
            )
            return None


def analyze_class_sample(class_name: str, files: List[Path],
                          sample_size: int, rng: random.Random) -> ClassStats:
    """Échantillonne jusqu'à `sample_size` fichiers d'une classe et en
    extrait les métadonnées audio."""
    stats = ClassStats(name=class_name, n_files=len(files))
    if not files:
        return stats

    sample = files if len(files) <= sample_size else rng.sample(files, sample_size)
    for path in sample:
        info = get_audio_info(path)
        if info is not None:
            stats.infos.append(info)
        else:
            stats.n_corrupted += 1
    return stats


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_class_distribution(class_counts: Dict[str, int], output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    ordered_names = [c for c in CLASS_ORDER if c in class_counts] + \
                     [c for c in class_counts if c not in CLASS_ORDER]
    values = [class_counts[c] for c in ordered_names]

    palette = sns.color_palette("viridis", n_colors=len(ordered_names))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(ordered_names, values, color=palette, edgecolor="black", linewidth=0.6)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{value:,}".replace(",", " "), ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_title("Distribution des classes — Beehive Buzz Anomalies",
                  fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Classe", fontsize=12)
    ax.set_ylabel("Nombre de fichiers audio", fontsize=12)
    ax.margins(y=0.12)
    sns.despine()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Graphique sauvegardé : {output_path}")


def plot_mel_spectrogram_example(bee_files: List[Path], output_path: Path,
                                  rng: random.Random, max_duration_s: float = 10.0) -> bool:
    """Cherche un fichier 'Bee' valide, calcule et sauvegarde son
    spectrogramme log-Mel. Renvoie True en cas de succès."""
    if not bee_files:
        logger.warning("Aucun fichier 'Bee' disponible : spectrogramme Mel ignoré.")
        return False

    candidates = bee_files[:]
    rng.shuffle(candidates)

    for path in candidates:
        try:
            y, sr = librosa.load(str(path), sr=None, duration=max_duration_s)
            if y.size == 0:
                continue
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr / 2)
            mel_db = librosa.power_to_db(mel, ref=np.max)

            fig, ax = plt.subplots(figsize=(10, 4.5))
            img = librosa.display.specshow(
                mel_db, sr=sr, x_axis="time", y_axis="mel", fmax=sr / 2,
                cmap="magma", ax=ax,
            )
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            ax.set_title(f"Spectrogramme Log-Mel — classe 'Bee' ({path.name})",
                          fontsize=12, fontweight="bold")
            fig.tight_layout()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            logger.info(f"Graphique sauvegardé : {output_path} (source : {path.name})")
            return True
        except Exception as e:
            logger.warning(f"Échec du chargement pour spectrogramme ({path.name}) : {e}")
            continue

    logger.error("Aucun fichier 'Bee' n'a pu être chargé pour le spectrogramme.")
    return False


# ---------------------------------------------------------------------------
# Résumé console
# ---------------------------------------------------------------------------

def fmt_duration(seconds: float) -> str:
    return f"{seconds:.2f} s"


def print_summary(class_files: Dict[str, List[Path]],
                   class_stats: Dict[str, ClassStats]) -> None:
    total_files = sum(len(v) for v in class_files.values())
    ordered_names = [c for c in CLASS_ORDER if c in class_files] + \
                     [c for c in class_files if c not in CLASS_ORDER]

    line = "=" * 72
    print("\n" + line)
    print(" RÉSUMÉ — Exploration du dataset Beehive Buzz Anomalies")
    print(line)

    print(f"\nTotal de fichiers audio détectés : {total_files:,}".replace(",", " "))
    print("\nRépartition par classe :")
    for name in ordered_names:
        n = len(class_files[name])
        pct = (n / total_files * 100) if total_files else 0
        print(f"  - {name:<15} : {n:>6,} fichiers  ({pct:5.1f}%)".replace(",", " "))

    print("\n" + "-" * 72)
    print(" Caractéristiques audio (calculées sur un échantillon par classe)")
    print("-" * 72)

    for name in ordered_names:
        stats = class_stats.get(name)
        if stats is None or not stats.infos:
            print(f"\n[{name}] Aucune métadonnée disponible "
                  f"(0 fichier valide analysé, {stats.n_corrupted if stats else 0} corrompu(s)).")
            continue

        durations = [i.duration_s for i in stats.infos]
        sample_rates = [i.sample_rate for i in stats.infos]
        channels = [i.channels for i in stats.infos]
        sr_counts = Counter(sample_rates)
        ch_counts = Counter(channels)

        print(f"\n[{name}]  ({len(stats.infos)} fichier(s) analysé(s) sur "
              f"{stats.n_files} au total, {stats.n_corrupted} corrompu(s)/illisible(s))")
        print(f"  Sample rate : {dict(sr_counts)} Hz")
        print(f"  Canaux      : {dict(ch_counts)}  (1 = mono, 2 = stéréo)")
        print(f"  Durée       : moyenne={fmt_duration(np.mean(durations))} | "
              f"min={fmt_duration(min(durations))} | max={fmt_duration(max(durations))}")

    print("\n" + line)
    print(" Graphiques générés dans le dossier output/ :")
    print("   - distribution_classes.png")
    print("   - mel_spectrogram_bee.png")
    print(line + "\n")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exploration du dataset Beehive Buzz Anomalies (Modèle 2)."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data"),
                         help="Dossier racine du dataset décompressé (défaut : ./data)")
    parser.add_argument("--output-dir", type=Path, default=Path("./output"),
                         help="Dossier de sortie pour les graphiques (défaut : ./output)")
    parser.add_argument("--sample-size", type=int, default=30,
                         help="Nombre de fichiers échantillonnés par classe pour "
                              "l'analyse des métadonnées audio (défaut : 30)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Graine aléatoire pour l'échantillonnage (défaut : 42)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    data_dir: Path = args.data_dir
    output_dir: Path = args.output_dir

    if not data_dir.exists():
        logger.error(f"Le dossier de données n'existe pas : {data_dir.resolve()}")
        return 1
    if not data_dir.is_dir():
        logger.error(f"Le chemin fourni n'est pas un dossier : {data_dir.resolve()}")
        return 1

    # 1) & 2) Découverte de la structure et comptage par classe
    class_files = discover_dataset_structure(data_dir)
    if not class_files or all(len(v) == 0 for v in class_files.values()):
        logger.error("Aucun fichier audio exploitable trouvé. Arrêt.")
        return 1

    # 3) Analyse des métadonnées audio (échantillon)
    logger.info(f"Analyse d'un échantillon de {args.sample_size} fichier(s) par classe...")
    class_stats: Dict[str, ClassStats] = {}
    for class_name, files in class_files.items():
        class_stats[class_name] = analyze_class_sample(
            class_name, files, args.sample_size, rng
        )

    # 4) Graphiques
    class_counts = {name: len(files) for name, files in class_files.items()}
    plot_class_distribution(class_counts, output_dir / "distribution_classes.png")
    plot_mel_spectrogram_example(
        class_files.get("Bee", []), output_dir / "mel_spectrogram_bee.png", rng
    )

    # 5) Résumé console
    print_summary(class_files, class_stats)

    return 0


if __name__ == "__main__":
    sys.exit(main())