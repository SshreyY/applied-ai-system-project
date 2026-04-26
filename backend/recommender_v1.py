"""
V1 Music Recommender -- copied from Module 3 (ai110-module3show-musicrecommendersimulation-starter).

This is the original rule-based scoring logic preserved as the V1 baseline.
It is used directly by the classic_scorer tool (backend/tools/classic_scorer.py)
so the agent can call the original formula as one of its tools and the
V1-vs-Agent comparison scripts can run both systems on the same inputs.

Known limitations documented in the original model_card.md:
  - Binary genre matching: genre match is all-or-nothing (+2.0 or 0)
  - No soft genre similarity (lofi vs ambient both score 0 for a rock query)
  - Genre lock-in: the +2.0 genre bonus dominates, pushing same-genre songs
    to the top regardless of mood/energy fit
  - No diversity enforcement: top-5 can be all the same genre
  - No feedback loop: preferences are static per run
  - Cold-start: requires fully specified numeric profile (no natural language)
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import csv


@dataclass
class Song:
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file into a list of dicts."""
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["id"] = int(row["id"])
            row["energy"] = float(row["energy"])
            row["tempo_bpm"] = float(row["tempo_bpm"])
            row["valence"] = float(row["valence"])
            row["danceability"] = float(row["danceability"])
            row["acousticness"] = float(row["acousticness"])
            songs.append(row)
    return songs


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    V1 scoring formula. Scores a single song against user preferences.
    Returns (score, reasons) where score is out of 7.5.

    Weights:
      genre match:        +2.0  (binary -- exact string match only)
      mood match:         +1.5  (binary -- exact string match only)
      energy proximity:   up to +1.5
      valence proximity:  up to +1.0
      danceability prox:  up to +0.8
      acousticness prox:  up to +0.7
    """
    score = 0.0
    reasons = []

    if song["genre"] == user_prefs.get("genre"):
        score += 2.0
        reasons.append(f"genre match ({song['genre']}) +2.0")

    if song["mood"] == user_prefs.get("mood"):
        score += 1.5
        reasons.append(f"mood match ({song['mood']}) +1.5")

    energy_points = round((1 - abs(song["energy"] - user_prefs.get("energy", 0.5))) * 1.5, 2)
    score += energy_points
    reasons.append(f"energy proximity +{energy_points}")

    valence_points = round((1 - abs(song["valence"] - user_prefs.get("valence", 0.5))) * 1.0, 2)
    score += valence_points
    reasons.append(f"valence proximity +{valence_points}")

    dance_points = round((1 - abs(song["danceability"] - user_prefs.get("danceability", 0.5))) * 0.8, 2)
    score += dance_points
    reasons.append(f"danceability proximity +{dance_points}")

    acoustic_points = round((1 - abs(song["acousticness"] - user_prefs.get("acousticness", 0.5))) * 0.7, 2)
    score += acoustic_points
    reasons.append(f"acousticness proximity +{acoustic_points}")

    return (round(score, 2), reasons)


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Score every song in the catalog, sort descending, return top k.
    Return format: list of (song_dict, score, explanation_string)
    """
    scored = [(song, *score_song(user_prefs, song)) for song in songs]
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    return [(song, score, ", ".join(reasons)) for song, score, reasons in ranked[:k]]


# --- Original 4 user profiles from Module 3 ---

V1_PROFILES = {
    "High-Energy Pop": {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.90,
        "valence": 0.85,
        "danceability": 0.85,
        "acousticness": 0.10,
    },
    "Chill Lofi": {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.35,
        "valence": 0.60,
        "danceability": 0.55,
        "acousticness": 0.80,
    },
    "Deep Intense Rock": {
        "genre": "rock",
        "mood": "intense",
        "energy": 0.92,
        "valence": 0.40,
        "danceability": 0.60,
        "acousticness": 0.08,
    },
    "Conflicted Vibe (high energy + chill mood)": {
        "genre": "ambient",
        "mood": "chill",
        "energy": 0.88,
        "valence": 0.65,
        "danceability": 0.75,
        "acousticness": 0.50,
    },
}
