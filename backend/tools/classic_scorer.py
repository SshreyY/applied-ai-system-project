"""
score_song_classic tool -- wraps the original V1 score_song formula as a tool.

Preserving V1 inside the new system serves two purposes:
1. The agent can call it to get a baseline score for any song, then compare
   its own reasoning against the formula output.
2. The evals/compare_v1_v2.py script can run both systems on the same inputs
   for a head-to-head comparison.
"""

import csv
import os
from langchain_core.tools import tool
from backend.recommender_v1 import score_song, recommend_songs, load_songs

CATALOG_PATH = os.getenv("CATALOG_PATH", "backend/data/songs.csv")

_catalog: list[dict] | None = None


def _get_catalog() -> list[dict]:
    global _catalog
    if _catalog is None:
        _catalog = load_songs(CATALOG_PATH)
    return _catalog


@tool
def score_song_classic(
    genre: str,
    mood: str,
    energy: float,
    valence: float,
    danceability: float,
    acousticness: float,
    top_k: int = 5,
) -> dict:
    """
    Score and rank songs using the original V1 rule-based formula from Module 3.

    Use this tool to:
    - Get a baseline recommendation to compare against your own reasoning
    - Check what the old formula would have recommended for a given profile
    - Use as a tiebreaker when two songs seem equally good

    The V1 formula scores songs out of 7.5 using weighted proximity:
      genre match (exact): +2.0
      mood match (exact):  +1.5
      energy proximity:    up to +1.5
      valence proximity:   up to +1.0
      danceability prox:   up to +0.8
      acousticness prox:   up to +0.7

    Args:
        genre: User's preferred genre (exact string match).
        mood: User's preferred mood (exact string match).
        energy: Target energy level (0.0–1.0).
        valence: Target valence / positivity (0.0–1.0).
        danceability: Target danceability (0.0–1.0).
        acousticness: Target acousticness (0.0–1.0).
        top_k: Number of top songs to return (default 5).

    Returns:
        A dict with 'recommendations' (list of songs with v1_score and reasons)
        and 'formula_note' explaining V1 limitations.
    """
    user_prefs = {
        "genre": genre,
        "mood": mood,
        "energy": energy,
        "valence": valence,
        "danceability": danceability,
        "acousticness": acousticness,
    }

    try:
        catalog = _get_catalog()
    except FileNotFoundError:
        return {"recommendations": [], "error": f"Catalog not found at {CATALOG_PATH}"}

    ranked = recommend_songs(user_prefs, catalog, k=top_k)

    recommendations = []
    for song, score, explanation in ranked:
        recommendations.append({
            "id": song["id"],
            "title": song["title"],
            "artist": song["artist"],
            "genre": song["genre"],
            "mood": song["mood"],
            "v1_score": score,
            "v1_max_score": 7.5,
            "reasons": explanation,
        })

    return {
        "recommendations": recommendations,
        "formula_note": (
            "V1 uses binary genre/mood matching (+2.0/+1.5 only if exact string match). "
            "Genre lock-in is a known bias: songs with a matching genre always outscore "
            "cross-genre songs even when other attributes are a much better fit."
        ),
    }
