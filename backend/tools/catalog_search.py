"""
catalog_search tool -- structured filter over the songs catalog.

The agent calls this when the user has clear attribute preferences
(genre, mood, energy range, etc.) and wants direct filtered results.
"""

import csv
import os
from typing import Optional
from langchain_core.tools import tool

CATALOG_PATH = os.getenv("CATALOG_PATH", "backend/data/songs.csv")


def _load_catalog(path: str) -> list[dict]:
    songs = []
    with open(path, newline="", encoding="utf-8") as f:
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


@tool
def catalog_search(
    genre: Optional[str] = None,
    mood: Optional[str] = None,
    min_energy: Optional[float] = None,
    max_energy: Optional[float] = None,
    min_valence: Optional[float] = None,
    max_valence: Optional[float] = None,
    min_danceability: Optional[float] = None,
    max_danceability: Optional[float] = None,
    min_acousticness: Optional[float] = None,
    max_acousticness: Optional[float] = None,
    exclude_ids: Optional[list[int]] = None,
    limit: int = 10,
) -> dict:
    """
    Search the song catalog using structured filters.

    Use this tool when you know specific genre, mood, or attribute ranges
    the user wants. All parameters are optional — omit any you don't need.

    Args:
        genre: Genre string to match (e.g. 'lofi', 'pop', 'rock').
        mood: Mood string to match (e.g. 'chill', 'happy', 'intense').
        min_energy: Minimum energy level (0.0–1.0).
        max_energy: Maximum energy level (0.0–1.0).
        min_valence: Minimum valence / positivity (0.0–1.0).
        max_valence: Maximum valence / positivity (0.0–1.0).
        min_danceability: Minimum danceability (0.0–1.0).
        max_danceability: Maximum danceability (0.0–1.0).
        min_acousticness: Minimum acousticness (0.0–1.0).
        max_acousticness: Maximum acousticness (0.0–1.0).
        exclude_ids: List of song IDs to exclude (already shown / disliked).
        limit: Maximum number of results to return (default 10).

    Returns:
        A dict with 'songs' (list of matching song dicts) and 'total_found' count.
    """
    try:
        songs = _load_catalog(CATALOG_PATH)
    except FileNotFoundError:
        return {"songs": [], "total_found": 0, "error": f"Catalog not found at {CATALOG_PATH}"}

    exclude = set(exclude_ids or [])
    results = []

    for song in songs:
        if song["id"] in exclude:
            continue
        if genre and song["genre"].lower() != genre.lower():
            continue
        if mood and song["mood"].lower() != mood.lower():
            continue
        if min_energy is not None and song["energy"] < min_energy:
            continue
        if max_energy is not None and song["energy"] > max_energy:
            continue
        if min_valence is not None and song["valence"] < min_valence:
            continue
        if max_valence is not None and song["valence"] > max_valence:
            continue
        if min_danceability is not None and song["danceability"] < min_danceability:
            continue
        if max_danceability is not None and song["danceability"] > max_danceability:
            continue
        if min_acousticness is not None and song["acousticness"] < min_acousticness:
            continue
        if max_acousticness is not None and song["acousticness"] > max_acousticness:
            continue
        results.append(song)

    return {
        "songs": results[:limit],
        "total_found": len(results),
    }
