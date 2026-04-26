"""
lookup_genre_info tool -- soft genre similarity lookup from genre_knowledge.json.

Fixes the V1 binary genre matching problem: instead of only matching exact genre
strings, the agent can look up related genres and their typical attributes and
widen the search accordingly.
"""

import json
import os
from langchain_core.tools import tool

GENRE_KNOWLEDGE_PATH = os.getenv("GENRE_KNOWLEDGE_PATH", "backend/data/genre_knowledge.json")

_genre_data: dict | None = None


def _load_genre_data() -> dict:
    global _genre_data
    if _genre_data is None:
        with open(GENRE_KNOWLEDGE_PATH, encoding="utf-8") as f:
            _genre_data = json.load(f)
    return _genre_data


@tool
def lookup_genre_info(genre: str) -> dict:
    """
    Look up a genre's similar genres and typical audio attribute ranges.

    Use this tool to:
    - Find genres that are sonically similar to what the user asked for
      (e.g. 'lofi' → ['ambient', 'jazz', 'indie folk', 'bossa nova'])
    - Get typical energy, tempo, acousticness ranges for a genre so you
      can set better attribute filters when calling catalog_search
    - Understand what a genre sounds like before recommending it

    Args:
        genre: The genre name to look up (e.g. 'lofi', 'hip-hop', 'k-pop').

    Returns:
        A dict with 'similar_genres', 'typical_attributes', and 'description'.
        Returns a 'not_found' flag with available genres if the genre is unknown.
    """
    data = _load_genre_data()
    genres = data.get("genres", {})

    genre_lower = genre.lower().strip()
    if genre_lower not in genres:
        return {
            "genre": genre,
            "not_found": True,
            "available_genres": sorted(genres.keys()),
            "suggestion": "Try one of the available genres, or use vibe_search for natural language matching.",
        }

    info = genres[genre_lower]
    return {
        "genre": genre_lower,
        "similar_genres": info.get("similar", []),
        "typical_attributes": info.get("typical_attributes", {}),
        "description": info.get("description", ""),
        "not_found": False,
    }
