"""
lookup_activity_context tool -- maps situation descriptions to music attributes.

Fixes the V1 cold-start problem: instead of requiring a fully specified numeric
profile, the agent can describe a situation (e.g. 'studying', 'working out')
and get back suggested attribute ranges and preferred genres to seed the search.
"""

import json
import os
from langchain_core.tools import tool

ACTIVITY_MOODS_PATH = os.getenv("ACTIVITY_MOODS_PATH", "backend/data/activity_moods.json")

_activity_data: dict | None = None


def _load_activity_data() -> dict:
    global _activity_data
    if _activity_data is None:
        with open(ACTIVITY_MOODS_PATH, encoding="utf-8") as f:
            _activity_data = json.load(f)
    return _activity_data


@tool
def lookup_activity_context(activity: str) -> dict:
    """
    Look up suggested music attributes for a given activity or situation.

    Use this tool when the user describes what they're doing or how they feel
    rather than specifying music attributes directly. For example: 'studying',
    'working out', 'road trip', 'late night', 'heartbreak', 'romance'.

    Args:
        activity: Activity or situation keyword (e.g. 'studying', 'working_out',
                  'road_trip', 'late_night', 'party', 'relaxing', 'heartbreak',
                  'morning_routine', 'creative_work', 'romance').

    Returns:
        A dict with 'suggested_attributes' (energy/valence/etc ranges),
        'preferred_genres', 'preferred_moods', 'avoid_moods', and 'description'.
        Returns 'not_found' with available activities if no match.
    """
    data = _load_activity_data()
    activities = data.get("activities", {})

    query = activity.lower().strip().replace(" ", "_")

    # Direct key match first
    if query in activities:
        match = activities[query]
        return {
            "activity": query,
            "suggested_attributes": match.get("suggested_attributes", {}),
            "preferred_genres": match.get("preferred_genres", []),
            "preferred_moods": match.get("preferred_moods", []),
            "avoid_moods": match.get("avoid_moods", []),
            "description": match.get("description", ""),
            "not_found": False,
        }

    # Keyword scan fallback
    query_words = set(query.replace("_", " ").split())
    for activity_key, info in activities.items():
        keywords = set(kw.lower() for kw in info.get("keywords", []))
        if query_words & keywords:
            return {
                "activity": activity_key,
                "matched_via_keyword": True,
                "suggested_attributes": info.get("suggested_attributes", {}),
                "preferred_genres": info.get("preferred_genres", []),
                "preferred_moods": info.get("preferred_moods", []),
                "avoid_moods": info.get("avoid_moods", []),
                "description": info.get("description", ""),
                "not_found": False,
            }

    return {
        "activity": activity,
        "not_found": True,
        "available_activities": sorted(activities.keys()),
        "suggestion": "Try one of the available activities, or describe the vibe using vibe_search.",
    }
