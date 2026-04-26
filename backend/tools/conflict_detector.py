"""
detect_preference_conflicts tool -- finds contradictory preferences in a user profile.

Instead of silently failing or producing nonsensical results when a user asks for
something contradictory (e.g. 'high energy but chill mood'), the agent uses this
tool to detect the conflict and ask the user to clarify.
"""

from langchain_core.tools import tool


# Known attribute pairs that are typically contradictory
_CONFLICT_RULES = [
    {
        "name": "high_energy_chill_mood",
        "condition": lambda p: (
            p.get("energy") is not None and p.get("energy") > 0.75
            and p.get("mood") in ("chill", "relaxed")
        ),
        "description": (
            "High energy (>{energy:.2f}) combined with a '{mood}' mood is unusual. "
            "Most chill/relaxed songs have low energy (0.2–0.5). "
            "Did you mean high-tempo but calm (like focus music), "
            "or actually energetic and upbeat?"
        ),
    },
    {
        "name": "low_energy_intense_mood",
        "condition": lambda p: (
            p.get("energy") is not None and p.get("energy") < 0.35
            and p.get("mood") in ("intense", "angry", "energetic")
        ),
        "description": (
            "Low energy (<{energy:.2f}) combined with an '{mood}' mood rarely exists in the catalog. "
            "Intense/angry songs are typically high energy (0.8+). "
            "Did you mean something emotionally heavy but quiet, like melancholic?"
        ),
    },
    {
        "name": "high_acousticness_electronic_genre",
        "condition": lambda p: (
            p.get("acousticness") is not None and p.get("acousticness") > 0.7
            and p.get("genre") in ("electronic", "synthwave", "trap", "drum and bass", "metal")
        ),
        "description": (
            "High acousticness (>{acousticness:.2f}) with genre '{genre}' is contradictory — "
            "{genre} music is almost entirely electronic/synthetic. "
            "Did you mean a different genre, or are you open to acoustic alternatives?"
        ),
    },
    {
        "name": "low_acousticness_acoustic_genre",
        "condition": lambda p: (
            p.get("acousticness") is not None and p.get("acousticness") < 0.3
            and p.get("genre") in ("folk", "classical", "indie folk", "bossa nova", "blues", "country")
        ),
        "description": (
            "Low acousticness (<{acousticness:.2f}) with genre '{genre}' is unusual — "
            "{genre} is typically very acoustic. "
            "Did you want an electric/produced version of this style?"
        ),
    },
    {
        "name": "romantic_mood_angry_genre",
        "condition": lambda p: (
            p.get("mood") == "romantic"
            and p.get("genre") in ("metal", "punk", "drum and bass")
        ),
        "description": (
            "A 'romantic' mood with genre '{genre}' is an unusual combination. "
            "This combination barely exists in the catalog. "
            "Did you mean something passionate and intense, or actually romantic?"
        ),
    },
]


@tool
def detect_preference_conflicts(
    genre: str | None = None,
    mood: str | None = None,
    energy: float | None = None,
    valence: float | None = None,
    danceability: float | None = None,
    acousticness: float | None = None,
) -> dict:
    """
    Check a user profile for contradictory music preferences.

    Use this tool after building the user profile but before searching for songs.
    If conflicts are detected, ask the user to clarify rather than silently
    producing poor recommendations.

    Args:
        genre: User's preferred genre (optional).
        mood: User's preferred mood (optional).
        energy: Target energy level 0.0–1.0 (optional).
        valence: Target valence 0.0–1.0 (optional).
        danceability: Target danceability 0.0–1.0 (optional).
        acousticness: Target acousticness 0.0–1.0 (optional).

    Returns:
        A dict with 'has_conflicts' (bool), 'conflicts' (list of descriptions),
        and 'clarification_questions' to ask the user.
    """
    profile = {
        "genre": genre,
        "mood": mood,
        "energy": energy,
        "valence": valence,
        "danceability": danceability,
        "acousticness": acousticness,
    }

    conflicts = []
    questions = []

    for rule in _CONFLICT_RULES:
        if rule["condition"](profile):
            description = rule["description"].format(**{k: v for k, v in profile.items() if v is not None})
            conflicts.append({"name": rule["name"], "description": description})
            questions.append(f"Clarification needed: {description}")

    return {
        "has_conflicts": len(conflicts) > 0,
        "conflicts": conflicts,
        "clarification_questions": questions,
        "profile_checked": {k: v for k, v in profile.items() if v is not None},
    }
