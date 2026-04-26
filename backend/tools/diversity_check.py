"""
check_diversity tool -- analyzes a list of candidate songs for diversity.

Fixes V1's no-diversity-enforcement problem: the agent can inspect its
candidate list before finalizing recommendations and detect genre lock-in,
narrow mood range, or energy clustering.
"""

from langchain_core.tools import tool


@tool
def check_diversity(songs: list[dict]) -> dict:
    """
    Analyze a list of candidate songs for genre, mood, and energy diversity.

    Use this tool after collecting candidates but before finalizing recommendations.
    If diversity is low, consider broadening your search with vibe_search or
    catalog_search using different genre/mood filters.

    Args:
        songs: List of song dicts. Each must have at least 'id', 'title',
               'genre', 'mood', and 'energy' keys.

    Returns:
        A dict with diversity metrics and a 'passed' flag. If passed is False,
        'issues' lists specific problems (e.g. genre lock-in, narrow energy range).
    """
    if not songs:
        return {
            "passed": False,
            "issues": ["No songs provided — candidate list is empty."],
            "metrics": {},
        }

    genres = [s.get("genre", "unknown") for s in songs]
    moods = [s.get("mood", "unknown") for s in songs]
    energies = [float(s.get("energy", 0.5)) for s in songs]

    unique_genres = list(set(genres))
    unique_moods = list(set(moods))
    energy_range = round(max(energies) - min(energies), 3)
    avg_energy = round(sum(energies) / len(energies), 3)

    dominant_genre = max(set(genres), key=genres.count)
    dominant_genre_pct = round(genres.count(dominant_genre) / len(genres), 2)

    dominant_mood = max(set(moods), key=moods.count)
    dominant_mood_pct = round(moods.count(dominant_mood) / len(moods), 2)

    issues = []

    if dominant_genre_pct >= 0.8 and len(songs) >= 3:
        issues.append(
            f"Genre lock-in detected: {dominant_genre_pct*100:.0f}% of songs are '{dominant_genre}'. "
            f"Consider adding songs from: {', '.join(g for g in unique_genres if g != dominant_genre) or 'other genres'}."
        )

    if len(unique_genres) == 1 and len(songs) >= 3:
        issues.append(
            f"All {len(songs)} songs are the same genre ('{unique_genres[0]}'). "
            "Use lookup_genre_info to find similar genres and broaden results."
        )

    if len(unique_moods) == 1 and len(songs) >= 3:
        issues.append(
            f"All songs share the same mood ('{unique_moods[0]}'). "
            "Consider including songs with adjacent moods for variety."
        )

    if energy_range < 0.15 and len(songs) >= 3:
        issues.append(
            f"Narrow energy range ({min(energies):.2f}–{max(energies):.2f}). "
            "All songs have very similar energy levels — some variety may improve recommendations."
        )

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "metrics": {
            "total_songs": len(songs),
            "unique_genres": unique_genres,
            "unique_moods": unique_moods,
            "dominant_genre": dominant_genre,
            "dominant_genre_pct": dominant_genre_pct,
            "dominant_mood": dominant_mood,
            "dominant_mood_pct": dominant_mood_pct,
            "energy_min": min(energies),
            "energy_max": max(energies),
            "energy_range": energy_range,
            "avg_energy": avg_energy,
        },
    }
