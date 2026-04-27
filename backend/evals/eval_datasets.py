"""
Eval datasets for VibeFinder Agent.

Contains test cases built from:
- The 4 original Module 3 profiles (direct comparison with V1)
- New vibe-description cases (natural language → recommendations)
- Edge cases (contradictory prefs, empty profile, unknown genre, feedback loop)
"""

# -----------------------------------------------------------------------
# Original 4 Module 3 profiles -- used for V1 vs Agent comparison
# -----------------------------------------------------------------------

V1_PROFILES = [
    {
        "id": "high_energy_pop",
        "label": "High-Energy Pop",
        "message": "I want high energy pop music, something happy and danceable",
        "expected_profile": {
            "genre": "pop",
            "mood": "happy",
            "energy": 0.90,
            "valence": 0.85,
            "danceability": 0.85,
            "acousticness": 0.10,
        },
        "expected_top_genres": ["pop", "indie pop", "synthpop", "k-pop"],
        "should_not_include_genres": ["lofi", "classical", "ambient"],
    },
    {
        "id": "chill_lofi",
        "label": "Chill Lofi",
        "message": "I want chill lofi music for studying, something relaxed and acoustic",
        "expected_profile": {
            "genre": "lofi",
            "mood": "chill",
            "energy": 0.35,
            "valence": 0.60,
            "danceability": 0.55,
            "acousticness": 0.80,
        },
        "expected_top_genres": ["lofi", "ambient", "jazz", "indie folk"],
        "should_not_include_genres": ["metal", "drum and bass", "trap"],
    },
    {
        "id": "deep_intense_rock",
        "label": "Deep Intense Rock",
        "message": "I want intense rock music, heavy and aggressive",
        "expected_profile": {
            "genre": "rock",
            "mood": "intense",
            "energy": 0.92,
        },
        "expected_top_genres": ["rock", "metal"],
        "should_not_include_genres": ["lofi", "bossa nova", "ambient"],
    },
    {
        "id": "conflicted_vibe",
        "label": "Conflicted Vibe",
        "message": "I want ambient music but really high energy and danceable",
        "expected_profile": {
            "genre": "ambient",
            "mood": "chill",
            "energy": 0.88,
        },
        "expect_conflict_detection": True,
        "conflict_description": "High energy with ambient/chill mood is contradictory",
    },
]

# -----------------------------------------------------------------------
# New vibe-description cases (natural language situations)
# -----------------------------------------------------------------------

VIBE_CASES = [
    {
        "id": "late_night_studying",
        "label": "Late Night Studying",
        "message": "I'm studying at 2am, it's really quiet and I need to focus but stay awake",
        "expected_activity": "studying",
        "expected_top_genres": ["lofi", "ambient", "jazz", "classical"],
        "expected_mood_range": ["chill", "focused"],
    },
    {
        "id": "road_trip",
        "label": "Road Trip",
        "message": "I'm going on a road trip with friends, we want something fun and energetic",
        "expected_activity": "road_trip",
        "expected_top_genres": ["pop", "rock", "indie pop", "latin"],
        "expected_mood_range": ["happy", "energetic", "confident"],
    },
    {
        "id": "heartbreak",
        "label": "Heartbreak",
        "message": "I just went through a bad breakup and I'm feeling sad and lonely",
        "expected_activity": "heartbreak",
        "expected_top_genres": ["indie folk", "r&b", "blues", "soul"],
        "expected_mood_range": ["melancholic", "romantic"],
    },
    {
        "id": "gym_session",
        "label": "Gym Session",
        "message": "I'm about to hit the gym, I need something really intense to push through",
        "expected_activity": "working_out",
        "expected_top_genres": ["metal", "electronic", "hip-hop", "drum and bass"],
        "expected_mood_range": ["intense", "energetic", "angry"],
    },
    {
        "id": "sunday_morning",
        "label": "Sunday Morning",
        "message": "It's a lazy Sunday morning, I want something relaxed and feel-good",
        "expected_activity": "relaxing",
        "expected_top_genres": ["jazz", "folk", "lofi", "reggae"],
        "expected_mood_range": ["relaxed", "happy", "chill"],
    },
]

# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------

EDGE_CASES = [
    {
        "id": "empty_profile",
        "label": "Empty Profile (No Preferences)",
        "message": "I want some music",
        "expect_recommendations": True,
        "note": "Agent should return something reasonable even with no profile info",
    },
    {
        "id": "unknown_genre",
        "label": "Unknown Genre",
        "message": "I want some chiptune music",
        "expect_recommendations": True,
        "note": "Genre not in catalog — agent should fall back to vibe_search or similar genres",
    },
    {
        "id": "feedback_loop",
        "label": "Feedback Loop",
        "messages": [
            "I want chill lofi music",
            "Not that one, too acoustic. Something slightly more upbeat",
        ],
        "expect_exclusion_after_feedback": True,
        "note": "Second message should update profile and exclude previously recommended songs",
    },
    {
        "id": "contradictory_explicit",
        "label": "Explicit Contradiction",
        "message": "I want really loud aggressive metal music but also super chill and relaxing",
        "expect_conflict_detection": True,
        "note": "Agent should flag the contradiction and ask for clarification",
    },
    {
        "id": "off_topic",
        "label": "Off-Topic Message",
        "message": "What's the capital of France?",
        "expected_intent": "general_chat",
        "expect_recommendations": False,
    },
]

# -----------------------------------------------------------------------
# All cases combined
# -----------------------------------------------------------------------

ALL_CASES = V1_PROFILES + VIBE_CASES + EDGE_CASES
