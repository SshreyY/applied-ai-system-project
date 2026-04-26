"""
Unit tests for all 7 agent tools.

Run with: pytest backend/tests/test_tools.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.tools.catalog_search import catalog_search
from backend.tools.genre_knowledge import lookup_genre_info
from backend.tools.activity_context import lookup_activity_context
from backend.tools.classic_scorer import score_song_classic
from backend.tools.diversity_check import check_diversity
from backend.tools.conflict_detector import detect_preference_conflicts


# --- catalog_search ---

class TestCatalogSearch:
    def test_returns_songs(self):
        result = catalog_search.invoke({})
        assert "songs" in result
        assert len(result["songs"]) > 0

    def test_genre_filter(self):
        result = catalog_search.invoke({"genre": "lofi"})
        assert all(s["genre"] == "lofi" for s in result["songs"])

    def test_mood_filter(self):
        result = catalog_search.invoke({"mood": "chill"})
        assert all(s["mood"] == "chill" for s in result["songs"])

    def test_energy_range_filter(self):
        result = catalog_search.invoke({"min_energy": 0.8, "max_energy": 1.0})
        assert all(s["energy"] >= 0.8 for s in result["songs"])

    def test_exclude_ids(self):
        all_songs = catalog_search.invoke({})
        first_id = all_songs["songs"][0]["id"]
        result = catalog_search.invoke({"exclude_ids": [first_id]})
        ids = [s["id"] for s in result["songs"]]
        assert first_id not in ids

    def test_limit_respected(self):
        result = catalog_search.invoke({"limit": 3})
        assert len(result["songs"]) <= 3

    def test_no_results_returns_empty(self):
        result = catalog_search.invoke({"genre": "nonexistent_genre_xyz"})
        assert result["songs"] == []
        assert result["total_found"] == 0

    def test_combined_filters(self):
        result = catalog_search.invoke({"genre": "lofi", "mood": "chill", "max_energy": 0.5})
        for s in result["songs"]:
            assert s["genre"] == "lofi"
            assert s["mood"] == "chill"
            assert s["energy"] <= 0.5


# --- lookup_genre_info ---

class TestGenreKnowledge:
    def test_known_genre(self):
        result = lookup_genre_info.invoke({"genre": "lofi"})
        assert result["not_found"] is False
        assert "similar_genres" in result
        assert len(result["similar_genres"]) > 0
        assert "typical_attributes" in result

    def test_similar_genres_are_strings(self):
        result = lookup_genre_info.invoke({"genre": "rock"})
        assert all(isinstance(g, str) for g in result["similar_genres"])

    def test_unknown_genre_returns_not_found(self):
        result = lookup_genre_info.invoke({"genre": "completely_made_up_genre"})
        assert result["not_found"] is True
        assert "available_genres" in result

    def test_case_insensitive(self):
        result = lookup_genre_info.invoke({"genre": "LOFI"})
        assert result["not_found"] is False

    def test_hip_hop_similar_contains_trap(self):
        result = lookup_genre_info.invoke({"genre": "hip-hop"})
        assert "trap" in result["similar_genres"]


# --- lookup_activity_context ---

class TestActivityContext:
    def test_known_activity(self):
        result = lookup_activity_context.invoke({"activity": "studying"})
        assert result["not_found"] is False
        assert "suggested_attributes" in result
        assert "preferred_genres" in result

    def test_studying_avoids_intense(self):
        result = lookup_activity_context.invoke({"activity": "studying"})
        assert "intense" in result["avoid_moods"]

    def test_working_out_prefers_high_energy(self):
        result = lookup_activity_context.invoke({"activity": "working_out"})
        assert result["suggested_attributes"]["energy"][0] >= 0.7

    def test_unknown_activity_returns_not_found(self):
        result = lookup_activity_context.invoke({"activity": "underwater_basket_weaving"})
        assert result["not_found"] is True
        assert "available_activities" in result

    def test_keyword_fallback(self):
        result = lookup_activity_context.invoke({"activity": "gym"})
        assert result["not_found"] is False


# --- score_song_classic ---

class TestClassicScorer:
    def test_returns_recommendations(self):
        result = score_song_classic.invoke({
            "genre": "lofi", "mood": "chill", "energy": 0.35,
            "valence": 0.60, "danceability": 0.55, "acousticness": 0.80,
        })
        assert "recommendations" in result
        assert len(result["recommendations"]) > 0

    def test_scores_are_numeric(self):
        result = score_song_classic.invoke({
            "genre": "pop", "mood": "happy", "energy": 0.9,
            "valence": 0.85, "danceability": 0.85, "acousticness": 0.10,
        })
        for rec in result["recommendations"]:
            assert isinstance(rec["v1_score"], float)
            assert 0.0 <= rec["v1_score"] <= 7.5

    def test_genre_match_boosts_score(self):
        result_pop = score_song_classic.invoke({
            "genre": "pop", "mood": "happy", "energy": 0.82,
            "valence": 0.84, "danceability": 0.79, "acousticness": 0.18,
        })
        top_song = result_pop["recommendations"][0]
        assert top_song["genre"] == "pop"

    def test_top_k_respected(self):
        result = score_song_classic.invoke({
            "genre": "rock", "mood": "intense", "energy": 0.9,
            "valence": 0.4, "danceability": 0.6, "acousticness": 0.08,
            "top_k": 3,
        })
        assert len(result["recommendations"]) == 3

    def test_formula_note_present(self):
        result = score_song_classic.invoke({
            "genre": "lofi", "mood": "chill", "energy": 0.35,
            "valence": 0.60, "danceability": 0.55, "acousticness": 0.80,
        })
        assert "formula_note" in result


# --- check_diversity ---

class TestDiversityCheck:
    def _make_songs(self, genres, moods, energies):
        return [
            {"id": i, "title": f"Song {i}", "genre": g, "mood": m, "energy": e}
            for i, (g, m, e) in enumerate(zip(genres, moods, energies), 1)
        ]

    def test_diverse_list_passes(self):
        songs = self._make_songs(
            ["lofi", "pop", "rock", "jazz", "hip-hop"],
            ["chill", "happy", "intense", "relaxed", "confident"],
            [0.3, 0.8, 0.9, 0.4, 0.75],
        )
        result = check_diversity.invoke({"songs": songs})
        assert result["passed"] is True

    def test_genre_lock_in_detected(self):
        songs = self._make_songs(
            ["lofi", "lofi", "lofi", "lofi", "pop"],
            ["chill", "chill", "focused", "chill", "happy"],
            [0.3, 0.35, 0.4, 0.32, 0.8],
        )
        result = check_diversity.invoke({"songs": songs})
        assert result["passed"] is False
        assert any("lock-in" in issue.lower() or "genre" in issue.lower() for issue in result["issues"])

    def test_all_same_genre_flagged(self):
        songs = self._make_songs(
            ["lofi"] * 4,
            ["chill"] * 4,
            [0.3, 0.35, 0.4, 0.32],
        )
        result = check_diversity.invoke({"songs": songs})
        assert result["passed"] is False

    def test_narrow_energy_range_flagged(self):
        songs = self._make_songs(
            ["pop", "rock", "jazz", "lofi"],
            ["happy", "intense", "relaxed", "chill"],
            [0.50, 0.52, 0.51, 0.50],
        )
        result = check_diversity.invoke({"songs": songs})
        assert result["passed"] is False

    def test_empty_list_fails(self):
        result = check_diversity.invoke({"songs": []})
        assert result["passed"] is False

    def test_metrics_present(self):
        songs = self._make_songs(["lofi", "pop"], ["chill", "happy"], [0.3, 0.8])
        result = check_diversity.invoke({"songs": songs})
        assert "metrics" in result
        assert "unique_genres" in result["metrics"]


# --- detect_preference_conflicts ---

class TestConflictDetector:
    def test_no_conflict_clean_profile(self):
        result = detect_preference_conflicts.invoke({
            "genre": "lofi", "mood": "chill", "energy": 0.35, "acousticness": 0.8,
        })
        assert result["has_conflicts"] is False
        assert result["conflicts"] == []

    def test_high_energy_chill_mood_conflict(self):
        result = detect_preference_conflicts.invoke({
            "genre": "pop", "mood": "chill", "energy": 0.90,
        })
        assert result["has_conflicts"] is True
        names = [c["name"] for c in result["conflicts"]]
        assert "high_energy_chill_mood" in names

    def test_low_energy_intense_mood_conflict(self):
        result = detect_preference_conflicts.invoke({
            "mood": "intense", "energy": 0.20,
        })
        assert result["has_conflicts"] is True
        names = [c["name"] for c in result["conflicts"]]
        assert "low_energy_intense_mood" in names

    def test_high_acousticness_electronic_genre(self):
        result = detect_preference_conflicts.invoke({
            "genre": "electronic", "acousticness": 0.85,
        })
        assert result["has_conflicts"] is True

    def test_partial_profile_no_crash(self):
        result = detect_preference_conflicts.invoke({"mood": "happy"})
        assert "has_conflicts" in result

    def test_clarification_questions_present_on_conflict(self):
        result = detect_preference_conflicts.invoke({
            "mood": "chill", "energy": 0.92,
        })
        assert result["has_conflicts"] is True
        assert len(result["clarification_questions"]) > 0
