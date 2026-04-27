"""
Guardrail and edge case tests for VibeFinder Agent.

Tests the system's behavior under unusual or adversarial inputs.
All LLM calls are mocked so these run fast without API keys.

Run with: pytest backend/tests/test_guardrails.py -v
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.state import AgentState, UserProfile, SongRecommendation, ConversationMessage
from backend.tools.catalog_search import catalog_search
from backend.tools.conflict_detector import detect_preference_conflicts
from backend.tools.diversity_check import check_diversity
from backend.tools.classic_scorer import score_song_classic


# -----------------------------------------------------------------------
# Empty / minimal input guardrails
# -----------------------------------------------------------------------

class TestEmptyInputGuardrails:
    def test_catalog_search_no_filters_returns_songs(self):
        """With no filters, catalog_search should return all songs (up to limit)."""
        result = catalog_search.invoke({})
        assert result["total_found"] > 0
        assert len(result["songs"]) > 0

    def test_catalog_search_impossible_filter_returns_empty(self):
        """Genre that doesn't exist returns empty gracefully, no crash."""
        result = catalog_search.invoke({"genre": "xyzzy_nonexistent"})
        assert result["songs"] == []
        assert result["total_found"] == 0
        assert "error" not in result

    def test_diversity_check_single_song_passes(self):
        """Single song should not trigger diversity flags."""
        result = check_diversity.invoke({"songs": [
            {"id": 1, "title": "X", "genre": "lofi", "mood": "chill", "energy": 0.3}
        ]})
        assert "metrics" in result

    def test_conflict_detector_no_inputs_no_crash(self):
        """Empty profile should not detect any conflicts."""
        result = detect_preference_conflicts.invoke({})
        assert result["has_conflicts"] is False

    def test_classic_scorer_unknown_genre_still_scores(self):
        """V1 scorer with a genre not in catalog should still return results (just no genre match bonus)."""
        result = score_song_classic.invoke({
            "genre": "polka", "mood": "happy", "energy": 0.7,
            "valence": 0.7, "danceability": 0.7, "acousticness": 0.3,
        })
        assert "recommendations" in result
        assert len(result["recommendations"]) > 0
        # None should have a genre match since 'polka' isn't in catalog
        for rec in result["recommendations"]:
            assert rec["genre"] != "polka"


# -----------------------------------------------------------------------
# Contradictory preference guardrails
# -----------------------------------------------------------------------

class TestContradictoryPreferences:
    def test_high_energy_chill_detected(self):
        result = detect_preference_conflicts.invoke({"energy": 0.92, "mood": "chill"})
        assert result["has_conflicts"] is True

    def test_low_energy_intense_detected(self):
        result = detect_preference_conflicts.invoke({"energy": 0.15, "mood": "intense"})
        assert result["has_conflicts"] is True

    def test_acoustic_electronic_detected(self):
        result = detect_preference_conflicts.invoke({"genre": "electronic", "acousticness": 0.90})
        assert result["has_conflicts"] is True

    def test_clean_profile_no_false_positive(self):
        result = detect_preference_conflicts.invoke({
            "genre": "lofi", "mood": "chill", "energy": 0.35, "acousticness": 0.8
        })
        assert result["has_conflicts"] is False

    def test_partial_profile_no_crash(self):
        """Only one attribute — should never raise."""
        result = detect_preference_conflicts.invoke({"energy": 0.5})
        assert "has_conflicts" in result


# -----------------------------------------------------------------------
# All-songs-filtered guardrail
# -----------------------------------------------------------------------

class TestAllSongsFiltered:
    def test_impossible_combined_filter_returns_empty_not_crash(self):
        """Contradictory filters (lofi + very high energy) return empty gracefully."""
        result = catalog_search.invoke({
            "genre": "lofi",
            "min_energy": 0.95,
            "max_energy": 1.0,
        })
        assert result["songs"] == []
        assert result["total_found"] == 0

    def test_all_ids_excluded_returns_empty(self):
        """Excluding every song ID should return empty."""
        all_ids = list(range(1, 49))
        result = catalog_search.invoke({"exclude_ids": all_ids})
        assert result["songs"] == []


# -----------------------------------------------------------------------
# Bias auditor state guardrails
# -----------------------------------------------------------------------

class TestBiasAuditorGuardrails:
    def test_should_rerank_caps_at_one_attempt(self):
        from backend.nodes.bias_auditor import should_rerank
        from backend.state import BiasAuditResult

        state = AgentState(session_id="test")
        state.bias_audit = BiasAuditResult(passed=False, issues=["genre lock-in"])
        state.rerank_count = 1  # Already re-ranked once
        state.candidate_songs = [
            SongRecommendation(
                id=1, title="X", artist="Y", genre="lofi", mood="chill",
                energy=0.3, valence=0.5, danceability=0.5, acousticness=0.7,
                score=5.0, confidence=0.6, explanation="test"
            )
        ]
        assert should_rerank(state) == "finalize"

    def test_should_rerank_forces_finalize_with_empty_candidates(self):
        from backend.nodes.bias_auditor import should_rerank
        from backend.state import BiasAuditResult

        state = AgentState(session_id="test")
        state.bias_audit = BiasAuditResult(passed=False, issues=["no songs"])
        state.candidate_songs = []
        assert should_rerank(state) == "finalize"

    def test_should_rerank_forces_finalize_over_tool_limit(self):
        from backend.nodes.bias_auditor import should_rerank
        from backend.state import BiasAuditResult

        state = AgentState(session_id="test")
        state.bias_audit = BiasAuditResult(passed=False, issues=["issue"])
        state.tool_calls_made = ["tool"] * 11
        state.candidate_songs = []
        assert should_rerank(state) == "finalize"


# -----------------------------------------------------------------------
# Router guardrails
# -----------------------------------------------------------------------

class TestRouterGuardrails:
    def test_empty_messages_defaults_to_general_chat(self):
        from backend.nodes.router import router_node
        state = AgentState(session_id="test")
        result = router_node(state)
        assert result.intent == "general_chat"

    def test_non_user_message_defaults_to_general_chat(self):
        from backend.nodes.router import router_node
        state = AgentState(session_id="test")
        state.messages.append(ConversationMessage(role="assistant", content="here are your songs"))
        result = router_node(state)
        assert result.intent == "general_chat"

    @patch("backend.nodes.router.ChatGroq")
    def test_llm_failure_defaults_to_general_chat(self, mock_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM unavailable")
        mock_cls.return_value = mock_llm

        from backend.nodes.router import router_node
        import backend.nodes.router as r_module
        r_module._llm = mock_llm

        state = AgentState(session_id="test")
        state.messages.append(ConversationMessage(role="user", content="I want music"))
        result = router_node(state)
        assert result.intent == "general_chat"
        assert result.error is not None
