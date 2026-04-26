"""
Integration tests for the VibeFinder LangGraph agent.

These tests mock the LLM calls so they run fast without API keys.
Run with: pytest backend/tests/test_graph.py -v

Tests verify:
- Router correctly dispatches intents to the right nodes
- Profile builder merges preferences into state
- Bias auditor correctly gates output
- Feedback handler updates profile and exclusions
- Full graph runs without crashing (smoke tests)
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.state import AgentState, UserProfile, SongRecommendation, BiasAuditResult, ConversationMessage


# --- Router tests ---

class TestRouterNode:
    def _make_state(self, message: str) -> AgentState:
        state = AgentState(session_id="test")
        state.messages.append(ConversationMessage(role="user", content=message))
        return state

    @patch("backend.nodes.router.ChatGoogleGenerativeAI")
    def test_recommend_intent(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="recommend")
        mock_llm_cls.return_value = mock_llm

        from backend.nodes.router import router_node
        import backend.nodes.router as router_module
        router_module._llm = mock_llm

        state = self._make_state("I want some chill lofi music")
        result = router_node(state)
        assert result.intent == "recommend"

    @patch("backend.nodes.router.ChatGoogleGenerativeAI")
    def test_feedback_intent(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="feedback")
        mock_llm_cls.return_value = mock_llm

        from backend.nodes.router import router_node
        import backend.nodes.router as router_module
        router_module._llm = mock_llm

        state = self._make_state("Not that one, too acoustic")
        result = router_node(state)
        assert result.intent == "feedback"

    @patch("backend.nodes.router.ChatGoogleGenerativeAI")
    def test_invalid_response_defaults_to_general(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="something_random_not_an_intent")
        mock_llm_cls.return_value = mock_llm

        from backend.nodes.router import router_node
        import backend.nodes.router as router_module
        router_module._llm = mock_llm

        state = self._make_state("hello")
        result = router_node(state)
        assert result.intent == "general_chat"

    def test_empty_messages_defaults_to_general(self):
        from backend.nodes.router import router_node
        state = AgentState(session_id="test")
        result = router_node(state)
        assert result.intent == "general_chat"


# --- Profile Builder tests ---

class TestProfileBuilderNode:
    @patch("backend.nodes.profile_builder.ChatGoogleGenerativeAI")
    def test_extracts_genre_and_mood(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"genre": "lofi", "mood": "chill", "energy": 0.35}'
        )
        mock_llm_cls.return_value = mock_llm

        from backend.nodes.profile_builder import profile_builder_node
        import backend.nodes.profile_builder as pb_module
        pb_module._llm = mock_llm

        state = AgentState(session_id="test")
        state.messages.append(ConversationMessage(role="user", content="I want chill lofi"))
        result = profile_builder_node(state)

        assert result.user_profile.genre == "lofi"
        assert result.user_profile.mood == "chill"
        assert result.user_profile.energy == 0.35

    @patch("backend.nodes.profile_builder.ChatGoogleGenerativeAI")
    def test_merges_with_existing_profile(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"mood": "happy"}'
        )
        mock_llm_cls.return_value = mock_llm

        from backend.nodes.profile_builder import profile_builder_node
        import backend.nodes.profile_builder as pb_module
        pb_module._llm = mock_llm

        state = AgentState(session_id="test")
        state.user_profile.genre = "pop"  # Pre-existing
        state.messages.append(ConversationMessage(role="user", content="something happy"))
        result = profile_builder_node(state)

        assert result.user_profile.genre == "pop"   # Preserved
        assert result.user_profile.mood == "happy"  # Updated

    @patch("backend.nodes.profile_builder.ChatGoogleGenerativeAI")
    def test_handles_json_parse_error_gracefully(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="not json at all")
        mock_llm_cls.return_value = mock_llm

        from backend.nodes.profile_builder import profile_builder_node
        import backend.nodes.profile_builder as pb_module
        pb_module._llm = mock_llm

        state = AgentState(session_id="test")
        state.messages.append(ConversationMessage(role="user", content="anything"))
        result = profile_builder_node(state)
        assert result is not None  # Doesn't crash


# --- Bias Auditor tests ---

class TestBiasAuditorNode:
    def _make_recs(self, genres, moods, energies, confidences):
        return [
            SongRecommendation(
                id=i, title=f"Song {i}", artist="Artist",
                genre=g, mood=m, energy=e, valence=0.5,
                danceability=0.5, acousticness=0.5,
                score=6.0, confidence=c, explanation="test"
            )
            for i, (g, m, e, c) in enumerate(zip(genres, moods, energies, confidences), 1)
        ]

    @patch("backend.nodes.bias_auditor.ChatGoogleGenerativeAI")
    def test_passes_diverse_recommendations(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"passed": true, "issues": [], "suggestions": []}'
        )
        mock_llm_cls.return_value = mock_llm

        from backend.nodes.bias_auditor import bias_auditor_node
        import backend.nodes.bias_auditor as ba_module
        ba_module._llm = mock_llm

        state = AgentState(session_id="test")
        state.candidate_songs = self._make_recs(
            ["lofi", "pop", "rock", "jazz", "hip-hop"],
            ["chill", "happy", "intense", "relaxed", "confident"],
            [0.3, 0.8, 0.9, 0.4, 0.75],
            [0.8, 0.7, 0.85, 0.75, 0.8],
        )
        result = bias_auditor_node(state)
        assert result.bias_audit.passed is True
        assert result.final_recommendations == state.candidate_songs

    @patch("backend.nodes.bias_auditor.ChatGoogleGenerativeAI")
    def test_flags_genre_lock_in(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"passed": false, "issues": ["Genre lock-in: all songs are lofi"], "suggestions": ["Add songs from other genres"]}'
        )
        mock_llm_cls.return_value = mock_llm

        from backend.nodes.bias_auditor import bias_auditor_node
        import backend.nodes.bias_auditor as ba_module
        ba_module._llm = mock_llm

        state = AgentState(session_id="test")
        state.candidate_songs = self._make_recs(
            ["lofi"] * 5, ["chill"] * 5,
            [0.3, 0.35, 0.4, 0.32, 0.38],
            [0.8] * 5,
        )
        result = bias_auditor_node(state)
        assert result.bias_audit.passed is False
        assert len(result.bias_audit.issues) > 0

    def test_empty_candidates_fails_audit(self):
        from backend.nodes.bias_auditor import bias_auditor_node
        state = AgentState(session_id="test")
        state.candidate_songs = []
        result = bias_auditor_node(state)
        assert result.bias_audit.passed is False

    def test_should_rerank_returns_rerank_on_fail(self):
        from backend.nodes.bias_auditor import should_rerank
        state = AgentState(session_id="test")
        state.bias_audit = BiasAuditResult(passed=False, issues=["genre lock-in"])
        assert should_rerank(state) == "rerank"

    def test_should_rerank_returns_finalize_on_pass(self):
        from backend.nodes.bias_auditor import should_rerank
        state = AgentState(session_id="test")
        state.bias_audit = BiasAuditResult(passed=True, issues=[])
        assert should_rerank(state) == "finalize"

    def test_should_rerank_forces_finalize_after_max_tools(self):
        from backend.nodes.bias_auditor import should_rerank
        state = AgentState(session_id="test")
        state.bias_audit = BiasAuditResult(passed=False, issues=["issue"])
        state.tool_calls_made = ["tool"] * 13  # Over the limit
        state.candidate_songs = []
        assert should_rerank(state) == "finalize"


# --- Feedback Handler tests ---

class TestFeedbackNode:
    @patch("backend.nodes.feedback.ChatGoogleGenerativeAI")
    def test_dislike_adds_to_exclusions(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='''{
            "song_id": 3, "rating": "disliked", "comment": "too loud",
            "profile_adjustments": {}, "exclude_song_id": 3
        }''')
        mock_llm_cls.return_value = mock_llm

        from backend.nodes.feedback import feedback_node
        import backend.nodes.feedback as fb_module
        fb_module._llm = mock_llm

        state = AgentState(session_id="test")
        state.messages.append(ConversationMessage(role="user", content="not that one"))
        result = feedback_node(state)
        assert 3 in result.user_profile.excluded_song_ids

    @patch("backend.nodes.feedback.ChatGoogleGenerativeAI")
    def test_profile_adjusted_on_too_acoustic(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='''{
            "song_id": 4, "rating": "disliked", "comment": "too acoustic",
            "profile_adjustments": {"acousticness": 0.2}, "exclude_song_id": 4
        }''')
        mock_llm_cls.return_value = mock_llm

        from backend.nodes.feedback import feedback_node
        import backend.nodes.feedback as fb_module
        fb_module._llm = mock_llm

        state = AgentState(session_id="test")
        state.messages.append(ConversationMessage(role="user", content="too acoustic"))
        result = feedback_node(state)
        assert result.user_profile.acousticness == 0.2

    @patch("backend.nodes.feedback.ChatGoogleGenerativeAI")
    def test_resets_candidates_for_re_recommendation(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='''{
            "song_id": 1, "rating": "disliked", "comment": "meh",
            "profile_adjustments": {}, "exclude_song_id": null
        }''')
        mock_llm_cls.return_value = mock_llm

        from backend.nodes.feedback import feedback_node
        import backend.nodes.feedback as fb_module
        fb_module._llm = mock_llm

        state = AgentState(session_id="test")
        state.messages.append(ConversationMessage(role="user", content="not great"))
        state.final_recommendations = [
            SongRecommendation(
                id=1, title="X", artist="Y", genre="pop", mood="happy",
                energy=0.8, valence=0.7, danceability=0.8, acousticness=0.1,
                score=6.0, confidence=0.7, explanation="test"
            )
        ]
        result = feedback_node(state)
        assert result.final_recommendations == []
        assert result.candidate_songs == []
        assert result.intent == "recommend"
