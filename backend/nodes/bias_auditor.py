"""
Bias Auditor node -- self-critique gate before finalizing recommendations.

A separate LLM call reviews the candidate recommendations against known biases.
If issues are found, the graph routes back to the recommender for re-ranking.
If clean, recommendations are finalized and flow continues to the API response.
"""

import json
import logging
import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from backend.state import AgentState, BiasAuditResult
from backend.langfuse_callback import get_callback_handler

logger = logging.getLogger(__name__)

_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
    return _llm


SYSTEM_PROMPT = """You are a bias auditor for a music recommender. Check for:
1. Genre lock-in: 4+ of 5 songs same genre (unless user asked for one genre)
2. All songs same mood with zero variety
3. Any recommended song in the excluded_ids list

Return ONLY valid JSON:
{"passed": true/false, "issues": ["..."], "suggestions": ["..."]}"""


def bias_auditor_node(state: AgentState) -> AgentState:
    """Review candidate_songs for bias. Sets state.bias_audit and routes accordingly."""
    candidates = state.candidate_songs
    profile = state.user_profile

    if not candidates:
        state.bias_audit = BiasAuditResult(
            passed=False,
            issues=["No recommendations were generated."],
            suggestions=["Try broadening the search with vibe_search or relaxing attribute filters."],
        )
        return state

    audit_input = {
        "recommendations": [
            {
                "id": r.id,
                "title": r.title,
                "genre": r.genre,
                "mood": r.mood,
                "energy": r.energy,
                "confidence": r.confidence,
                "explanation": r.explanation,
            }
            for r in candidates
        ],
        "user_profile": {
            "genre": profile.genre,
            "mood": profile.mood,
            "energy": profile.energy,
            "excluded_ids": profile.excluded_song_ids,
        },
        "tools_called": state.tool_calls_made,
    }

    try:
        llm = _get_llm()
        cb, lf_meta = get_callback_handler(state.session_id, "bias_auditor")
        kwargs = {"config": {"callbacks": [cb], "metadata": lf_meta, "run_name": "bias_auditor"}} if cb else {}
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(audit_input, indent=2)),
        ], **kwargs)

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:].strip()

        result = json.loads(raw)
        audit = BiasAuditResult(
            passed=bool(result.get("passed", False)),
            issues=result.get("issues", []),
            suggestions=result.get("suggestions", []),
        )
        logger.info(f"[bias_auditor] passed={audit.passed} issues={audit.issues}")
        state.bias_audit = audit

        # Always populate final_recommendations — the graph reads this at END
        state.final_recommendations = candidates

        # Increment rerank_count HERE (inside the node) so LangGraph persists it.
        # should_rerank() is a conditional edge — mutations there are NOT saved.
        if not audit.passed:
            state.rerank_count += 1

    except Exception as e:
        logger.error(f"[bias_auditor] error: {e}")
        state.bias_audit = BiasAuditResult(
            passed=True,
            issues=[],
            suggestions=[f"Auditor error (non-blocking): {e}"],
        )
        state.final_recommendations = candidates

    return state


def should_rerank(state: AgentState) -> str:
    """
    Conditional edge function for LangGraph.
    rerank_count is incremented by bias_auditor_node (not here — LangGraph
    does not persist state mutations made inside conditional edge functions).

    Allows exactly ONE re-rank (rerank_count goes 1 → 2 on second failure).
    """
    if state.bias_audit is None or state.bias_audit.passed:
        return "finalize"

    if not state.candidate_songs:
        logger.warning("[bias_auditor] no candidates, forcing finalize")
        return "finalize"

    # rerank_count was incremented in bias_auditor_node before this is called.
    # After the first failure it is 1 → allow one re-rank.
    # After the second failure it is 2 → force finalize.
    if state.rerank_count >= 2:
        logger.warning("[bias_auditor] rerank limit reached, forcing finalize")
        return "finalize"

    return "rerank"
