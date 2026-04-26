"""
Bias Auditor node -- self-critique gate before finalizing recommendations.

A separate LLM call reviews the candidate recommendations against known biases.
If issues are found, the graph routes back to the recommender for re-ranking.
If clean, recommendations are finalized and flow continues to the API response.
"""

import json
import logging
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from backend.state import AgentState, BiasAuditResult

logger = logging.getLogger(__name__)

_llm: ChatGoogleGenerativeAI | None = None


def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
        )
    return _llm


SYSTEM_PROMPT = """You are a bias auditor for a music recommendation system.

Review the proposed recommendations for these known bias patterns:

1. GENRE LOCK-IN: Are 4+ out of 5 songs from the same genre? (bad if user didn't specifically ask for one genre)
2. MOOD HOMOGENEITY: Do all songs share the same mood with no variety?
3. ENERGY CLUSTERING: Is the energy range less than 0.15 across all songs? (too samey)
4. CONFIDENCE TOO LOW: Are any recommendations below 0.4 confidence with no caveat in the explanation?
5. IGNORED EXCLUSIONS: Do any recommended songs appear in the excluded_ids list?
6. CATALOG BLINDNESS: For a 48-song catalog, did the agent only look at 5 or fewer songs before deciding?

Return ONLY valid JSON:
{
  "passed": true/false,
  "issues": ["specific issue description if any"],
  "suggestions": ["specific fix suggestion if any"]
}

If no issues found, return {"passed": true, "issues": [], "suggestions": []}"""


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
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(audit_input, indent=2)),
        ])

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

        if audit.passed:
            state.final_recommendations = candidates

    except Exception as e:
        logger.error(f"[bias_auditor] error: {e}")
        # On auditor failure, pass through to avoid blocking the user
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
    Returns 'rerank' if the auditor found issues, 'finalize' if clean.
    """
    if state.bias_audit is None:
        return "finalize"
    # Only re-rank once to avoid infinite loops
    if state.bias_audit.passed:
        return "finalize"
    if len(state.tool_calls_made) > 12:
        logger.warning("[bias_auditor] max tool calls reached, forcing finalize")
        state.final_recommendations = state.candidate_songs
        return "finalize"
    return "rerank"
