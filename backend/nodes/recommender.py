"""
Recommender node -- the core ReAct tool-calling loop.

The LLM reasons about the user profile, decides which tools to call and in
what order, iterates until it has a good set of candidates, then formats
final recommendations with explanations and confidence scores.
"""

import json
import logging
import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from backend.state import AgentState, SongRecommendation
from backend.langfuse_callback import get_callback_handler
from backend.tools.catalog_search import catalog_search
from backend.tools.vibe_search import vibe_search
from backend.tools.genre_knowledge import lookup_genre_info
from backend.tools.activity_context import lookup_activity_context
from backend.tools.classic_scorer import score_song_classic
from backend.tools.diversity_check import check_diversity
from backend.tools.conflict_detector import detect_preference_conflicts

logger = logging.getLogger(__name__)

TOOLS = [
    catalog_search,
    vibe_search,
    lookup_genre_info,
    lookup_activity_context,
    score_song_classic,
    check_diversity,
    detect_preference_conflicts,
]

_llm_with_tools: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm_with_tools
    if _llm_with_tools is None:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3,
        )
        _llm_with_tools = llm.bind_tools(TOOLS)
    return _llm_with_tools


SYSTEM_PROMPT = """You are VibeFinder, an expert music recommendation agent.

You have access to these tools:
- catalog_search: filter songs by genre, mood, energy ranges
- vibe_search: semantic search using natural language descriptions
- lookup_genre_info: find similar genres and typical audio attributes
- lookup_activity_context: map activities/situations to music attributes
- score_song_classic: run the V1 formula for baseline comparison
- check_diversity: analyze candidate diversity (genre lock-in, energy spread)
- detect_preference_conflicts: flag contradictory preferences before searching

Your process:
1. Look at the user profile and conversation history
2. If there's an activity described, call lookup_activity_context first
3. If genre is mentioned, call lookup_genre_info to find similar genres too
4. Call catalog_search and/or vibe_search to get candidates
5. Call check_diversity on your candidates — if diversity is low, broaden your search
6. Optionally call score_song_classic to get a V1 baseline for comparison
7. Select the top 3-5 songs and assign each a confidence score (0.0-1.0)
8. Return your final recommendations as a JSON array

For each recommended song include:
- id, title, artist, genre, mood, energy, valence, danceability, acousticness
- score (your overall match score, 0-10)
- confidence (0.0-1.0, how confident you are this fits the user)
- explanation (1-2 sentences why this song fits)
- v1_score (from classic_scorer if you called it, otherwise null)

If confidence is below 0.5, note it in the explanation.

Return your final answer as JSON:
{"recommendations": [...]}

The catalog has 48 songs. Song IDs are integers 1-48."""


def recommender_node(state: AgentState) -> AgentState:
    """Run the ReAct tool-calling loop and populate state.candidate_songs."""
    profile = state.user_profile
    conversation = state.messages

    profile_summary = {
        k: v for k, v in {
            "genre": profile.genre,
            "mood": profile.mood,
            "energy": profile.energy,
            "valence": profile.valence,
            "danceability": profile.danceability,
            "acousticness": profile.acousticness,
            "activity": profile.activity,
            "excluded_ids": profile.excluded_song_ids,
            "liked_ids": profile.liked_song_ids,
        }.items() if v is not None and v != []
    }

    user_request = next(
        (m.content for m in reversed(conversation) if m.role == "user"), ""
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"User request: {user_request}\n\n"
            f"Current user profile: {json.dumps(profile_summary)}\n\n"
            "Please recommend songs using the tools available."
        )),
    ]

    tool_map = {t.name: t for t in TOOLS}
    tools_called = []
    max_iterations = 8

    try:
        llm = _get_llm()
        cb = get_callback_handler(state.session_id, "recommender")
        invoke_kwargs = {"config": {"callbacks": [cb]}} if cb else {}

        for iteration in range(max_iterations):
            response = llm.invoke(messages, **invoke_kwargs)
            messages.append(response)

            if not response.tool_calls:
                # LLM is done with tool calls — extract final recommendations
                break

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tools_called.append(tool_name)
                logger.info(f"[recommender] tool_call={tool_name} args={tool_args}")

                tool_fn = tool_map.get(tool_name)
                if tool_fn is None:
                    tool_result = {"error": f"Unknown tool: {tool_name}"}
                else:
                    try:
                        tool_result = tool_fn.invoke(tool_args)
                    except Exception as e:
                        logger.warning(f"[recommender] tool {tool_name} error: {e}")
                        tool_result = {"error": str(e)}

                messages.append(ToolMessage(
                    content=json.dumps(tool_result),
                    tool_call_id=tc["id"],
                ))

        state.tool_calls_made = tools_called

        # Parse final JSON recommendations from last AI message
        final_content = response.content or ""
        recommendations = _parse_recommendations(final_content)
        state.candidate_songs = recommendations

    except Exception as e:
        logger.error(f"[recommender] error: {e}")
        state.error = str(e)
        state.candidate_songs = []

    return state


def _parse_recommendations(content: str) -> list[SongRecommendation]:
    """Extract JSON recommendations from the LLM's final response."""
    try:
        # Strip markdown fences
        text = content.strip()
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                if part.strip().startswith("{") or part.strip().startswith("json"):
                    text = part.strip()
                    if text.startswith("json"):
                        text = text[4:].strip()
                    break

        # Find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning("[recommender] no JSON found in response")
            return []

        data = json.loads(text[start:end])
        recs_raw = data.get("recommendations", [])

        results = []
        for r in recs_raw:
            try:
                rec = SongRecommendation(
                    id=int(r["id"]),
                    title=r["title"],
                    artist=r["artist"],
                    genre=r["genre"],
                    mood=r["mood"],
                    energy=float(r.get("energy", 0.5)),
                    valence=float(r.get("valence", 0.5)),
                    danceability=float(r.get("danceability", 0.5)),
                    acousticness=float(r.get("acousticness", 0.5)),
                    score=float(r.get("score", 5.0)),
                    confidence=float(r.get("confidence", 0.5)),
                    explanation=r.get("explanation", ""),
                    v1_score=float(r["v1_score"]) if r.get("v1_score") is not None else None,
                )
                results.append(rec)
            except Exception as e:
                logger.warning(f"[recommender] failed to parse recommendation: {e} -- {r}")

        return results

    except Exception as e:
        logger.warning(f"[recommender] JSON parse error: {e}")
        return []
