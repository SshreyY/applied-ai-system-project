"""
Recommender node.

Tools are called directly (no LLM tool-calling API) to avoid Groq's
tool_use_failed errors. The LLM's only job is formatting the final JSON.
"""

import json
import logging
import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from backend.state import AgentState, SongRecommendation
from backend.langfuse_callback import get_callback_handler
from backend.tools.catalog_search import catalog_search
from backend.tools.vibe_search import vibe_search
from backend.tools.diversity_check import check_diversity

logger = logging.getLogger(__name__)

_llm: ChatGroq | None = None

FORMAT_PROMPT = (
    "You are a music recommendation engine. "
    "Given the candidate songs below, select the best 3-5 for the user and return ONLY valid JSON — "
    "no markdown, no explanation outside the JSON:\n"
    '{"recommendations": ['
    '{"id": <int>, "title": <str>, "artist": <str>, "genre": <str>, "mood": <str>, '
    '"energy": <float>, "valence": <float>, "danceability": <float>, "acousticness": <float>, '
    '"score": <float 0-1>, "confidence": <float 0-1>, "explanation": <str>, "v1_score": null}'
    "]}"
)


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
    return _llm


def recommender_node(state: AgentState) -> AgentState:
    """Call search tools directly, then ask the LLM to format results."""
    profile = state.user_profile

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
        (m.content for m in reversed(state.messages) if m.role == "user"), ""
    )

    tools_called: list[str] = []
    candidates: list[dict] = []

    try:
        # --- Step 1: catalog search using structured profile fields ---
        catalog_args: dict = {}
        if profile.genre:
            catalog_args["genre"] = profile.genre
        if profile.mood:
            catalog_args["mood"] = profile.mood
        if profile.energy is not None:
            catalog_args["min_energy"] = max(0.0, profile.energy - 0.2)
            catalog_args["max_energy"] = min(1.0, profile.energy + 0.2)

        logger.info(f"[recommender] catalog_search args={catalog_args}")
        catalog_result = catalog_search.invoke(catalog_args)
        tools_called.append("catalog_search")
        if isinstance(catalog_result, dict) and "songs" in catalog_result:
            candidates.extend(catalog_result["songs"])

        # --- Step 2: semantic vibe search using the raw user request ---
        vibe_query = user_request or " ".join(
            str(v) for v in [profile.genre, profile.mood, profile.activity] if v
        )
        logger.info(f"[recommender] vibe_search query={vibe_query!r}")
        vibe_result = vibe_search.invoke({"query": vibe_query, "n_results": 8})
        tools_called.append("vibe_search")
        if isinstance(vibe_result, dict) and "songs" in vibe_result:
            existing_ids = {s.get("id") for s in candidates}
            for song in vibe_result["songs"]:
                if song.get("id") not in existing_ids:
                    candidates.append(song)
                    existing_ids.add(song.get("id"))

        # --- Step 3: diversity check ---
        if candidates:
            logger.info(f"[recommender] check_diversity on {len(candidates)} candidates")
            diversity_result = check_diversity.invoke({"songs": candidates})
            tools_called.append("check_diversity")
            logger.info(f"[recommender] diversity={diversity_result}")

        state.tool_calls_made = tools_called
        logger.info(f"[recommender] {len(candidates)} candidates before LLM formatting")

        # --- Step 4: LLM formats the final top picks (no tools bound) ---
        cb, lf_meta = get_callback_handler(state.session_id, "recommender")
        invoke_kwargs = {"config": {"callbacks": [cb], "metadata": lf_meta, "run_name": "recommender"}} if cb else {}

        format_messages = [
            SystemMessage(content=FORMAT_PROMPT),
            HumanMessage(content=(
                f"User profile: {json.dumps(profile_summary)}\n"
                f"User request: {user_request}\n\n"
                f"Candidate songs ({len(candidates)} total):\n"
                f"{json.dumps(candidates, indent=2)}"
            )),
        ]
        response = _get_llm().invoke(format_messages, **invoke_kwargs)
        state.candidate_songs = _parse_recommendations(response.content or "")

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
