"""
Feedback Handler node -- processes user feedback and updates session state.

Handles: thumbs up/down, "more like this", "not that one", "too acoustic", etc.
Updates excluded_song_ids, liked_song_ids, and adjusts profile attributes
so the recommender has better context on the next pass.
"""

import json
import logging
import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from backend.state import AgentState, FeedbackEntry
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


SYSTEM_PROMPT = """You are a feedback interpreter for a music recommendation system.

The user gave feedback on a recommendation. Extract structured feedback and
any profile adjustments implied by the comment.

Return ONLY valid JSON:
{
  "song_id": <int or null if unclear>,
  "rating": "liked" | "disliked" | "more_like_this" | "less_like_this",
  "comment": "<original feedback text>",
  "profile_adjustments": {
    "energy": <float 0-1 or null>,
    "acousticness": <float 0-1 or null>,
    "valence": <float 0-1 or null>,
    "danceability": <float 0-1 or null>,
    "mood": "<string or null>",
    "genre": "<string or null>"
  },
  "exclude_song_id": <int or null>
}

For 'too acoustic' → lower acousticness; 'too intense' → lower energy and mood to chill;
'more upbeat' → raise energy and valence; 'not that one' → exclude that song_id."""


def feedback_node(state: AgentState) -> AgentState:
    """Parse feedback message, update profile and exclusion list."""
    latest = next(
        (m for m in reversed(state.messages) if m.role == "user"), None
    )
    if not latest:
        return state

    # Provide context: what was last recommended
    last_recs = state.final_recommendations or state.candidate_songs
    recs_context = [
        {"id": r.id, "title": r.title, "artist": r.artist}
        for r in last_recs[:5]
    ]

    try:
        llm = _get_llm()
        cb, lf_meta = get_callback_handler(state.session_id, "feedback")
        kwargs = {"config": {"callbacks": [cb], "metadata": lf_meta, "run_name": "feedback"}} if cb else {}
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                f"User feedback: {latest.content}\n\n"
                f"Last recommendations: {json.dumps(recs_context)}"
            )),
        ], **kwargs)

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:].strip()

        data = json.loads(raw)
        logger.info(f"[feedback] parsed={data}")

        # Record feedback entry
        song_id = data.get("song_id")
        rating = data.get("rating", "disliked")
        if rating not in ("liked", "disliked", "more_like_this", "less_like_this"):
            rating = "disliked"

        if song_id:
            entry = FeedbackEntry(
                song_id=int(song_id),
                rating=rating,
                comment=data.get("comment", latest.content),
            )
            state.feedback_entries.append(entry)

            if rating in ("disliked", "less_like_this"):
                if song_id not in state.user_profile.excluded_song_ids:
                    state.user_profile.excluded_song_ids.append(int(song_id))
            elif rating in ("liked", "more_like_this"):
                if song_id not in state.user_profile.liked_song_ids:
                    state.user_profile.liked_song_ids.append(int(song_id))

        # Explicit exclude
        exclude_id = data.get("exclude_song_id")
        if exclude_id and exclude_id not in state.user_profile.excluded_song_ids:
            state.user_profile.excluded_song_ids.append(int(exclude_id))

        # Apply profile adjustments
        adjustments = data.get("profile_adjustments", {})
        profile = state.user_profile
        if adjustments.get("energy") is not None:
            profile.energy = float(adjustments["energy"])
        if adjustments.get("acousticness") is not None:
            profile.acousticness = float(adjustments["acousticness"])
        if adjustments.get("valence") is not None:
            profile.valence = float(adjustments["valence"])
        if adjustments.get("danceability") is not None:
            profile.danceability = float(adjustments["danceability"])
        if adjustments.get("mood"):
            profile.mood = adjustments["mood"]
        if adjustments.get("genre"):
            profile.genre = adjustments["genre"]
        state.user_profile = profile

        # Reset candidates so recommender runs fresh
        state.candidate_songs = []
        state.final_recommendations = []
        state.bias_audit = None
        state.intent = "recommend"

    except Exception as e:
        logger.error(f"[feedback] error: {e}")
        state.error = str(e)

    return state
