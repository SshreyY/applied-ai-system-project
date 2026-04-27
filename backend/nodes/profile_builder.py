"""
Profile Builder node -- extracts structured music preferences from natural language
and merges them into the session's UserProfile.
"""

import json
import logging
import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from backend.state import AgentState, UserProfile

logger = logging.getLogger(__name__)

_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
    return _llm


SYSTEM_PROMPT = """You are a music preference extractor for a music recommendation system.

Extract music preferences from the user's message and return a JSON object.
Only include fields you can confidently infer. Omit fields you can't determine.

Valid genres: lofi, pop, rock, metal, hip-hop, r&b, jazz, classical, electronic,
synthwave, synthpop, indie pop, indie folk, folk, country, blues, soul, reggae,
afrobeats, latin, k-pop, trap, drum and bass, bossa nova, jazz fusion, flamenco, world, ambient

Valid moods: happy, chill, intense, relaxed, focused, nostalgic, moody, romantic,
melancholic, confident, angry, energetic

Numeric fields (0.0 to 1.0): energy, valence, danceability, acousticness
Activity field: any situation description (e.g. "studying", "working out", "road trip")

Return ONLY valid JSON, no markdown, no explanation:
{
  "genre": "...",
  "mood": "...",
  "energy": 0.0,
  "valence": 0.0,
  "danceability": 0.0,
  "acousticness": 0.0,
  "activity": "..."
}"""


def profile_builder_node(state: AgentState) -> AgentState:
    """Extract preferences from the latest user message and merge into state.user_profile."""
    if not state.messages:
        return state

    latest = next(
        (m for m in reversed(state.messages) if m.role == "user"), None
    )
    if not latest:
        return state

    try:
        llm = _get_llm()
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=latest.content),
        ])
        raw = response.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        extracted = json.loads(raw)
        logger.info(f"[profile_builder] extracted={extracted}")

        profile = state.user_profile

        # Merge: only overwrite fields that were actually extracted
        if "genre" in extracted and extracted["genre"]:
            profile.genre = extracted["genre"].lower().strip()
        if "mood" in extracted and extracted["mood"]:
            profile.mood = extracted["mood"].lower().strip()
        if "energy" in extracted and extracted["energy"] is not None:
            profile.energy = float(extracted["energy"])
        if "valence" in extracted and extracted["valence"] is not None:
            profile.valence = float(extracted["valence"])
        if "danceability" in extracted and extracted["danceability"] is not None:
            profile.danceability = float(extracted["danceability"])
        if "acousticness" in extracted and extracted["acousticness"] is not None:
            profile.acousticness = float(extracted["acousticness"])
        if "activity" in extracted and extracted["activity"]:
            profile.activity = extracted["activity"].lower().strip()

        state.user_profile = profile

    except json.JSONDecodeError as e:
        logger.warning(f"[profile_builder] JSON parse error: {e} -- raw: {response.content[:200]}")
    except Exception as e:
        logger.error(f"[profile_builder] error: {e}")
        state.error = str(e)

    return state
