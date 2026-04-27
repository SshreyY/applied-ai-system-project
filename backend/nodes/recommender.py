"""
Recommender node.

Tools are called directly (no LLM tool-calling API) to avoid Groq's
tool_use_failed errors. The LLM's only job is formatting the final JSON.
"""

import csv
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
from backend.tools.genre_knowledge import lookup_genre_info
from backend.tools.activity_context import lookup_activity_context
from backend.tools.classic_scorer import score_song_classic

logger = logging.getLogger(__name__)

_llm: ChatGroq | None = None
_catalog_lookup: dict[int, dict] | None = None

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


def _get_catalog_lookup() -> dict[int, dict]:
    """Load songs.csv once and return a dict keyed by song id."""
    global _catalog_lookup
    if _catalog_lookup is None:
        catalog_path = os.getenv("CATALOG_PATH", "backend/data/songs.csv")
        _catalog_lookup = {}
        try:
            with open(catalog_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    sid = int(row["id"])
                    _catalog_lookup[sid] = {
                        "title": row.get("title", ""),
                        "artist": row.get("artist", ""),
                        "genre": row.get("genre") or "unknown",
                        "mood": row.get("mood") or "unknown",
                        "energy": _safe_float(row.get("energy"), 0.5),
                        "valence": _safe_float(row.get("valence"), 0.5),
                        "danceability": _safe_float(row.get("danceability"), 0.5),
                        "acousticness": _safe_float(row.get("acousticness"), 0.5),
                    }
        except Exception as e:
            logger.warning(f"[recommender] catalog lookup load failed: {e}")
    return _catalog_lookup


def _safe_float(value, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _enrich_from_catalog(candidates: list[dict]) -> list[dict]:
    """Fill in None/missing fields on candidates using the CSV catalog."""
    lookup = _get_catalog_lookup()
    enriched = []
    for song in candidates:
        sid = song.get("id")
        if sid and sid in lookup:
            catalog_song = lookup[sid]
            merged = dict(catalog_song)  # start with full catalog data
            merged.update({k: v for k, v in song.items() if v is not None and v != "unknown"})
            merged["id"] = sid  # always keep the id
            enriched.append(merged)
        else:
            enriched.append(song)
    return enriched


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
        catalog_args: dict = {}

        # --- Step 0a: Genre knowledge expansion ---
        # Use similar_genres and typical_attributes to widen the search
        # beyond exact genre matching (fixes V1's binary genre lock-in).
        if profile.genre:
            try:
                genre_info = lookup_genre_info.invoke({"genre": profile.genre})
                tools_called.append("lookup_genre_info")
                logger.info(f"[recommender] lookup_genre_info={genre_info}")
                if not genre_info.get("not_found"):
                    # Refine energy range from genre's typical attributes
                    # if the user didn't specify energy directly
                    if profile.energy is None:
                        attrs = genre_info.get("typical_attributes", {})
                        e = attrs.get("energy")
                        if isinstance(e, dict):
                            catalog_args["min_energy"] = e.get("min", 0.0)
                            catalog_args["max_energy"] = e.get("max", 1.0)
                        elif isinstance(e, list) and len(e) == 2:
                            catalog_args["min_energy"] = float(e[0])
                            catalog_args["max_energy"] = float(e[1])

                    # Search similar genres for additional candidates
                    for sim_genre in genre_info.get("similar_genres", [])[:2]:
                        sim_result = catalog_search.invoke({"genre": sim_genre})
                        if isinstance(sim_result, dict) and "songs" in sim_result:
                            existing_ids = {s.get("id") for s in candidates}
                            for song in sim_result["songs"]:
                                if song.get("id") not in existing_ids:
                                    candidates.append(song)
                                    existing_ids.add(song.get("id"))
            except Exception as e:
                logger.warning(f"[recommender] lookup_genre_info failed (non-fatal): {e}")

        # --- Step 0b: Activity context mapping ---
        # Map activity description → suggested audio attributes + preferred genres
        # (fixes V1's cold-start problem for activity-based requests).
        if profile.activity:
            try:
                activity_info = lookup_activity_context.invoke({"activity": profile.activity})
                tools_called.append("lookup_activity_context")
                logger.info(f"[recommender] lookup_activity_context={activity_info}")
                if not activity_info.get("not_found"):
                    attrs = activity_info.get("suggested_attributes", {})
                    # Fill in energy from activity if not set by profile
                    if profile.energy is None and "energy" in attrs:
                        e = attrs["energy"]
                        if isinstance(e, dict):
                            mid = (e.get("min", 0.3) + e.get("max", 0.7)) / 2
                            catalog_args.setdefault("min_energy", e.get("min", 0.0))
                            catalog_args.setdefault("max_energy", e.get("max", 1.0))
                            profile.energy = round(mid, 2)
                        elif isinstance(e, list) and len(e) == 2:
                            catalog_args.setdefault("min_energy", float(e[0]))
                            catalog_args.setdefault("max_energy", float(e[1]))
                            profile.energy = round((e[0] + e[1]) / 2, 2)

                    # Search preferred genres from activity
                    for pg in activity_info.get("preferred_genres", [])[:2]:
                        pg_result = catalog_search.invoke({"genre": pg})
                        if isinstance(pg_result, dict) and "songs" in pg_result:
                            existing_ids = {s.get("id") for s in candidates}
                            for song in pg_result["songs"]:
                                if song.get("id") not in existing_ids:
                                    candidates.append(song)
                                    existing_ids.add(song.get("id"))
            except Exception as e:
                logger.warning(f"[recommender] lookup_activity_context failed (non-fatal): {e}")

        # --- Step 1: catalog search using structured profile fields ---
        if profile.genre:
            catalog_args["genre"] = profile.genre
        if profile.mood:
            catalog_args["mood"] = profile.mood
        if profile.energy is not None and "min_energy" not in catalog_args:
            catalog_args["min_energy"] = max(0.0, profile.energy - 0.2)
            catalog_args["max_energy"] = min(1.0, profile.energy + 0.2)

        logger.info(f"[recommender] catalog_search args={catalog_args}")
        catalog_result = catalog_search.invoke(catalog_args)
        tools_called.append("catalog_search")
        if isinstance(catalog_result, dict) and "songs" in catalog_result:
            existing_ids = {s.get("id") for s in candidates}
            for song in catalog_result["songs"]:
                if song.get("id") not in existing_ids:
                    candidates.append(song)
                    existing_ids.add(song.get("id"))

        # --- Step 2: semantic vibe search using the raw user request ---
        vibe_query = user_request or " ".join(
            str(v) for v in [profile.genre, profile.mood, profile.activity] if v
        )

        # Enrich vibe query with liked song details so "more like this" works
        liked_ids = profile.liked_song_ids or []
        if liked_ids:
            catalog = _get_catalog_lookup()
            liked_snippets = []
            for lid in liked_ids[:3]:  # cap to avoid bloating the query
                s = catalog.get(lid)
                if s:
                    liked_snippets.append(f"{s['title']} ({s['genre']}, {s['mood']})")
            if liked_snippets:
                vibe_query = vibe_query + " similar to " + ", ".join(liked_snippets)
                logger.info(f"[recommender] liked seeds added to vibe query: {liked_snippets}")

        logger.info(f"[recommender] vibe_search query={vibe_query!r}")
        vibe_result = vibe_search.invoke({"query": vibe_query, "n_results": 8})
        tools_called.append("vibe_search")
        if isinstance(vibe_result, dict) and "songs" in vibe_result:
            existing_ids = {s.get("id") for s in candidates}
            for song in vibe_result["songs"]:
                if song.get("id") not in existing_ids:
                    candidates.append(song)
                    existing_ids.add(song.get("id"))

        # --- Step 2.5: enrich candidates with full catalog data ---
        # vibe_search returns ChromaDB documents that may have None fields.
        candidates = _enrich_from_catalog(candidates)

        # Remove songs the user excluded
        excluded = set(profile.excluded_song_ids or [])
        candidates = [c for c in candidates if c.get("id") not in excluded]

        # --- Step 2.7: V1 classic scoring for comparison ---
        # Run the original rule-based formula to populate v1_score on candidates.
        # Only fires when the profile has enough structured fields.
        _has_full_profile = (
            profile.genre and profile.mood
            and profile.energy is not None and profile.valence is not None
            and profile.danceability is not None and profile.acousticness is not None
        )
        if _has_full_profile:
            try:
                v1_result = score_song_classic.invoke({
                    "genre": profile.genre,
                    "mood": profile.mood,
                    "energy": profile.energy,
                    "valence": profile.valence,
                    "danceability": profile.danceability,
                    "acousticness": profile.acousticness,
                    "top_k": len(candidates) + 5,
                })
                tools_called.append("score_song_classic")
                v1_by_id = {r["id"]: r["v1_score"] for r in v1_result.get("recommendations", [])}
                for c in candidates:
                    if c.get("id") in v1_by_id:
                        c["v1_score"] = v1_by_id[c["id"]]
                logger.info(f"[recommender] score_song_classic scored {len(v1_by_id)} songs")
            except Exception as e:
                logger.warning(f"[recommender] score_song_classic failed (non-fatal): {e}")

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

        liked_context = ""
        if liked_ids:
            cat = _get_catalog_lookup()
            liked_songs = [cat[lid] for lid in liked_ids if lid in cat]
            if liked_songs:
                liked_context = (
                    f"\nUser has LIKED these songs — prioritise similar ones:\n"
                    + json.dumps(liked_songs, indent=2) + "\n"
                )

        format_messages = [
            SystemMessage(content=FORMAT_PROMPT),
            HumanMessage(content=(
                f"User profile: {json.dumps(profile_summary)}\n"
                f"User request: {user_request}\n"
                f"{liked_context}\n"
                f"Candidate songs ({len(candidates)} total):\n"
                f"{json.dumps(candidates, indent=2)}"
            )),
        ]
        response = _get_llm().invoke(format_messages, **invoke_kwargs)
        state.candidate_songs = _parse_recommendations(response.content or "", candidates)

    except Exception as e:
        logger.error(f"[recommender] error: {e}")
        state.error = str(e)
        state.candidate_songs = []

    return state


def _parse_recommendations(content: str, candidates: list[dict]) -> list[SongRecommendation]:
    """Extract JSON recommendations from the LLM's response.

    Falls back to the catalog data for any None fields the LLM omits.
    """
    catalog = _get_catalog_lookup()

    # Build id → candidate lookup for fallback
    candidate_by_id = {c["id"]: c for c in candidates if c.get("id")}

    try:
        text = content.strip()
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                clean = part.strip()
                if clean.startswith("{") or clean.startswith("json"):
                    text = clean[4:].strip() if clean.startswith("json") else clean
                    break

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
                sid = int(r["id"])
                # Merge: catalog > candidate > LLM output (fallback chain)
                base = {**(catalog.get(sid, {})), **(candidate_by_id.get(sid, {})), **r}

                rec = SongRecommendation(
                    id=sid,
                    title=str(base.get("title") or r.get("title", "")),
                    artist=str(base.get("artist") or r.get("artist", "")),
                    genre=str(base.get("genre") or "unknown"),
                    mood=str(base.get("mood") or "unknown"),
                    energy=_safe_float(base.get("energy"), 0.5),
                    valence=_safe_float(base.get("valence"), 0.5),
                    danceability=_safe_float(base.get("danceability"), 0.5),
                    acousticness=_safe_float(base.get("acousticness"), 0.5),
                    score=_safe_float(base.get("score"), 0.5),
                    confidence=_safe_float(base.get("confidence"), 0.5),
                    explanation=str(r.get("explanation") or ""),
                    v1_score=_safe_float(r.get("v1_score"), None) if r.get("v1_score") is not None else None,
                )
                results.append(rec)
            except Exception as e:
                logger.warning(f"[recommender] failed to parse recommendation: {e} -- {r}")

        return results

    except Exception as e:
        logger.warning(f"[recommender] JSON parse error: {e}")
        return []
