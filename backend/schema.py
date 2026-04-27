"""
Strawberry GraphQL schema for VibeFinder Agent.

Queries:   session, songs, songDetail
Mutations: createSession, sendMessage, sendFeedback, clearSession
"""

import csv
import logging
import os
import strawberry
from typing import Optional
from strawberry.types import Info

from backend import session as session_mgr
from backend.graph import run_agent

logger = logging.getLogger(__name__)

CATALOG_PATH = os.getenv("CATALOG_PATH", "backend/data/songs.csv")


# ---------------------------------------------------------------------------
# GraphQL Types
# ---------------------------------------------------------------------------

@strawberry.type
class SongType:
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@strawberry.type
class RecommendationType:
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    valence: float
    danceability: float
    acousticness: float
    score: float
    confidence: float
    explanation: str
    v1_score: Optional[float]


@strawberry.type
class UserProfileType:
    genre: Optional[str]
    mood: Optional[str]
    energy: Optional[float]
    valence: Optional[float]
    danceability: Optional[float]
    acousticness: Optional[float]
    activity: Optional[str]
    excluded_song_ids: list[int]
    liked_song_ids: list[int]


@strawberry.type
class MessageType:
    role: str
    content: str


@strawberry.type
class SessionType:
    session_id: str
    user_profile: UserProfileType
    messages: list[MessageType]
    final_recommendations: list[RecommendationType]
    conflict_detected: bool
    conflict_description: Optional[str]


@strawberry.type
class AgentResponseType:
    session_id: str
    recommendations: list[RecommendationType]
    assistant_message: Optional[str]
    conflict_detected: bool
    conflict_description: Optional[str]
    bias_issues: list[str]
    tools_called: list[str]
    error: Optional[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_catalog() -> list[dict]:
    songs = []
    try:
        with open(CATALOG_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["id"] = int(row["id"])
                row["energy"] = float(row["energy"])
                row["tempo_bpm"] = float(row["tempo_bpm"])
                row["valence"] = float(row["valence"])
                row["danceability"] = float(row["danceability"])
                row["acousticness"] = float(row["acousticness"])
                songs.append(row)
    except FileNotFoundError:
        logger.error(f"Catalog not found at {CATALOG_PATH}")
    return songs


def _state_to_session_type(state: dict) -> SessionType:
    profile = state.get("user_profile", {})
    return SessionType(
        session_id=state["session_id"],
        user_profile=UserProfileType(
            genre=profile.get("genre"),
            mood=profile.get("mood"),
            energy=profile.get("energy"),
            valence=profile.get("valence"),
            danceability=profile.get("danceability"),
            acousticness=profile.get("acousticness"),
            activity=profile.get("activity"),
            excluded_song_ids=profile.get("excluded_song_ids", []),
            liked_song_ids=profile.get("liked_song_ids", []),
        ),
        messages=[MessageType(role=m["role"], content=m["content"]) for m in state.get("messages", [])],
        final_recommendations=[
            RecommendationType(**{k: r[k] for k in RecommendationType.__annotations__})
            for r in state.get("final_recommendations", [])
        ],
        conflict_detected=state.get("conflict_detected", False),
        conflict_description=state.get("conflict_description"),
    )


def _recs_from_state(state: dict) -> list[RecommendationType]:
    recs = state.get("final_recommendations") or state.get("candidate_songs") or []
    result = []
    for r in recs:
        try:
            result.append(RecommendationType(
                id=r["id"],
                title=r["title"],
                artist=r["artist"],
                genre=r["genre"],
                mood=r["mood"],
                energy=r["energy"],
                valence=r["valence"],
                danceability=r["danceability"],
                acousticness=r["acousticness"],
                score=r["score"],
                confidence=r["confidence"],
                explanation=r["explanation"],
                v1_score=r.get("v1_score"),
            ))
        except Exception as e:
            logger.warning(f"[schema] failed to parse rec: {e}")
    return result


def _last_assistant_message(state: dict) -> str | None:
    messages = state.get("messages", [])
    for m in reversed(messages):
        if m["role"] == "assistant":
            return m["content"]
    return None


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

@strawberry.type
class Query:
    @strawberry.field
    def session(self, session_id: str) -> Optional[SessionType]:
        state = session_mgr.get_session(session_id)
        if not state:
            return None
        return _state_to_session_type(state)

    @strawberry.field
    def songs(self, genre: Optional[str] = None, mood: Optional[str] = None) -> list[SongType]:
        catalog = _load_catalog()
        filtered = [
            s for s in catalog
            if (genre is None or s["genre"].lower() == genre.lower())
            and (mood is None or s["mood"].lower() == mood.lower())
        ]
        return [SongType(**s) for s in filtered]

    @strawberry.field
    def song_detail(self, id: int) -> Optional[SongType]:
        catalog = _load_catalog()
        for s in catalog:
            if s["id"] == id:
                return SongType(**s)
        return None


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------

@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_session(self) -> str:
        return session_mgr.create_session()

    @strawberry.mutation
    def send_message(self, session_id: str, message: str) -> AgentResponseType:
        existing = session_mgr.get_session(session_id)
        if existing is None:
            return AgentResponseType(
                session_id=session_id,
                recommendations=[],
                assistant_message=None,
                conflict_detected=False,
                conflict_description=None,
                bias_issues=[],
                tools_called=[],
                error=f"Session {session_id} not found. Call createSession first.",
            )

        try:
            result_state = run_agent(session_id, message, existing)
            session_mgr.update_session(session_id, result_state)

            recs = _recs_from_state(result_state)
            assistant_msg = _last_assistant_message(result_state)
            bias_audit = result_state.get("bias_audit") or {}
            bias_issues = bias_audit.get("issues", []) if isinstance(bias_audit, dict) else []

            return AgentResponseType(
                session_id=session_id,
                recommendations=recs,
                assistant_message=assistant_msg,
                conflict_detected=result_state.get("conflict_detected", False),
                conflict_description=result_state.get("conflict_description"),
                bias_issues=bias_issues,
                tools_called=result_state.get("tool_calls_made", []),
                error=result_state.get("error"),
            )

        except Exception as e:
            logger.error(f"[schema] send_message error: {e}")
            return AgentResponseType(
                session_id=session_id,
                recommendations=[],
                assistant_message=None,
                conflict_detected=False,
                conflict_description=None,
                bias_issues=[],
                tools_called=[],
                error=str(e),
            )

    @strawberry.mutation
    def send_feedback(self, session_id: str, song_id: int, rating: str) -> AgentResponseType:
        valid_ratings = {"liked", "disliked", "more_like_this", "less_like_this"}
        if rating not in valid_ratings:
            return AgentResponseType(
                session_id=session_id,
                recommendations=[],
                assistant_message=None,
                conflict_detected=False,
                conflict_description=None,
                bias_issues=[],
                tools_called=[],
                error=f"Invalid rating '{rating}'. Must be one of: {', '.join(valid_ratings)}",
            )

        rating_message = {
            "liked": f"I liked song #{song_id}",
            "disliked": f"I didn't like song #{song_id}, please don't recommend it again",
            "more_like_this": f"More songs like #{song_id} please",
            "less_like_this": f"Fewer songs like #{song_id}",
        }[rating]

        return self.send_message(session_id=session_id, message=rating_message)

    @strawberry.mutation
    def clear_session(self, session_id: str) -> bool:
        return session_mgr.clear_session(session_id)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

schema = strawberry.Schema(query=Query, mutation=Mutation)
