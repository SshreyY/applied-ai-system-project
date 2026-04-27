"""
In-memory session manager.

Maps session_id -> AgentState dict. Each session is one conversation.
In production this would be Redis/DB; for now a plain dict is fine
since we're running a single server process.
"""

import uuid
import logging
from backend.state import AgentState

logger = logging.getLogger(__name__)

_sessions: dict[str, dict] = {}


def create_session() -> str:
    session_id = str(uuid.uuid4())
    state = AgentState(session_id=session_id)
    _sessions[session_id] = state.model_dump()
    logger.info(f"[session] created session_id={session_id}")
    return session_id


def get_session(session_id: str) -> dict | None:
    return _sessions.get(session_id)


def get_or_create_session(session_id: str) -> dict:
    """Return the session if it exists; otherwise create a fresh one with the same ID.

    This handles server restarts (hot-reload in dev) where in-memory sessions
    are wiped but the frontend still holds the old session_id.
    """
    existing = _sessions.get(session_id)
    if existing is not None:
        return existing
    state = AgentState(session_id=session_id)
    _sessions[session_id] = state.model_dump()
    logger.info(f"[session] auto-recreated session_id={session_id} (server restart recovery)")
    return _sessions[session_id]


def update_session(session_id: str, state: dict) -> None:
    _sessions[session_id] = state


def clear_session(session_id: str) -> bool:
    if session_id in _sessions:
        del _sessions[session_id]
        logger.info(f"[session] cleared session_id={session_id}")
        return True
    return False


def list_sessions() -> list[str]:
    return list(_sessions.keys())
