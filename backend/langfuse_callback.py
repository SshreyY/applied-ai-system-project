"""
Langfuse instrumentation for VibeFinder Agent (v4 SDK).

Pattern per the v4 docs:
  - get_client()      → singleton client, reads env vars automatically
  - CallbackHandler() → zero-arg constructor; session_id set via metadata at invocation

Required env vars (.env):
  LANGFUSE_PUBLIC_KEY
  LANGFUSE_SECRET_KEY
  LANGFUSE_BASE_URL   (e.g. https://us.cloud.langfuse.com)
"""

import logging
import os

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    """Lazy init — checked on every call so hot-reloads and delayed load_dotenv() work."""
    global _client
    if _client is not None:
        return _client

    pub = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sec = os.getenv("LANGFUSE_SECRET_KEY", "")
    if not pub or not sec or pub == "pk-lf-..." or sec == "sk-lf-...":
        return None

    try:
        from langfuse import get_client
        _client = get_client()
        logger.info(
            "[langfuse] Tracing enabled → %s",
            os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
        )
    except Exception as exc:
        logger.warning("[langfuse] Init failed (non-fatal): %s", exc)

    return _client


# ── LangChain / LangGraph callback ───────────────────────────────────────────

def get_callback_handler(session_id: str, trace_name: str = "vibefinder-turn"):
    """
    Return (handler, metadata) for use in LangChain config:
        config={"callbacks": [handler], "metadata": metadata, "run_name": trace_name}

    In Langfuse v4 the CallbackHandler takes NO constructor args.
    session_id is attached via metadata["langfuse_session_id"] at invocation time.
    Returns (None, {}) when Langfuse is not configured.
    """
    if _get_client() is None:
        return None, {}
    try:
        from langfuse.langchain import CallbackHandler
        handler = CallbackHandler()
        metadata = {
            "langfuse_session_id": session_id,
            "langfuse_tags": ["vibefinder", trace_name],
        }
        return handler, metadata
    except Exception as exc:
        logger.warning("[langfuse] CallbackHandler error: %s", exc)
        return None, {}


# ── Score helpers ─────────────────────────────────────────────────────────────

def log_feedback_score(session_id: str, song_id: int, rating: str) -> None:
    """
    Log user feedback as a numeric Langfuse score on the session.

    Mapping: liked=1.0  more_like_this=0.8  less_like_this=0.2  disliked=0.0
    """
    client = _get_client()
    if client is None:
        return
    score_map = {"liked": 1.0, "more_like_this": 0.8, "less_like_this": 0.2, "disliked": 0.0}
    value = score_map.get(rating, 0.5)
    try:
        client.create_score(
            name="user_feedback",
            value=value,
            comment=f"song_id={song_id} rating={rating}",
            session_id=session_id,
        )
    except Exception as exc:
        logger.warning("[langfuse] create_score error: %s", exc)


def log_score(trace_id: str, name: str, value: float, comment: str = "") -> None:
    """Log an eval score (e.g. genre_relevance) to a specific trace."""
    client = _get_client()
    if client is None:
        return
    try:
        client.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment,
        )
    except Exception as exc:
        logger.warning("[langfuse] create_score error: %s", exc)


def flush() -> None:
    """Flush pending events at shutdown — prevents traces being dropped."""
    client = _get_client()
    if client is not None:
        try:
            client.flush()
        except Exception:
            pass
