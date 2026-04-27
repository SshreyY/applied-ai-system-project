"""
Langfuse instrumentation helper for VibeFinder Agent.

Provides a shared CallbackHandler and trace/span helpers used by every
node and tool in the graph. All LLM calls and tool invocations are
automatically logged as spans inside a parent trace per session turn.
"""

import os
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

_langfuse_enabled = False
_langfuse_client = None


def _init():
    global _langfuse_enabled, _langfuse_client
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key or public_key.startswith("pk-lf-..."):
        logger.info("[langfuse] No valid keys found — tracing disabled.")
        return

    try:
        from langfuse import Langfuse
        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        _langfuse_enabled = True
        logger.info("[langfuse] Tracing enabled.")
    except ImportError:
        logger.warning("[langfuse] langfuse package not installed — tracing disabled.")
    except Exception as e:
        logger.warning(f"[langfuse] Init failed (non-fatal): {e}")


_init()


def get_callback_handler(session_id: str, trace_name: str = "vibefinder-turn"):
    """
    Returns a Langfuse CallbackHandler for LangChain/LangGraph LLM calls.
    Returns None if Langfuse is not configured.
    """
    if not _langfuse_enabled:
        return None
    try:
        from langfuse.callback import CallbackHandler
        return CallbackHandler(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            session_id=session_id,
            trace_name=trace_name,
        )
    except Exception as e:
        logger.warning(f"[langfuse] CallbackHandler error: {e}")
        return None


def log_score(trace_id: str, name: str, value: float, comment: str = "") -> None:
    """Log an eval score to a Langfuse trace (e.g. recommendation relevance)."""
    if not _langfuse_enabled or not _langfuse_client:
        return
    try:
        _langfuse_client.score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment,
        )
    except Exception as e:
        logger.warning(f"[langfuse] score error: {e}")


def log_feedback_score(session_id: str, song_id: int, rating: str) -> None:
    """
    Log user feedback as a Langfuse score on the most recent trace for this session.
    rating: 'liked' -> 1.0, 'disliked' -> 0.0, 'more_like_this' -> 0.8, 'less_like_this' -> 0.2
    """
    if not _langfuse_enabled or not _langfuse_client:
        return
    score_map = {
        "liked": 1.0,
        "disliked": 0.0,
        "more_like_this": 0.8,
        "less_like_this": 0.2,
    }
    value = score_map.get(rating, 0.5)
    try:
        _langfuse_client.score(
            name="user_feedback",
            value=value,
            comment=f"song_id={song_id} rating={rating}",
            session_id=session_id,
        )
    except Exception as e:
        logger.warning(f"[langfuse] feedback score error: {e}")


def flush() -> None:
    """Flush pending Langfuse events (call at app shutdown)."""
    if _langfuse_enabled and _langfuse_client:
        try:
            _langfuse_client.flush()
        except Exception:
            pass
