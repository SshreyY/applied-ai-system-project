"""
Server-Sent Events (SSE) streaming endpoint for VibeFinder Agent.

POST /stream  →  text/event-stream
  Body: { "session_id": str, "message": str }
  Events:
    {"type": "node",  "node": str, "icon": str, "label": str, "detail": str|null}
    {"type": "done",  "assistantMessage": str|null, "recommendations": [...],
                      "biasIssues": [...], "toolsCalled": [...], "error": str|null}
    {"type": "error", "error": str}
"""

import asyncio
import json
import logging
import queue
import threading

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from backend import session as session_mgr
from backend.graph import compiled_graph
from backend.state import AgentState, ConversationMessage

logger = logging.getLogger(__name__)
router = APIRouter()

# Human-readable metadata for each LangGraph node
NODE_META: dict[str, dict] = {
    "router":            {"icon": "🔍", "label": "Detecting intent"},
    "profile_builder":   {"icon": "👤", "label": "Building your profile"},
    "recommender":       {"icon": "🎵", "label": "Searching for songs"},
    "bias_auditor":      {"icon": "⚖️",  "label": "Auditing diversity"},
    "finalize_response": {"icon": "✨", "label": "Crafting response"},
    "feedback_handler":  {"icon": "💬", "label": "Processing feedback"},
    "general_chat":      {"icon": "💬", "label": "Thinking"},
}


def _extract_event(node_name: str, node_output: dict) -> dict:
    """Pull key info from a node's output dict into a human-readable event."""
    meta = NODE_META.get(node_name, {"icon": "⚙️", "label": node_name})
    detail: str | None = None

    if node_name == "router":
        intent = node_output.get("intent", "unknown")
        detail = f"Intent → {intent}"

    elif node_name == "profile_builder":
        p = node_output.get("user_profile", {})
        parts = []
        if p.get("genre"):    parts.append(f"genre={p['genre']}")
        if p.get("mood"):     parts.append(f"mood={p['mood']}")
        if p.get("energy") is not None: parts.append(f"energy={p['energy']}")
        if p.get("activity"): parts.append(f"activity={p['activity']}")
        detail = ", ".join(parts) if parts else None

    elif node_name == "recommender":
        tools = node_output.get("tool_calls_made", [])
        count = len(node_output.get("candidate_songs", []))
        if tools:
            detail = f"{', '.join(tools)} → {count} candidates"

    elif node_name == "bias_auditor":
        audit = node_output.get("bias_audit") or {}
        if isinstance(audit, dict):
            if audit.get("passed"):
                detail = "Passed ✓"
            else:
                issues = audit.get("issues", [])
                rerank = node_output.get("rerank_count", 0)
                tag = f" (re-rank #{rerank})" if rerank else ""
                detail = (issues[0] + tag) if issues else "Issues found, re-ranking"

    elif node_name == "finalize_response":
        count = len(node_output.get("final_recommendations", []))
        detail = f"{count} tracks ready"

    elif node_name == "feedback_handler":
        entries = node_output.get("feedback_entries", [])
        detail = f"{len(entries)} feedback entries recorded"

    return {
        "type": "node",
        "node": node_name,
        "icon": meta["icon"],
        "label": meta["label"],
        "detail": detail,
    }


def _to_frontend_rec(r: dict) -> dict:
    """Convert snake_case state dict to camelCase for the frontend."""
    return {
        "id":           r.get("id"),
        "title":        r.get("title", ""),
        "artist":       r.get("artist", ""),
        "genre":        r.get("genre", "unknown"),
        "mood":         r.get("mood", "unknown"),
        "energy":       r.get("energy", 0.5),
        "valence":      r.get("valence", 0.5),
        "danceability": r.get("danceability", 0.5),
        "acousticness": r.get("acousticness", 0.5),
        "score":        r.get("score", 0.5),
        "confidence":   r.get("confidence", 0.5),
        "explanation":  r.get("explanation", ""),
        "v1Score":      r.get("v1_score"),
    }


@router.post("/stream")
async def stream_agent(request: Request) -> StreamingResponse:
    body = await request.json()
    session_id: str = body.get("session_id", "")
    message: str = body.get("message", "")

    existing = session_mgr.get_session(session_id)
    if existing is None:
        async def _err():
            yield f"data: {json.dumps({'type': 'error', 'error': 'Session not found — call createSession first.'})}\n\n"
        return StreamingResponse(_err(), media_type="text/event-stream")

    # Build initial state with the new user message
    state = AgentState(**existing)
    state.messages.append(ConversationMessage(role="user", content=message))
    state.error = None
    state_dict = state.model_dump()

    # Thread → async bridge
    q: queue.Queue = queue.Queue()

    def _run_graph() -> None:
        last_state = state_dict
        try:
            for chunk in compiled_graph.stream(state_dict, config={"recursion_limit": 25}):
                for node_name, node_output in chunk.items():
                    q.put(_extract_event(node_name, node_output))
                    last_state = node_output  # track latest full state

            # Persist final state to session
            session_mgr.update_session(session_id, last_state)

            # Build "done" payload
            raw_recs = (
                last_state.get("final_recommendations")
                or last_state.get("candidate_songs")
                or []
            )
            recs = [_to_frontend_rec(r) for r in raw_recs]

            assistant_msg: str | None = None
            for m in reversed(last_state.get("messages", [])):
                if isinstance(m, dict) and m.get("role") == "assistant":
                    assistant_msg = m.get("content")
                    break

            audit = last_state.get("bias_audit") or {}
            bias_issues = audit.get("issues", []) if isinstance(audit, dict) else []

            q.put({
                "type": "done",
                "assistantMessage": assistant_msg,
                "recommendations": recs,
                "biasIssues": bias_issues,
                "toolsCalled": last_state.get("tool_calls_made", []),
                "error": last_state.get("error"),
            })

        except Exception as e:
            logger.error(f"[stream] graph error: {e}")
            q.put({"type": "error", "error": str(e)})
        finally:
            q.put(None)  # sentinel — tells the async generator to stop

    threading.Thread(target=_run_graph, daemon=True).start()

    async def _event_gen():
        loop = asyncio.get_event_loop()
        while True:
            event = await loop.run_in_executor(None, q.get)
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        _event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )
