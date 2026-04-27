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
import base64
import json
import logging
import os
import queue
import threading
import time
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from backend import session as session_mgr
from backend.session import get_or_create_session
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
            # Group for readability: show unique tools, candidate count
            unique = list(dict.fromkeys(tools))  # preserve order, dedupe
            detail = f"{', '.join(unique)} → {count} candidates"

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

    # Auto-recreate session if missing (handles server hot-reloads in dev)
    existing = get_or_create_session(session_id)

    # Build initial state with the new user message
    state = AgentState(**existing)
    state.messages.append(ConversationMessage(role="user", content=message))
    state.error = None
    state_dict = state.model_dump()

    # Thread → async bridge
    q: queue.Queue = queue.Queue()

    def _run_graph() -> None:
        from backend.langfuse_callback import flush as langfuse_flush
        # Start with a full copy so merging partial node outputs stays correct
        last_state: dict = dict(state_dict)
        try:
            for chunk in compiled_graph.stream(state_dict, config={"recursion_limit": 25}):
                for node_name, node_output in chunk.items():
                    q.put(_extract_event(node_name, node_output))
                    # Merge node output into accumulated state (nodes return partial dicts)
                    if isinstance(node_output, dict):
                        last_state.update(node_output)

            # Persist final state to session
            session_mgr.update_session(session_id, last_state)

            # Flush Langfuse so traces are available immediately in the UI
            langfuse_flush()

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


# ---------------------------------------------------------------------------
# Langfuse trace proxy  –  GET /api/traces?session_id=...
# ---------------------------------------------------------------------------

# Simple TTL cache — key: session_id, value: (timestamp, result_dict)
_trace_cache: dict[str, tuple[float, dict]] = {}
_TRACE_CACHE_TTL = 30  # seconds


def _langfuse_auth_header() -> str | None:
    pub = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sec = os.getenv("LANGFUSE_SECRET_KEY", "")
    if not pub or not sec or pub.startswith("pk-lf-..."):
        return None
    token = base64.b64encode(f"{pub}:{sec}".encode()).decode()
    return f"Basic {token}"


def _fmt_trace(t: dict) -> dict:
    """Flatten a Langfuse trace dict to what the frontend needs."""
    observations: list[dict] = t.get("observations", [])
    # Sort observations by startTime so they appear in execution order
    observations.sort(key=lambda o: o.get("startTime", ""))

    total_input_tokens  = sum(
        (o.get("usage") or {}).get("input", 0) or 0 for o in observations
    )
    total_output_tokens = sum(
        (o.get("usage") or {}).get("output", 0) or 0 for o in observations
    )

    steps = []
    for o in observations:
        usage = o.get("usage") or {}
        steps.append({
            "id":           o.get("id"),
            "name":         o.get("name", ""),
            "type":         o.get("type", ""),          # GENERATION | SPAN | EVENT
            "model":        o.get("model", ""),
            "startTime":    o.get("startTime"),
            "endTime":      o.get("endTime"),
            "latencyMs":    o.get("latency"),            # ms (Langfuse v2)
            "inputTokens":  usage.get("input"),
            "outputTokens": usage.get("output"),
            "input":        o.get("input"),
            "output":       o.get("output"),
            "level":        o.get("level", "DEFAULT"),
        })

    # When observations aren't fetched (lazy mode), fall back to trace-level token counts
    # Langfuse returns these as top-level fields on the trace object
    usage = t.get("usage") or {}
    fallback_input  = usage.get("input") or usage.get("inputTokens") or 0
    fallback_output = usage.get("output") or usage.get("outputTokens") or 0

    return {
        "id":               t.get("id"),
        "name":             t.get("name", ""),
        "timestamp":        t.get("timestamp"),
        "latencyMs":        t.get("latency"),
        "inputTokens":      total_input_tokens or fallback_input,
        "outputTokens":     total_output_tokens or fallback_output,
        "totalCost":        t.get("totalCost"),
        "input":            t.get("input"),
        "output":           t.get("output"),
        "tags":             t.get("tags", []),
        "steps":            steps,
    }


async def _fetch_traces_only(base_url: str, auth: str, params: dict) -> list[dict]:
    """Fetch trace summaries from Langfuse — no observation calls (avoids N+1 rate-limit hits)."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{base_url}/api/public/traces",
            params=params,
            headers={"Authorization": auth},
        )
        resp.raise_for_status()
        raw_traces: list[dict] = resp.json().get("data", [])

    # Format without observations — they are loaded lazily per trace
    result = [_fmt_trace({**t, "observations": []}) for t in raw_traces]
    result.sort(key=lambda t: t.get("timestamp") or "", reverse=True)
    return result


@router.get("/api/traces")
async def get_traces(session_id: str, force: bool = False) -> dict[str, Any]:
    """Proxy Langfuse trace data — keeps API keys server-side.

    Strategy:
    1. Try filtering by sessionId (works if langfuse_session_id tag is sent)
    2. If empty, fall back to the 30 most recent traces for the project
       (helps when session tagging is not yet flushed or misconfigured)
    """
    auth = _langfuse_auth_header()
    if not auth:
        return {
            "configured": False, "traces": [], "sessionId": session_id,
            "error": "Langfuse is not configured — add LANGFUSE_PUBLIC_KEY / SECRET_KEY / BASE_URL to backend/.env",
        }

    # Return cached result if it's fresh (avoids hammering Langfuse API)
    now = time.time()
    if not force and session_id in _trace_cache:
        cached_at, cached_result = _trace_cache[session_id]
        age = now - cached_at
        if age < _TRACE_CACHE_TTL:
            cached_result["cachedFor"] = round(_TRACE_CACHE_TTL - age)
            return cached_result

    base_url = os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com").rstrip("/")

    try:
        # Strategy 1: filter by session (1 API call)
        traces = await _fetch_traces_only(
            base_url, auth, {"sessionId": session_id, "limit": 50}
        )
        source = "session"

        # Strategy 2: if session filter returned nothing, show recent project traces
        if not traces:
            traces = await _fetch_traces_only(base_url, auth, {"limit": 30})
            source = "recent"

        result = {
            "configured": True,
            "traces": traces,
            "sessionId": session_id,
            "source": source,   # "session" | "recent"
            "cachedFor": None,
            "error": None,
        }
        _trace_cache[session_id] = (now, result)
        return result

    except httpx.HTTPStatusError as e:
        logger.warning(f"[traces] Langfuse API {e.response.status_code}: {e.response.text[:200]}")
        if session_id in _trace_cache:
            _, stale = _trace_cache[session_id]
            stale["error"] = f"Rate-limited (HTTP {e.response.status_code}) — showing cached data."
            return stale
        return {
            "configured": True, "traces": [], "sessionId": session_id,
            "source": None,
            "error": f"Langfuse rate-limited (HTTP {e.response.status_code}). Wait a moment and try again.",
        }
    except Exception as e:
        logger.error(f"[traces] unexpected error: {e}")
        return {"configured": True, "traces": [], "sessionId": session_id, "source": None, "error": str(e)}


# Observations cache — key: trace_id, value: (timestamp, list)
_obs_cache: dict[str, tuple[float, list]] = {}


@router.get("/api/observations")
async def get_observations(trace_id: str) -> dict[str, Any]:
    """Lazily fetch observations (spans/LLM calls) for one trace — called on expand."""
    auth = _langfuse_auth_header()
    if not auth:
        return {"steps": [], "error": "Langfuse not configured"}

    now = time.time()
    if trace_id in _obs_cache:
        cached_at, steps = _obs_cache[trace_id]
        if now - cached_at < 120:  # 2-minute cache for observations
            return {"steps": steps, "error": None}

    base_url = os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{base_url}/api/public/observations",
                params={"traceId": trace_id, "limit": 50},
                headers={"Authorization": auth},
            )
            resp.raise_for_status()
            raw = resp.json().get("data", [])

        raw.sort(key=lambda o: o.get("startTime", ""))
        steps = []
        for o in raw:
            usage = o.get("usage") or {}
            steps.append({
                "id":           o.get("id"),
                "name":         o.get("name", ""),
                "type":         o.get("type", ""),
                "model":        o.get("model", ""),
                "startTime":    o.get("startTime"),
                "endTime":      o.get("endTime"),
                "latencyMs":    o.get("latency"),
                "inputTokens":  usage.get("input"),
                "outputTokens": usage.get("output"),
                "input":        o.get("input"),
                "output":       o.get("output"),
                "level":        o.get("level", "DEFAULT"),
            })

        _obs_cache[trace_id] = (now, steps)
        return {"steps": steps, "error": None}

    except httpx.HTTPStatusError as e:
        return {"steps": [], "error": f"Langfuse rate-limited (HTTP {e.response.status_code}). Try again shortly."}
    except Exception as e:
        return {"steps": [], "error": str(e)}
