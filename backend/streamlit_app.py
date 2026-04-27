"""
Streamlit quick-test app for VibeFinder Agent backend.

Uses LangGraph stream() to show live node-by-node execution.

Run with:
    streamlit run backend/streamlit_app.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import streamlit as st
from backend.graph import compiled_graph
from backend.state import AgentState, ConversationMessage
from backend.session import create_session, get_session, update_session
from backend.langfuse_callback import flush as langfuse_flush

# Flush Langfuse traces when the Streamlit session ends
import atexit
atexit.register(langfuse_flush)

st.set_page_config(page_title="VibeFinder Agent — Dev Console", page_icon="🎵", layout="wide")
st.title("🎵 VibeFinder Agent — Backend Dev Console")
st.caption("Direct agent access — no frontend required")

# Session management
if "session_id" not in st.session_state:
    st.session_state.session_id = create_session()

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
with col2:
    if st.button("New Session"):
        st.session_state.session_id = create_session()
        st.rerun()

st.divider()

chat_col, profile_col = st.columns([2, 1])

# Node display config
NODE_LABELS = {
    "router":           ("🔀", "Intent Router"),
    "profile_builder":  ("👤", "Profile Builder"),
    "recommender":      ("🎯", "Recommender (ReAct)"),
    "bias_auditor":     ("🔍", "Bias Auditor"),
    "feedback_handler": ("💬", "Feedback Handler"),
    "general_chat":     ("💡", "General Chat"),
}

with chat_col:
    st.subheader("Chat")

    state = get_session(st.session_state.session_id) or {}
    messages = state.get("messages", [])
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    recs = state.get("final_recommendations") or state.get("candidate_songs") or []
    if recs:
        st.subheader("Recommendations")
        for i, rec in enumerate(recs, 1):
            song_id = rec.get("id", i)
            with st.expander(f"#{i} {rec['title']} by {rec['artist']} — {rec['confidence']*100:.0f}% match"):
                cols = st.columns(3)
                cols[0].metric("Genre", rec["genre"])
                cols[1].metric("Mood", rec["mood"])
                cols[2].metric("Energy", f"{rec['energy']:.2f}")
                st.write(f"**Why:** {rec['explanation']}")
                if rec.get("v1_score") is not None:
                    st.caption(f"V1 score: {rec['v1_score']:.2f} / 7.5")

                # Feedback buttons
                fb_cols = st.columns(4)
                feedback_map = {
                    "👍 Liked": "liked",
                    "👎 Disliked": "disliked",
                    "➕ More like this": "more_like_this",
                    "➖ Less like this": "less_like_this",
                }
                for col, (label, rating) in zip(fb_cols, feedback_map.items()):
                    btn_key = f"fb_{song_id}_{rating}_{i}"
                    if col.button(label, key=btn_key):
                        # Record feedback directly into session state
                        existing = get_session(st.session_state.session_id) or {}
                        entries = existing.get("feedback_entries", [])
                        entries.append({
                            "song_id": song_id,
                            "rating": rating,
                            "comment": f"Button: {label}",
                        })
                        # Update profile: liked → add to liked_ids, disliked → exclude
                        profile = existing.get("user_profile", {})
                        if rating in ("liked", "more_like_this"):
                            liked = profile.get("liked_song_ids", [])
                            if song_id not in liked:
                                liked.append(song_id)
                            profile["liked_song_ids"] = liked
                        elif rating in ("disliked", "less_like_this"):
                            excluded = profile.get("excluded_song_ids", [])
                            if song_id not in excluded:
                                excluded.append(song_id)
                            profile["excluded_song_ids"] = excluded
                        existing["feedback_entries"] = entries
                        existing["user_profile"] = profile
                        update_session(st.session_state.session_id, existing)
                        # Log to Langfuse
                        from backend.langfuse_callback import log_feedback_score
                        log_feedback_score(st.session_state.session_id, song_id, rating)
                        st.toast(f"{label} recorded for **{rec['title']}**")
                        st.rerun()

        audit = state.get("bias_audit") or {}
        if isinstance(audit, dict) and audit.get("issues"):
            st.warning("**Bias Auditor flagged:**\n" + "\n".join(f"- {i}" for i in audit["issues"]))

    if prompt := st.chat_input("Describe your vibe, ask for songs, or give feedback..."):
        with st.chat_message("user"):
            st.write(prompt)

        # Build initial state
        existing = get_session(st.session_state.session_id)
        if existing:
            agent_state = AgentState(**existing)
        else:
            agent_state = AgentState(session_id=st.session_state.session_id)

        agent_state.messages.append(ConversationMessage(role="user", content=prompt))
        agent_state.error = None

        # Stream with live node updates
        with st.status("Agent processing...", expanded=True) as status:
            final_state = None

            try:
                for chunk in compiled_graph.stream(
                    agent_state.model_dump(),
                    config={"recursion_limit": 25},
                    stream_mode="updates",
                ):
                    for node_name, node_output in chunk.items():
                        icon, label = NODE_LABELS.get(node_name, ("⚙️", node_name))
                        details = []

                        intent = node_output.get("intent")
                        if intent:
                            details.append(f"Intent: **{intent}**")

                        profile = node_output.get("user_profile", {})
                        if profile:
                            profile_bits = [
                                f"{k}={v}" for k, v in profile.items()
                                if v is not None and v != [] and k not in ("excluded_song_ids", "liked_song_ids")
                            ]
                            if profile_bits:
                                details.append("Profile: " + ", ".join(profile_bits))

                        tools = node_output.get("tool_calls_made")
                        if tools:
                            details.append("Tools called: " + ", ".join(f"`{t}`" for t in tools))

                        candidates = node_output.get("candidate_songs") or []
                        if candidates:
                            details.append(f"Candidates found: **{len(candidates)}** songs")

                        audit = node_output.get("bias_audit") or {}
                        if isinstance(audit, dict):
                            if audit.get("passed") is True:
                                details.append("✅ Bias audit: **passed**")
                            elif audit.get("passed") is False:
                                issues = audit.get("issues", [])
                                details.append(f"⚠️ Bias audit: **{len(issues)} issue(s)** — " + "; ".join(issues[:2]))

                        final_recs = node_output.get("final_recommendations") or []
                        if final_recs:
                            details.append(f"Final recommendations: **{len(final_recs)}** songs")

                        error = node_output.get("error")
                        if error:
                            details.append(f"❌ Error: {error}")

                        detail_str = "\n\n".join(details) if details else ""
                        st.write(f"{icon} **{label}**" + (f"\n\n{detail_str}" if detail_str else ""))

                        final_state = node_output

                status.update(label="✅ Done", state="complete", expanded=False)

            except Exception as e:
                status.update(label=f"❌ Error: {e}", state="error")
                st.error(str(e))

        # Save updated state and refresh
        if final_state is not None:
            # Merge back into existing session state
            existing = get_session(st.session_state.session_id) or {}
            existing.update(final_state)
            update_session(st.session_state.session_id, existing)

        st.rerun()

with profile_col:
    st.subheader("User Profile")
    state = get_session(st.session_state.session_id) or {}
    profile = state.get("user_profile", {})

    if any(v is not None and v != [] for v in profile.values()):
        for field in ["genre", "mood", "activity", "energy", "valence", "danceability", "acousticness"]:
            val = profile.get(field)
            if val is not None:
                if isinstance(val, float):
                    st.metric(field.capitalize(), f"{val:.2f}")
                else:
                    st.metric(field.capitalize(), str(val))

        excluded = profile.get("excluded_song_ids", [])
        liked = profile.get("liked_song_ids", [])
        if excluded:
            st.caption(f"Excluded song IDs: {excluded}")
        if liked:
            st.caption(f"Liked song IDs: {liked}")
    else:
        st.caption("No preferences captured yet. Start chatting!")

    st.divider()
    st.subheader("Debug Info")
    tools = state.get("tool_calls_made", [])
    if tools:
        st.caption(f"Tools called: {', '.join(tools)}")

    intent = state.get("intent")
    if intent:
        st.caption(f"Last intent: {intent}")

    error = state.get("error")
    if error:
        st.error(f"Error: {error}")

    with st.expander("Raw state JSON"):
        st.json(state)
