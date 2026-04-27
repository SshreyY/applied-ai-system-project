"""
Streamlit quick-test app for VibeFinder Agent backend.

Calls the agent graph directly (no FastAPI layer) for rapid iteration.

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
from backend.graph import run_agent
from backend.session import create_session, get_session, update_session

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

# Two-column layout: chat left, profile right
chat_col, profile_col = st.columns([2, 1])

with chat_col:
    st.subheader("Chat")

    # Display conversation history
    state = get_session(st.session_state.session_id) or {}
    messages = state.get("messages", [])
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Display recommendations if any
    recs = state.get("final_recommendations") or state.get("candidate_songs") or []
    if recs:
        st.subheader("Recommendations")
        for i, rec in enumerate(recs, 1):
            with st.expander(f"#{i} {rec['title']} by {rec['artist']} — {rec['confidence']*100:.0f}% match"):
                cols = st.columns(3)
                cols[0].metric("Genre", rec["genre"])
                cols[1].metric("Mood", rec["mood"])
                cols[2].metric("Energy", f"{rec['energy']:.2f}")
                st.write(f"**Why:** {rec['explanation']}")
                if rec.get("v1_score") is not None:
                    st.caption(f"V1 score: {rec['v1_score']:.2f} / 7.5")

        # Bias audit
        audit = state.get("bias_audit") or {}
        if isinstance(audit, dict) and audit.get("issues"):
            st.warning("**Bias Auditor flagged:**\n" + "\n".join(f"- {i}" for i in audit["issues"]))

    # Chat input
    if prompt := st.chat_input("Describe your vibe, ask for songs, or give feedback..."):
        with st.spinner("Agent thinking..."):
            existing = get_session(st.session_state.session_id)
            result = run_agent(st.session_state.session_id, prompt, existing)
            update_session(st.session_state.session_id, result)
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
