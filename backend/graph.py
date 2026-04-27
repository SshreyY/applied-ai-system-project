"""
LangGraph StateGraph assembly for VibeFinder Agent.

Graph structure:
  START
    └─> router
          ├─ recommend ──> profile_builder ──> recommender ──> bias_auditor
          │                                                          ├─ rerank ──> recommender (loop)
          │                                                          └─ finalize ──> END
          ├─ feedback  ──> feedback_handler ──> recommender ──> bias_auditor ──> ...
          ├─ song_question ──> recommender ──> bias_auditor ──> ...
          └─ general_chat ──> END
"""

import logging
from langgraph.graph import StateGraph, START, END
from backend.state import AgentState
from backend.nodes.router import router_node
from backend.nodes.profile_builder import profile_builder_node
from backend.nodes.recommender import recommender_node
from backend.nodes.bias_auditor import bias_auditor_node, should_rerank
from backend.nodes.feedback import feedback_node

logger = logging.getLogger(__name__)


def _route_intent(state: AgentState) -> str:
    """Conditional edge after router: dispatch to the right subflow."""
    intent = state.intent or "general_chat"
    logger.info(f"[graph] routing intent={intent}")
    return intent


def _finalize_response(state: AgentState) -> AgentState:
    """Use the LLM to write a natural conversational reply that includes the recommendations."""
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage
    from backend.state import ConversationMessage
    import os
    import json

    recs = state.final_recommendations
    user_request = next(
        (m.content for m in reversed(state.messages) if m.role == "user"), ""
    )

    if not recs:
        state.messages.append(ConversationMessage(
            role="assistant",
            content="I couldn't find songs matching your vibe right now — try describing your mood, activity, or a genre you enjoy!",
        ))
        return state

    recs_summary = [
        {
            "title": r.title,
            "artist": r.artist,
            "genre": r.genre,
            "mood": r.mood,
            "energy": r.energy,
            "explanation": r.explanation,
        }
        for r in recs[:5]
    ]

    try:
        llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.7,
        )
        response = llm.invoke([
            SystemMessage(content=(
                "You are VibeFinder, a friendly music recommendation assistant. "
                "The user just asked for songs and you found great matches. "
                "Write a warm, conversational 2-3 sentence reply that:\n"
                "- Directly responds to what the user said (don't start with 'Here are')\n"
                "- Naturally mentions the top 1-2 song titles and why they fit\n"
                "- Ends with a short invite for feedback (e.g. 'Let me know if you want more like these!')\n"
                "Keep it under 60 words. Sound like a knowledgeable friend, not a list-maker."
            )),
            HumanMessage(content=(
                f"User said: \"{user_request}\"\n\n"
                f"Songs found:\n{json.dumps(recs_summary, indent=2)}"
            )),
        ])
        msg = response.content.strip()
    except Exception as e:
        logger.error(f"[finalize_response] LLM error: {e}")
        # Graceful fallback
        names = " and ".join(f"{r.title} by {r.artist}" for r in recs[:2])
        msg = f"I found {len(recs)} tracks that fit — starting with {names}. Let me know what you think!"

    state.messages.append(ConversationMessage(role="assistant", content=msg))
    return state


def _general_chat_response(state: AgentState) -> AgentState:
    """Simple pass-through for non-recommendation intents."""
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage
    import os

    latest = next((m for m in reversed(state.messages) if m.role == "user"), None)
    if not latest:
        return state

    try:
        llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.7,
        )
        response = llm.invoke([
            SystemMessage(content=(
                "You are VibeFinder, a friendly music recommendation assistant. "
                "Answer the user's question helpfully and briefly. "
                "If they ask for music recommendations, let them know you can help with that too."
            )),
            HumanMessage(content=latest.content),
        ])
        from backend.state import ConversationMessage
        state.messages.append(ConversationMessage(role="assistant", content=response.content))
    except Exception as e:
        logger.error(f"[general_chat] error: {e}")
        state.error = str(e)

    return state


def build_graph() -> StateGraph:
    """Build and compile the VibeFinder LangGraph StateGraph."""

    # Use dict-based state for LangGraph compatibility
    graph = StateGraph(dict)

    # --- Add nodes ---
    graph.add_node("router", lambda s: router_node(AgentState(**s)).model_dump())
    graph.add_node("profile_builder", lambda s: profile_builder_node(AgentState(**s)).model_dump())
    graph.add_node("recommender", lambda s: recommender_node(AgentState(**s)).model_dump())
    graph.add_node("bias_auditor", lambda s: bias_auditor_node(AgentState(**s)).model_dump())
    graph.add_node("finalize_response", lambda s: _finalize_response(AgentState(**s)).model_dump())
    graph.add_node("feedback_handler", lambda s: feedback_node(AgentState(**s)).model_dump())
    graph.add_node("general_chat", lambda s: _general_chat_response(AgentState(**s)).model_dump())

    # --- Entry point ---
    graph.add_edge(START, "router")

    # --- Router conditional dispatch ---
    graph.add_conditional_edges(
        "router",
        lambda s: AgentState(**s).intent or "general_chat",
        {
            "recommend": "profile_builder",
            "song_question": "recommender",
            "feedback": "feedback_handler",
            "general_chat": "general_chat",
        },
    )

    # --- Recommend flow ---
    graph.add_edge("profile_builder", "recommender")
    graph.add_edge("recommender", "bias_auditor")

    # --- Bias auditor gate: rerank or finalize ---
    graph.add_conditional_edges(
        "bias_auditor",
        lambda s: should_rerank(AgentState(**s)),
        {
            "rerank": "recommender",
            "finalize": "finalize_response",
        },
    )
    graph.add_edge("finalize_response", END)

    # --- Feedback flow: re-recommend after updating profile ---
    graph.add_edge("feedback_handler", "recommender")

    # --- General chat ends immediately ---
    graph.add_edge("general_chat", END)

    return graph.compile()


# Module-level compiled graph instance (recursion_limit prevents runaway loops)
compiled_graph = build_graph()


def run_agent(session_id: str, user_message: str, existing_state: dict | None = None) -> dict:
    """
    Convenience function: run the agent for one turn.

    Args:
        session_id: Unique session identifier.
        user_message: The user's latest message.
        existing_state: Previous AgentState dict (for continuing a conversation).

    Returns:
        Updated AgentState as a dict.
    """
    from backend.state import ConversationMessage

    if existing_state:
        state = AgentState(**existing_state)
    else:
        state = AgentState(session_id=session_id)

    state.messages.append(ConversationMessage(role="user", content=user_message))
    state.error = None

    result = compiled_graph.invoke(
        state.model_dump(),
        config={"recursion_limit": 25},
    )
    return result
