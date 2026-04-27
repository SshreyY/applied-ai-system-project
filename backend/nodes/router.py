"""
Intent Router node -- classifies user message into one of four intents.

Routes to: recommend | song_question | feedback | general_chat
"""

import os
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from backend.state import AgentState, ConversationMessage
from backend.langfuse_callback import get_callback_handler

logger = logging.getLogger(__name__)

_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
    return _llm


SYSTEM_PROMPT = """You are an intent classifier for a music recommendation chatbot.

Classify the user's message into EXACTLY one of these intents:
- recommend: User wants song/playlist recommendations or describes what they want to listen to
- song_question: User asks about a specific song, artist, or music fact
- feedback: User gives feedback on a previous recommendation (likes, dislikes, "more like this", etc.)
- general_chat: Anything else (greetings, off-topic questions, etc.)

Respond with ONLY the intent word, nothing else. No punctuation, no explanation."""


def router_node(state: AgentState) -> AgentState:
    """Classify the latest user message and set state.intent."""
    if not state.messages:
        state.intent = "general_chat"
        return state

    latest = state.messages[-1]
    if latest.role != "user":
        state.intent = "general_chat"
        return state

    try:
        llm = _get_llm()
        cb = get_callback_handler(state.session_id, "router")
        kwargs = {"config": {"callbacks": [cb]}} if cb else {}
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=latest.content),
        ], **kwargs)
        raw = response.content.strip().lower()
        valid = {"recommend", "song_question", "feedback", "general_chat"}
        intent = raw if raw in valid else "general_chat"
        logger.info(f"[router] intent={intent} for message='{latest.content[:60]}'")
        state.intent = intent
    except Exception as e:
        logger.error(f"[router] LLM error: {e}")
        state.intent = "general_chat"
        state.error = str(e)

    return state
