"""
Intent Router node -- classifies user message into one of four intents.

Routes to: recommend | song_question | feedback | general_chat
"""

import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from backend.state import AgentState, ConversationMessage

logger = logging.getLogger(__name__)

_llm: ChatGoogleGenerativeAI | None = None


def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
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
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=latest.content),
        ])
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
