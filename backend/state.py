"""
Pydantic state models for the VibeFinder Agent.

AgentState is the top-level LangGraph state object passed between nodes.
All other models are sub-schemas used within it.
"""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """Extracted music preferences built up over a conversation."""
    genre: Optional[str] = None
    mood: Optional[str] = None
    energy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    valence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    danceability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    acousticness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    activity: Optional[str] = None
    excluded_song_ids: list[int] = Field(default_factory=list)
    liked_song_ids: list[int] = Field(default_factory=list)


class SongRecommendation(BaseModel):
    """A single song recommendation with explanation and confidence."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    valence: float
    danceability: float
    acousticness: float
    score: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
    v1_score: Optional[float] = None


class FeedbackEntry(BaseModel):
    """A single piece of user feedback on a recommendation."""
    song_id: int
    rating: Literal["liked", "disliked", "more_like_this", "less_like_this"]
    comment: Optional[str] = None


class ConversationMessage(BaseModel):
    """A single turn in the conversation history."""
    role: Literal["user", "assistant"]
    content: str


class BiasAuditResult(BaseModel):
    """Result from the Bias Auditor node."""
    passed: bool
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class AgentState(BaseModel):
    """
    Top-level LangGraph state. Passed between every node in the graph.
    Fields are updated in-place as the graph traverses nodes.
    """
    session_id: str
    messages: list[ConversationMessage] = Field(default_factory=list)
    user_profile: UserProfile = Field(default_factory=UserProfile)
    intent: Optional[Literal["recommend", "song_question", "feedback", "general_chat"]] = None
    candidate_songs: list[SongRecommendation] = Field(default_factory=list)
    final_recommendations: list[SongRecommendation] = Field(default_factory=list)
    feedback_entries: list[FeedbackEntry] = Field(default_factory=list)
    bias_audit: Optional[BiasAuditResult] = None
    conflict_detected: bool = False
    conflict_description: Optional[str] = None
    error: Optional[str] = None
    tool_calls_made: list[str] = Field(default_factory=list)

    class Config:
        # Allow extra fields so LangGraph can attach its own metadata
        extra = "allow"
