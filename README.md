# VibeFinder Agent

> CodePath AI110 Final Project (Week 9)

## Original Project

**Base project:** Music Recommender Simulation — Module 3, AI110

The original system is a rule-based content recommender that scores songs by weighted proximity to a user profile across six attributes: genre, mood, energy, valence, danceability, and acousticness. It ranked 18 songs and returned the top matches for 4 hardcoded user profiles.

**Documented limitations of the V1 system:**
- Binary genre matching — genre match is all-or-nothing (+2.0 or 0), no soft similarity
- Genre lock-in — the +2.0 genre bonus dominates rankings regardless of mood/energy fit
- No diversity enforcement — top-5 results can all be the same genre
- No feedback loop — preferences are static per run, system cannot adapt
- Cold-start problem — requires a fully specified numeric profile, cannot handle natural language
- No confidence scoring — all recommendations are returned with equal implicit confidence

The original source lives in `backend/recommender_v1.py` and `backend/data/songs.csv`. The V1 scoring formula is preserved as a callable tool inside the new agent system for comparison purposes.

---

## What We're Building

**VibeFinder Agent** is a full-stack agentic music recommender that replaces the rule-based system with a LangGraph-powered AI agent. It understands your vibe through natural language, searches and scores songs using multiple tools, self-critiques for bias, and adapts to your feedback in real time.

Full documentation, architecture diagrams, setup instructions, sample interactions, design decisions, testing summary, and responsible AI reflection will be added here as the project is built across Phases 1–7.

---

*This README will be fully expanded in Phase 7.*
