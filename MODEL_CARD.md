# Model Card — VibeFinder Agent

> Responsible AI documentation for the VibeFinder Agent system.
> Addresses limitations, potential misuse, testing surprises, and AI collaboration.

---

## System Overview

**Task:** Conversational music recommendation via agentic LLM reasoning.
**Model used:** `meta-llama/llama-4-scout-17b-16e-instruct` via Groq API.
**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (local, for ChromaDB vibe search).
**Catalog:** 18 songs stored in `songs.csv`.

---

## Limitations and Biases

### Catalog bias (most significant)
The song catalog contains 18 songs curated by hand. Every recommendation made by this system is constrained to this set. The catalog skews toward pop, hip-hop, lofi, and indie — genres well-represented in the original Module 3 project. Users asking for classical, jazz, country, or non-English music will receive poor or no matches. This is a data bias, not a model bias, but it is the largest single source of inaccurate recommendations.

### LLM genre and mood stereotyping
The LLM (Llama 4 Scout) inherits the biases of its training data. In practice this means:
- "Working out" almost always maps to high energy and hip-hop/EDM, even if the user might prefer low-tempo metal or acoustic rock for focus during exercise.
- "Chill" maps almost exclusively to lofi, even though many genres (jazz, acoustic folk, ambient) are equally valid.
- Vibe descriptions using Western cultural references produce more confident scores than non-Western references.

### Cold-start bias
When a user provides only a genre or only a mood (not both), the agent tends to fill in missing attributes with genre-typical defaults (e.g., hip-hop → high energy, high danceability). This can produce over-confident recommendations that miss what the user actually wanted.

### Confidence score inflation
Confidence scores (0.0–1.0) are LLM-generated, not calibrated against any ground truth. In practice they cluster between 0.75 and 0.95, making it hard to distinguish a strong match from a marginal one. They are useful as relative rankings within a response, but should not be treated as absolute reliability measures.

### Bias Auditor limitations
The self-critique node checks for genre lock-in (≥ 4 songs from same genre) and mood uniformity. It does not check for:
- Artist diversity (all songs could be from the same artist)
- Energy uniformity
- Over-representation of any single decade or cultural origin
- Demographic stereotyping in activity-genre mappings

---

## Potential Misuse

| Scenario | Risk | Mitigation |
|---|---|---|
| Feeding manipulated prompts to extract unintended content | Low — the catalog is fixed; the agent can only recommend from 18 known songs | No direct mitigation needed at this scale |
| Using the feedback loop to game rankings | A user could repeatedly "like" one song to dominate all future sessions | Session-scoped memory means this only affects one session; no persistent manipulation |
| Trusting confidence scores as ground truth | Scores are LLM estimates, not calibrated probabilities; over-reliance could mislead users | UI labels scores as "match %" not "accuracy"; V1 rule-based score shown as comparison |
| Rate-limit abuse of the Langfuse proxy | Rapid manual requests could bypass the 30s TTL cache and hit Langfuse limits | Cache enforced on backend; frontend "Refresh" button has a 30s cooldown |

This is a low-risk hobby/educational application. The catalog is fixed and benign. The main responsible-AI concern is **over-trusting AI output** — users should treat recommendations as suggestions, not authoritative judgments.

---

## What Surprised Us During Testing

**1. The recursion bug was invisible in the logs.**
The agent was looping indefinitely (10,000+ iterations) before crashing. The logs showed the bias auditor repeatedly saying "issues found, re-ranking" — which looked correct — but `rerank_count` was stuck at 0 because LangGraph doesn't persist state changes made inside conditional edge functions. The fix (moving the increment into the node body) was one line, but finding it required understanding an undocumented subtlety of LangGraph's execution model.

**2. The LLM was more brittle than expected with structured output.**
Llama 4 Scout occasionally returned a raw JSON array `[{...}]` instead of the expected tool-call format `{"recommendations": [...]}`. This crashed the parser silently — the agent returned 0 recommendations, the bias auditor failed, and the loop ran again. The fix was to add `_safe_float()` guards and to enrich candidates from the CSV before passing them to the LLM, reducing the chance of `None` values triggering a schema mismatch.

**3. Rate limits were a product design problem, not just an ops problem.**
Hitting Langfuse's free-tier rate limit (429) in the Traces tab forced us to design a lazy-loading system (fetch observations only on expand) and a TTL cache on the proxy. What started as an API error became a UI pattern that's actually better UX: the panel loads instantly instead of waiting for N observation fetches.

**4. Confidence scores don't degrade gracefully.**
When the LLM was under load and returned truncated output, some songs got a confidence of `0.0` and others got `null`. These appeared as low-match songs at the bottom of the list — which looked like poor recommendations rather than parsing errors. Adding a `_safe_float()` default of `0.5` for null scores made failures less visible but also masked real problems. A better solution would be to surface a "data quality" indicator on cards with imputed values.

---

## AI Collaboration Reflection

This project was built with continuous AI assistance (Cursor Agent). Here is an honest account of where that collaboration worked well and where it failed.

### One instance where AI gave a genuinely helpful suggestion

When implementing the Bias Auditor, the AI suggested making it a **self-critique node** inside the graph rather than a post-processing filter outside it. This architectural suggestion was non-obvious: by placing the auditor inside the graph with a conditional back-edge to the recommender, the re-ranking becomes part of the agent's own reasoning loop rather than an external correction layer. This is a cleaner, more agentic design and made the system easier to extend (e.g., adding a second audit criterion later only required changing the auditor node, not the graph structure).

### One instance where AI's suggestion was flawed

During the LangGraph streaming implementation, the AI initially wrote:

```python
for chunk in compiled_graph.stream(state_dict):
    for node_name, node_output in chunk.items():
        last_state = node_output  # WRONG
```

This replaced the full accumulated state with each node's partial output dict. The AI's assumption — that each node emits a complete state snapshot — was incorrect for LangGraph's streaming mode. The actual fix was to **merge** node outputs incrementally:

```python
last_state.update(node_output)  # CORRECT
```

This bug caused the backend to save an incomplete session state after each turn, meaning the second message in a conversation would lose the user profile built in the first turn. The AI did eventually identify and fix the bug when confronted with the symptom (session state missing fields), but the initial suggestion introduced a subtle, hard-to-reproduce defect.

---

## Summary

VibeFinder Agent is an educational demonstration of agentic AI patterns. Its recommendations are constrained to a small curated catalog and should not be treated as comprehensive music discovery. The system is transparent by design: every recommendation includes a confidence score, a rule-based baseline score, and a live trace of the AI's reasoning. Users are encouraged to use the feedback buttons and the Traces tab to develop their own intuition for when and why AI recommendations are trustworthy.

---

## What This Project Says About Me as an AI Engineer

I care about the full system, not just the model. A lot of AI projects stop at "I got the LLM to return something reasonable." VibeFinder Agent goes further — the agent has a self-critique loop that catches its own mistakes, a feedback mechanism that actually changes future outputs, real observability through Langfuse, and a frontend that makes all of that visible to the user in real time. I built those things not because they were required, but because a system without them isn't really trustworthy.

I also think carefully about failure modes before they happen. The `_safe_float()` guard, the session auto-recovery on server restart, the TTL cache on the Langfuse proxy, the re-rank cap on the bias auditor — none of these were in the original plan. They came from actually running the system and watching it break in specific ways, then fixing the root cause rather than masking the symptom. That's the kind of engineering mindset I want to bring to every project.

Finally, I'm not afraid to start from something real and improve it. This project began as a rule-based scoring function from Module 3 and ended as a multi-node agentic system with streaming, observability, and feedback loops. The V1 formula is still in there — running as a tool, stamping a baseline score on every recommendation — because I think it's worth keeping the original work visible rather than throwing it away. That respect for what came before, combined with the drive to push it further, is how I approach building things.
