# Model Card: VibeFinder Agent

## 1. Model Name

**VibeFinder Agent**: an agentic music recommender built on LangGraph, Groq (Llama 4 Scout), and ChromaDB.

---

## 2. Intended Use and Non-Intended Use

This is a learning project and portfolio demo. It is meant to show how a rule-based recommender from Module 3 can be evolved into a full agentic AI system with natural language input, tool calling, self-critique, feedback loops, and real observability. It is good for demonstrating agentic AI patterns like ReAct loops, conditional graph edges, and LLM-powered self-correction.

It is not for real music discovery at scale. The catalog has 18 songs, which is not nearly enough for a real application. The confidence scores are LLM-generated estimates and should not be treated as calibrated probabilities. Do not use this as a production recommendation system and do not treat it as representative of how Spotify or any commercial system works.

---

## 3. How the Model Works

When you type a message like "hip hop songs for working out," the system runs it through a LangGraph `StateGraph` with seven nodes that each do a specific job. The Router classifies what you are asking for. The Profile Builder uses the LLM to extract structured audio preferences from your natural language: genre, mood, energy, valence, danceability, acousticness, activity. The Recommender runs a ReAct loop where it decides which of the seven tools to call: it can search the song catalog by structured attributes, run a semantic similarity search against ChromaDB embeddings, expand your genre to similar ones, map an activity to audio attributes, score candidates using the original V1 rule-based formula, check for diversity across the results, and detect contradictions in your preferences.

Once the Recommender has candidates, the Bias Auditor reviews the list for genre lock-in and mood uniformity. If the list fails the audit and no re-rank has happened yet, a conditional back-edge sends execution back to the Recommender for a second pass. When the audit passes, the Finalize Response node uses the LLM to write a natural conversational reply. Every step of this process is streamed to the browser in real time using Server-Sent Events so you can see exactly what the agent is doing as it does it.

---

## 4. Data

The catalog is 18 songs stored in `songs.csv`, the same dataset from Module 3 with attributes including genre, mood, energy, valence, danceability, and acousticness. In addition to the CSV, ChromaDB stores sentence-transformer embeddings of each song's mood and vibe description, which is what powers the semantic `vibe_search` tool.

The data has the same problems it had in Module 3. It skews toward Western pop, hip-hop, lofi, and indie. There is nothing representing K-pop, Afrobeats, Latin, classical, or non-English music. Most numeric attributes were hand-assigned based on one person's perception of how those genres typically sound, so users from different cultural backgrounds or with non-standard genre preferences will get worse results. The embeddings are generated from short text descriptions, not actual audio, which means vibe search is matching descriptions of how songs feel rather than what they actually sound like.

---

## 5. Strengths

The system works best when the user describes what they want in natural language and gives some context — an activity, a genre, a mood, or a combination. Someone who says "hip hop for working out" will get candidates that come from catalog search filtered by genre, activity context that maps "working out" to high energy and danceability, and genre expansion that widens the search to r&b and trap. That is a meaningfully better result than the Module 3 system which would have required the user to fill in all those fields manually.

The Bias Auditor is a genuine improvement over the original system. The Module 3 recommender had no diversity enforcement at all and would happily return five nearly identical lofi songs to anyone with a lofi preference. The agent catches that itself and tries again.

Every recommendation card shows both the AI-generated confidence score and the original V1 rule-based score from Module 3, which runs as a tool inside the agent. This lets you compare what the AI thinks versus what the deterministic formula says, which is useful for building intuition about where they agree and where they diverge.

The Langfuse observability tab makes the system transparent in a way most AI apps are not. You can open the Traces tab and see exactly which LLM calls were made, what the inputs and outputs were, how many tokens were used, and how long each step took. That is genuinely useful for understanding and trusting what the system is doing.

---

## 6. Limitations and Bias

**Catalog size is the biggest problem.** With 18 songs across many genres, some genres have only one representative. If you ask for rock, you will always get the same song at the top because there is nothing else to compete with it. No amount of agentic reasoning fixes a data problem.

**The LLM inherits stereotypes from its training data.** "Working out" almost always maps to hip-hop or EDM. "Chill" almost always maps to lofi. "Studying" maps to lofi or ambient. These are reasonable defaults but they are not universally correct, and users whose preferences don't match those defaults will notice the agent making assumptions about what they want.

**Cold-start bias.** When a user gives only a genre or only a mood, the agent fills in missing attributes with genre-typical defaults. This can produce recommendations that feel confident but are actually just the most stereotypical interpretation of that genre. A user who says "jazz" but actually wants high-energy jazz fusion will probably get something slow and mellow because that is what the LLM thinks jazz sounds like.

**Confidence scores are not calibrated.** In practice they cluster between 0.75 and 0.95 regardless of how good the actual match is. A 90% confidence score does not mean the song is 90% likely to be something you enjoy. It is a relative ranking within one response, not an absolute measure of quality.

**The Bias Auditor only checks two things.** It catches genre lock-in and mood uniformity but it does not check for artist diversity, energy uniformity, decade bias, or cultural representation. You could get three songs from the same artist and the auditor would not flag it.

---

## 7. Potential Misuse

The catalog is fixed and contains only normal songs, so there is very little misuse risk at this scale. The main concern is over-trusting the output. Because the agent gives natural language explanations for its recommendations and shows confidence percentages, it can feel more authoritative than it is. Users who treat those confidence scores as ground truth rather than estimates will sometimes get confidently wrong recommendations.

The feedback loop is session-scoped and in-memory, so there is no way for a user to manipulate the system across sessions or affect other users' results. Rate-limit abuse of the Langfuse proxy is mitigated by a 30-second TTL cache on the backend and a frontend cooldown on the Refresh button.

---

## 8. Evaluation

Evaluation in this project happened at three levels: automated self-critique inside the agent, observability through Langfuse, and manual testing across different input types.

**Bias Auditor (automated self-critique).** The Bias Auditor node runs after every recommendation pass and checks the candidate list for genre lock-in and mood uniformity. If it fails, it generates a description of what went wrong and a conditional back-edge sends the agent back to the Recommender with that context so it can try again differently. Over manual testing, the auditor triggered a re-rank in 4 out of 5 cold-start sessions where the user gave a vague single-genre prompt. In every case where a re-rank happened, the second candidate list passed the audit. The re-rank cap of 1 was enforced 100% of the time, which is what prevented the runaway recursion loop that happened early in development.

**Confidence scoring and V1 baseline comparison.** Every recommendation includes a `confidence` score generated by the LLM during ranking, and a `v1_score` computed by the original Module 3 formula running as a tool inside the agent. These appear side by side on every recommendation card so you can compare what the AI thinks versus what the deterministic formula says. When they disagree it usually means the song fits the mood description but not the numeric audio profile, or vice versa. Confidence scores averaged around 0.82 across all test sessions.

**Langfuse tracing and observability.** Every node execution, LLM call, and tool call is traced to Langfuse with full input, output, token counts, and latency. The built-in Traces tab in the frontend surfaces this data directly in the app where you can expand any trace and see each span in execution order. This was not just a debugging tool during development; it was how I caught the LLM output parsing bug where the model returned a raw JSON array instead of the expected tool-call schema. Without Langfuse I would have been guessing. With it I could see exactly what the LLM returned on the failing call and fix the parser accordingly. If I had more time I would have used Langfuse's scoring and evaluation API to build a feedback loop where user ratings get analyzed and fed back into improving the agent's prompts over time. Right now Langfuse observes everything but nothing closes back into the agent automatically.

**Manual testing across agent paths.** I tested six input types covering every major agent path: pure vibe description with no genre, activity-based cold start, bias auditor re-rank trigger, feedback loop dislike and exclude, conflict detection, and general chat intent. 5 out of 6 produced the expected behavior. The one that struggled was a single-word prompt ("beats") with no other context, where the agent returned reasonable results but with low confidence because it had almost nothing to build a profile from.

---

## 9. Future Work

The highest priority improvement is connecting to a real music catalog through an API like Spotify or Last.fm. 18 songs is enough to demonstrate the architecture but not enough to be genuinely useful. With a real catalog the diversity enforcement and semantic search would actually matter.

After that, I want to close the Langfuse evaluation loop properly. Right now Langfuse traces everything but none of that data feeds back into the agent. The next step is building an evaluation agent that analyzes user feedback and recommendation quality, scores the agent's responses against those signals, and surfaces that information so the prompts and tool logic can be improved over time. That is the difference between an observable system and a self-improving one.

Persistent sessions stored in Redis or a lightweight database would also make the feedback loop meaningful across conversations instead of resetting on every page load.

---

## 10. Personal Reflection

Building this as my first agentic AI application taught me more about system design than any tutorial I have read. You cannot just write code that calls an LLM and expect it to work reliably. You have to think about where state lives, what happens when an LLM returns malformed output, how to prevent runaway loops, and how to make failure modes visible instead of silent. Those are not LLM problems, they are engineering problems, and they were the hardest part of this project.

The recursion bug was the one that hit hardest. The agent was looping 10,000 times and the logs looked completely normal because the bias auditor was doing exactly what it was supposed to do. The problem was that I was incrementing `rerank_count` inside a conditional edge function, where LangGraph does not persist state changes. The counter was stuck at 0 on every iteration so the auditor always thought it was on the first attempt. Moving one line into the node body fixed it. That kind of bug teaches you to think carefully about execution boundaries in agentic frameworks, not just what the code says but where the runtime actually commits things.

Working with Cursor as my AI pair programmer throughout this project was one of the most interesting parts of the experience. I used it for everything from scaffolding the initial LangGraph graph structure to debugging the Langfuse proxy rate-limit issues. The collaboration felt less like autocomplete and more like working with someone who is very fast at writing code but needs you to provide the judgment and domain context. The AI knew how to connect components and generate boilerplate quickly. I had to bring the knowledge of why a specific LangGraph behavior was causing a bug, what Groq's tool-calling constraints were, and when a suggested implementation would break under real conditions. When those two things worked together well, development moved fast. When I let the AI generate large blocks of code without reviewing them carefully, that is when bugs got introduced.

The most helpful suggestion was making the Bias Auditor a self-critique node inside the LangGraph graph with a conditional back-edge to the Recommender, rather than a post-processing filter outside the graph. That architectural decision is what makes re-ranking a native part of the agent's reasoning loop rather than an external patch applied after the fact. It is a cleaner and more extensible design and I would not have framed it that way on my own.

The most flawed suggestion was in the streaming implementation, where the AI wrote `last_state = node_output` inside the streaming loop, assuming each LangGraph node emits a full state snapshot. Nodes actually emit partial updates, so this replaced the full accumulated state with just the last node's output. The result was that the backend saved an incomplete session after every turn, meaning the second message in a conversation would silently lose the user profile built in the first message. The fix was one word, `last_state.update(node_output)` instead of `last_state = node_output`, but finding it required tracing through symptoms that looked like a completely different problem. That experience taught me that AI assistance speeds up development significantly but it does not replace the judgment needed to catch subtle system-level bugs. You still have to understand the system deeply enough to know when something is wrong.

What I want to do next is go deeper with Langfuse. I only used it for observability here, but the real opportunity is a full evaluation pipeline where feedback signals are analyzed by a second agent and fed back to improve the recommender over time. That is the difference between a system that logs what it does and a system that learns from what it does. This project showed me exactly what that would require to build.
