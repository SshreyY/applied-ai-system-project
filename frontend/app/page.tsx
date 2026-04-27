"use client";

import { useEffect, useRef, useState } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { MessageBubble } from "@/components/MessageBubble";
import { RecommendationCard } from "@/components/RecommendationCard";
import { ProfileSidebar } from "@/components/ProfileSidebar";
import { gqlRequest } from "@/lib/graphql/client";
import { CREATE_SESSION, SEND_MESSAGE, SEND_FEEDBACK, GET_SESSION } from "@/lib/graphql/queries";
import { AgentResponse, SessionState, Message, SongRecommendation, UserProfile } from "@/lib/graphql/types";

const EMPTY_PROFILE: UserProfile = {
  genre: null, mood: null, energy: null, valence: null,
  danceability: null, acousticness: null, activity: null,
  likedSongIds: [], excludedSongIds: [],
};

export default function Home() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [profile, setProfile] = useState<UserProfile>(EMPTY_PROFILE);
  const [recs, setRecs] = useState<SongRecommendation[]>([]);
  const [biasIssues, setBiasIssues] = useState<string[]>([]);
  const [toolsCalled, setToolsCalled] = useState<string[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Create session on mount
  useEffect(() => {
    gqlRequest<{ createSession: string }>(CREATE_SESSION)
      .then((data) => {
        setBackendError(null);
        setSessionId(data.createSession);
      })
      .catch(() =>
        setBackendError(
          "Cannot reach the backend at http://localhost:8000 — run: uvicorn backend.main:app --reload"
        )
      );
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, recs]);

  /** Apply an AgentResponse to local state — messages are appended, never replaced */
  function applyResponse(res: AgentResponse) {
    if (res.assistantMessage) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: res.assistantMessage! },
      ]);
    }
    if (res.recommendations?.length) setRecs(res.recommendations);
    if (res.biasIssues) setBiasIssues(res.biasIssues);
    if (res.toolsCalled) setToolsCalled(res.toolsCalled);
    if (res.error) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Something went wrong: ${res.error}` },
      ]);
    }
  }

  /** Fetch the session to sync only the profile sidebar — never overwrites local chat messages */
  async function refreshProfile(sid: string) {
    try {
      const data = await gqlRequest<{ session: SessionState }>(GET_SESSION, { sessionId: sid });
      if (data.session) {
        setProfile(data.session.userProfile);
      }
    } catch {
      // non-blocking — sidebar is nice-to-have
    }
  }

  async function sendMessage() {
    const msg = input.trim();
    if (!msg || loading || !sessionId) return;
    setInput("");
    setLoading(true);

    // Optimistic user bubble
    setMessages((prev) => [...prev, { role: "user", content: msg }]);

    try {
      const data = await gqlRequest<{ sendMessage: AgentResponse }>(SEND_MESSAGE, {
        sessionId,
        message: msg,
      });
      applyResponse(data.sendMessage);
      await refreshProfile(sessionId);
    } catch (err) {
      setBackendError(String(err));
    } finally {
      setLoading(false);
    }
  }

  async function handleFeedback(songId: number, rating: string) {
    if (!sessionId) return;
    try {
      const data = await gqlRequest<{ sendFeedback: AgentResponse }>(SEND_FEEDBACK, {
        sessionId,
        songId,
        rating,
      });
      applyResponse(data.sendFeedback);
      await refreshProfile(sessionId);
    } catch {
      // non-blocking
    }
  }

  async function startNewSession() {
    const data = await gqlRequest<{ createSession: string }>(CREATE_SESSION);
    setSessionId(data.createSession);
    setMessages([]);
    setProfile(EMPTY_PROFILE);
    setRecs([]);
    setBiasIssues([]);
    setToolsCalled([]);
  }

  return (
    <div className="flex h-screen bg-background text-foreground">
      {/* Main chat area */}
      <div className="flex flex-1 flex-col min-w-0">
        {/* Header */}
        <header className="flex items-center justify-between border-b border-border px-4 py-3">
          <div className="flex items-center gap-2">
            <span className="text-xl">🎵</span>
            <h1 className="font-bold tracking-tight">VibeFinder Agent</h1>
          </div>
          <button
            onClick={() => setSidebarOpen((o) => !o)}
            className="rounded-md border border-border px-3 py-1 text-sm hover:bg-muted"
          >
            {sidebarOpen ? "Hide" : "Show"} Profile
          </button>
        </header>

        {/* Backend error banner */}
        {backendError && (
          <div className="border-b border-destructive/40 bg-destructive/10 px-4 py-2 text-xs text-destructive">
            ⚠️ {backendError}
          </div>
        )}

        {/* Messages */}
        <ScrollArea className="flex-1 px-4 py-3">
          {messages.length === 0 && !loading && (
            <div className="flex flex-col items-center justify-center gap-2 py-16 text-center text-muted-foreground">
              <span className="text-4xl">🎧</span>
              <p className="text-sm">
                Describe your vibe, mood, or what you&apos;re doing — I&apos;ll find the perfect songs.
              </p>
              <p className="text-xs opacity-60">
                Try: &quot;chill lofi for studying&quot; or &quot;energetic hip-hop for working out&quot;
              </p>
            </div>
          )}

          <div className="space-y-3">
            {messages.map((msg, i) => (
              <MessageBubble key={i} message={msg} />
            ))}
          </div>

          {loading && (
            <div className="mt-3 flex items-center gap-2 text-sm text-muted-foreground">
              <span className="animate-pulse">🎵</span>
              <span>Agent is thinking…</span>
            </div>
          )}

          {/* Recommendations */}
          {!loading && recs.length > 0 && (
            <div className="mt-4 space-y-2">
              <Separator />
              <p className="text-xs font-medium text-muted-foreground">Recommendations</p>
              <div className="space-y-3">
                {recs.map((rec, i) => (
                  <RecommendationCard
                    key={rec.id}
                    rec={rec}
                    rank={i + 1}
                    onFeedback={handleFeedback}
                  />
                ))}
              </div>
              {biasIssues.length > 0 && (
                <div className="rounded-md border border-yellow-500/40 bg-yellow-500/10 p-3 text-xs text-yellow-300">
                  <p className="font-medium mb-1">⚠️ Bias Auditor flagged:</p>
                  <ul className="list-disc pl-4 space-y-0.5">
                    {biasIssues.map((issue, i) => (
                      <li key={i}>{issue}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          <div ref={bottomRef} />
        </ScrollArea>

        {/* Input bar */}
        <div className="border-t border-border p-3">
          <form
            onSubmit={(e) => { e.preventDefault(); sendMessage(); }}
            className="flex gap-2"
          >
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Describe your vibe, ask for songs, or give feedback…"
              disabled={loading || !sessionId}
              className="flex-1 rounded-md border border-border bg-muted/50 px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={loading || !input.trim() || !sessionId}
              className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-opacity disabled:opacity-50"
            >
              Send
            </button>
          </form>
        </div>
      </div>

      {/* Sidebar */}
      {sidebarOpen && (
        <div className="w-64 shrink-0 border-l border-border">
          <ProfileSidebar
            profile={profile}
            feedbackEntries={[]}
            toolsCalled={toolsCalled}
            messageCount={messages.length}
            onNewSession={startNewSession}
          />
        </div>
      )}
    </div>
  );
}
