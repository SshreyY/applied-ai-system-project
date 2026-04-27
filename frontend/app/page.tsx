"use client";

import { useEffect, useRef, useState } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { MessageBubble } from "@/components/MessageBubble";
import { RecommendationCard } from "@/components/RecommendationCard";
import { ProfileSidebar } from "@/components/ProfileSidebar";
import { ThinkingPanel, StreamEvent } from "@/components/ThinkingPanel";
import { ObservabilityPanel } from "@/components/ObservabilityPanel";
import { gqlRequest } from "@/lib/graphql/client";
import { CREATE_SESSION, GET_SESSION } from "@/lib/graphql/queries";
import { SessionState, Message, SongRecommendation, UserProfile } from "@/lib/graphql/types";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

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
  const [streamEvents, setStreamEvents] = useState<StreamEvent[]>([]);
  const [feedbackCount, setFeedbackCount] = useState(0);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeTab, setActiveTab] = useState<"chat" | "traces">("chat");
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
  }, [messages, streamEvents, recs]);

  /** Fetch full session to sync only the profile sidebar */
  async function refreshProfile(sid: string) {
    try {
      const data = await gqlRequest<{ session: SessionState }>(GET_SESSION, { sessionId: sid });
      if (data.session) setProfile(data.session.userProfile);
    } catch {
      // non-blocking
    }
  }

  /**
   * Core SSE streaming helper. Sends `message` to the agent and wires up
   * all state updates. Returns true on success.
   */
  async function streamToAgent(sid: string, message: string): Promise<boolean> {
    const response = await fetch(`${BACKEND_URL}/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sid, message }),
    });

    if (!response.ok || !response.body) {
      throw new Error(`Stream HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        try {
          const event = JSON.parse(line.slice(6));

          if (event.type === "node") {
            setStreamEvents((prev) => [...prev, event as StreamEvent]);

          } else if (event.type === "done") {
            if (event.assistantMessage) {
              setMessages((prev) => [
                ...prev,
                { role: "assistant", content: event.assistantMessage },
              ]);
            }
            if (event.recommendations?.length) setRecs(event.recommendations);
            if (event.biasIssues)  setBiasIssues(event.biasIssues);
            if (event.toolsCalled) setToolsCalled(event.toolsCalled);
            if (event.error) {
              setMessages((prev) => [
                ...prev,
                { role: "assistant", content: `Something went wrong: ${event.error}` },
              ]);
            }

          } else if (event.type === "error") {
            setMessages((prev) => [
              ...prev,
              { role: "assistant", content: `Something went wrong: ${event.error}` },
            ]);
          }
        } catch {
          // malformed SSE chunk — skip
        }
      }
    }
    return true;
  }

  /** Send a new chat message */
  async function sendMessage() {
    const msg = input.trim();
    if (!msg || loading || !sessionId) return;
    setInput("");
    setLoading(true);
    setStreamEvents([]);
    setRecs([]);
    setBiasIssues([]);
    setMessages((prev) => [...prev, { role: "user", content: msg }]);

    try {
      await streamToAgent(sessionId, msg);
    } catch (err) {
      setBackendError(String(err));
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Could not reach the backend. Is uvicorn running?" },
      ]);
    } finally {
      setLoading(false);
      await refreshProfile(sessionId);
    }
  }

  /** Handle thumbs-up / thumbs-down / more-like-this buttons */
  async function handleFeedback(songId: number, rating: string) {
    if (!sessionId || loading) return;

    const ratingMessage: Record<string, string> = {
      liked:          `I liked song #${songId}`,
      disliked:       `I didn't like song #${songId}, please don't recommend it again`,
      more_like_this: `More songs like #${songId} please`,
      less_like_this: `Fewer songs like #${songId}`,
    };
    const message = ratingMessage[rating];
    if (!message) return;

    setFeedbackCount((n) => n + 1);
    setMessages((prev) => [...prev, { role: "user", content: message }]);
    setLoading(true);
    setStreamEvents([]);

    try {
      await streamToAgent(sessionId, message);
    } catch {
      // feedback is best-effort — don't block the user
    } finally {
      setLoading(false);
      await refreshProfile(sessionId);
    }
  }

  async function startNewSession() {
    try {
      const data = await gqlRequest<{ createSession: string }>(CREATE_SESSION);
      setSessionId(data.createSession);
      setMessages([]);
      setProfile(EMPTY_PROFILE);
      setRecs([]);
      setBiasIssues([]);
      setStreamEvents([]);
      setToolsCalled([]);
      setFeedbackCount(0);
    } catch {
      // non-blocking
    }
  }

  return (
    <div className="flex h-screen bg-background text-foreground">
      {/* Main chat area */}
      <div className="flex flex-1 flex-col min-w-0">
        {/* Header + Tab bar */}
        <header className="border-b border-border px-4 pt-3 pb-0">
          <div className="flex items-center justify-between mb-2">
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
          </div>
          {/* Tabs */}
          <div className="flex gap-1">
            {(["chat", "traces"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-1.5 text-sm rounded-t border-x border-t transition-colors ${
                  activeTab === tab
                    ? "border-border bg-background font-medium text-foreground"
                    : "border-transparent text-muted-foreground hover:text-foreground"
                }`}
              >
                {tab === "chat" ? "💬 Chat" : "📊 Traces"}
              </button>
            ))}
          </div>
        </header>

        {/* Traces tab */}
        {activeTab === "traces" && (
          <div className="flex-1 overflow-hidden">
            <ObservabilityPanel sessionId={sessionId} />
          </div>
        )}

        {activeTab === "chat" && backendError && (
          <div className="border-b border-destructive/40 bg-destructive/10 px-4 py-2 text-xs text-destructive">
            ⚠️ {backendError}
          </div>
        )}

        {/* Messages — only rendered in chat tab */}
        {activeTab === "chat" && <ScrollArea className="flex-1 px-4 py-3">
          {messages.length === 0 && streamEvents.length === 0 && !loading && (
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

          {/* Live streaming thought panel — shown while loading or just after */}
          {(loading || streamEvents.length > 0) && (
            <div className="mt-3">
              <ThinkingPanel events={streamEvents} isLoading={loading} />
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
        </ScrollArea>}

        {/* Input bar — only in chat tab */}
        {activeTab === "chat" && <div className="border-t border-border p-3">
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
              {loading ? "…" : "Send"}
            </button>
          </form>
        </div>}
      </div>

      {/* Sidebar */}
      {sidebarOpen && (
        <div className="w-64 shrink-0 border-l border-border">
          <ProfileSidebar
            profile={profile}
            feedbackCount={feedbackCount}
            toolsCalled={toolsCalled}
            messageCount={messages.length}
            onNewSession={startNewSession}
          />
        </div>
      )}
    </div>
  );
}
