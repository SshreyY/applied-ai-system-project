"use client";

import { useEffect, useState, useCallback } from "react";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

// ── Types ──────────────────────────────────────────────────────────────────

interface TraceStep {
  id: string;
  name: string;
  type: string;
  model: string;
  startTime: string | null;
  endTime: string | null;
  latencyMs: number | null;
  inputTokens: number | null;
  outputTokens: number | null;
  input: unknown;
  output: unknown;
  level: string;
}

interface Trace {
  id: string;
  name: string;
  timestamp: string | null;
  latencyMs: number | null;
  inputTokens: number;
  outputTokens: number;
  totalCost: number | null;
  input: unknown;
  output: unknown;
  tags: string[];
  steps: TraceStep[];
}

// ── Helpers ────────────────────────────────────────────────────────────────

function fmtMs(ms: number | null | undefined): string {
  if (ms == null) return "—";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

function fmtTokens(n: number | null | undefined): string {
  if (n == null || n === 0) return "—";
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
}

function fmtCost(c: number | null | undefined): string {
  if (c == null || c === 0) return "—";
  return `$${c.toFixed(5)}`;
}

function fmtTime(iso: string | null): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch {
    return iso;
  }
}

function extractUserInput(input: unknown): string {
  if (!input) return "";
  if (typeof input === "string") return input.slice(0, 100);
  const obj = input as Record<string, unknown>;
  // LangChain traces often have messages array
  if (Array.isArray(obj.messages)) {
    const msgs = obj.messages as Array<Record<string, unknown>>;
    const last = msgs.findLast?.((m) => m.role === "user" || m.type === "human");
    if (last) return String(last.content ?? "").slice(0, 100);
  }
  if (obj.input) return String(obj.input).slice(0, 100);
  return JSON.stringify(input).slice(0, 100);
}

const STEP_ICONS: Record<string, string> = {
  GENERATION: "🤖",
  SPAN:       "⚙️",
  EVENT:      "📌",
};

const LEVEL_COLOR: Record<string, string> = {
  ERROR:   "text-red-400",
  WARNING: "text-yellow-400",
  DEFAULT: "text-muted-foreground",
};

// ── Sub-components ─────────────────────────────────────────────────────────

function StepRow({ step }: { step: TraceStep }) {
  const [open, setOpen] = useState(false);
  const icon = STEP_ICONS[step.type] ?? "⚙️";
  const levelColor = LEVEL_COLOR[step.level] ?? LEVEL_COLOR.DEFAULT;

  return (
    <div className="rounded border border-border/30 bg-muted/20 text-xs">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-2 py-1.5 hover:bg-muted/40 text-left"
      >
        <span>{icon}</span>
        <span className={`font-mono font-medium ${levelColor} truncate flex-1`}>{step.name || step.type}</span>
        {step.model && <Badge variant="outline" className="text-[10px] shrink-0">{step.model}</Badge>}
        <span className="shrink-0 text-muted-foreground">{fmtMs(step.latencyMs)}</span>
        {step.inputTokens != null && (
          <span className="shrink-0 text-muted-foreground">
            {fmtTokens(step.inputTokens)}↑ {fmtTokens(step.outputTokens)}↓
          </span>
        )}
        <span className="shrink-0 text-muted-foreground">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div className="border-t border-border/30 px-3 py-2 space-y-2">
          {step.input != null && (
            <div>
              <p className="text-[10px] uppercase font-medium text-muted-foreground mb-0.5">Input</p>
              <pre className="whitespace-pre-wrap break-words text-[10px] text-foreground/70 max-h-32 overflow-auto">
                {typeof step.input === "string" ? step.input : JSON.stringify(step.input, null, 2)}
              </pre>
            </div>
          )}
          {step.output != null && (
            <div>
              <p className="text-[10px] uppercase font-medium text-muted-foreground mb-0.5">Output</p>
              <pre className="whitespace-pre-wrap break-words text-[10px] text-foreground/70 max-h-32 overflow-auto">
                {typeof step.output === "string" ? step.output : JSON.stringify(step.output, null, 2)}
              </pre>
            </div>
          )}
          <p className="text-[10px] text-muted-foreground">{fmtTime(step.startTime)} → {fmtTime(step.endTime)}</p>
        </div>
      )}
    </div>
  );
}

function TraceCard({ trace }: { trace: Trace }) {
  const [open, setOpen] = useState(false);
  const [steps, setSteps] = useState<TraceStep[]>([]);
  const [stepsLoading, setStepsLoading] = useState(false);
  const [stepsError, setStepsError] = useState<string | null>(null);
  const userInput = extractUserInput(trace.input);
  const totalTokens = (trace.inputTokens ?? 0) + (trace.outputTokens ?? 0);

  async function handleExpand() {
    const next = !open;
    setOpen(next);
    if (next && steps.length === 0 && !stepsLoading) {
      setStepsLoading(true);
      try {
        const res = await fetch(`${BACKEND_URL}/api/observations?trace_id=${encodeURIComponent(trace.id)}`);
        const data = await res.json();
        setSteps(data.steps ?? []);
        if (data.error) setStepsError(data.error);
      } catch (e) {
        setStepsError(String(e));
      } finally {
        setStepsLoading(false);
      }
    }
  }

  return (
    <div className="rounded-lg border border-border/50 overflow-hidden">
      {/* Header row */}
      <button
        onClick={handleExpand}
        className="flex w-full items-start gap-3 px-3 py-2.5 hover:bg-muted/30 text-left"
      >
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-medium text-sm truncate">{trace.name || "Trace"}</span>
            {trace.tags.filter(t => t !== "vibefinder").map(tag => (
              <Badge key={tag} variant="outline" className="text-[10px]">{tag}</Badge>
            ))}
          </div>
          {userInput && (
            <p className="text-xs text-muted-foreground truncate mt-0.5">&ldquo;{userInput}&rdquo;</p>
          )}
        </div>
        <div className="shrink-0 text-right text-xs text-muted-foreground space-y-0.5">
          <p>{fmtMs(trace.latencyMs)}</p>
          {totalTokens > 0 && <p>{fmtTokens(totalTokens)} tok</p>}
          {trace.totalCost != null && trace.totalCost > 0 && <p>{fmtCost(trace.totalCost)}</p>}
        </div>
        <span className="text-muted-foreground text-xs shrink-0 self-center">{open ? "▲" : "▼"}</span>
      </button>

      {/* Expanded — observations loaded lazily */}
      {open && (
        <div className="border-t border-border/40 px-3 py-2 space-y-1.5 bg-background/50">
          <p className="text-[10px] text-muted-foreground">{fmtTime(trace.timestamp)}</p>
          {stepsLoading && (
            <p className="text-xs text-muted-foreground animate-pulse">Loading observations…</p>
          )}
          {stepsError && (
            <p className="text-xs text-destructive">{stepsError}</p>
          )}
          {!stepsLoading && steps.length === 0 && !stepsError && (
            <p className="text-xs text-muted-foreground italic">No observations captured for this trace.</p>
          )}
          {steps.map((step) => <StepRow key={step.id} step={step} />)}
        </div>
      )}
    </div>
  );
}

// ── Main panel ─────────────────────────────────────────────────────────────

interface Props {
  sessionId: string | null;
}

export function ObservabilityPanel({ sessionId }: Props) {
  const [traces, setTraces] = useState<Trace[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [configured, setConfigured] = useState(true);
  const [source, setSource] = useState<"session" | "recent" | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);
  const [cooldown, setCooldown] = useState(0); // seconds remaining before next manual refresh

  // Tick down the cooldown counter every second
  useEffect(() => {
    if (cooldown <= 0) return;
    const t = setTimeout(() => setCooldown((c) => Math.max(0, c - 1)), 1000);
    return () => clearTimeout(t);
  }, [cooldown]);

  const fetchTraces = useCallback(async (manual = false) => {
    if (!sessionId) return;
    setLoading(true);
    setError(null);
    try {
      const url = `${BACKEND_URL}/api/traces?session_id=${encodeURIComponent(sessionId)}${manual ? "&force=true" : ""}`;
      const res = await fetch(url);
      const data = await res.json();
      setConfigured(data.configured ?? true);
      setTraces(data.traces ?? []);
      setSource(data.source ?? null);
      if (data.error) setError(data.error);
      setLastRefresh(new Date());
      if (manual) setCooldown(30); // 30-second cooldown after a manual refresh
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    fetchTraces();
  }, [fetchTraces]);

  const totalTokens = traces.reduce((s, t) => s + (t.inputTokens ?? 0) + (t.outputTokens ?? 0), 0);
  const avgLatency = traces.length
    ? traces.reduce((s, t) => s + (t.latencyMs ?? 0), 0) / traces.length
    : 0;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="border-b border-border px-4 py-3 flex items-center justify-between">
        <div>
          <h2 className="font-semibold text-sm">AI Observability</h2>
          <p className="text-xs text-muted-foreground">Powered by Langfuse</p>
        </div>
        <button
          onClick={() => fetchTraces(true)}
          disabled={loading || cooldown > 0}
          className="rounded border border-border px-2 py-1 text-xs hover:bg-muted disabled:opacity-50"
          title={cooldown > 0 ? `Rate limit — wait ${cooldown}s` : "Fetch latest traces from Langfuse"}
        >
          {loading ? "Refreshing…" : cooldown > 0 ? `↻ ${cooldown}s` : "↻ Refresh"}
        </button>
      </div>

      <ScrollArea className="flex-1 px-4 py-3 space-y-3">
        {/* Not configured */}
        {!configured && (
          <div className="rounded-lg border border-yellow-500/30 bg-yellow-500/10 p-4 text-sm text-yellow-300">
            <p className="font-medium">Langfuse not configured</p>
            <p className="text-xs mt-1">
              Add <code className="font-mono">LANGFUSE_PUBLIC_KEY</code>,{" "}
              <code className="font-mono">LANGFUSE_SECRET_KEY</code>, and{" "}
              <code className="font-mono">LANGFUSE_BASE_URL</code> to{" "}
              <code className="font-mono">backend/.env</code> to enable tracing.
            </p>
          </div>
        )}

        {/* Error */}
        {error && configured && (
          <div className="rounded border border-destructive/40 bg-destructive/10 p-3 text-xs text-destructive">
            {error}
          </div>
        )}

        {/* Source banner — shown when falling back to recent traces */}
        {source === "recent" && traces.length > 0 && (
          <div className="rounded border border-blue-500/30 bg-blue-500/10 px-3 py-2 text-xs text-blue-300">
            ℹ️ Showing the 30 most recent project traces — no session-specific traces found yet.
            This happens when Langfuse hasn&apos;t flushed the session tag, or the session is new.
            Try sending a message first, then refresh.
          </div>
        )}

        {/* Session ID debug info */}
        {sessionId && (
          <p className="text-[10px] text-muted-foreground font-mono break-all">
            Session: {sessionId}
          </p>
        )}

        {/* Summary stats */}
        {traces.length > 0 && (
          <>
            <div className="grid grid-cols-3 gap-2 text-center text-xs">
              <div className="rounded-md bg-muted/50 p-2">
                <div className="text-lg font-bold">{traces.length}</div>
                <div className="text-muted-foreground">Traces</div>
              </div>
              <div className="rounded-md bg-muted/50 p-2">
                <div className="text-lg font-bold">{fmtTokens(totalTokens)}</div>
                <div className="text-muted-foreground">Tokens</div>
              </div>
              <div className="rounded-md bg-muted/50 p-2">
                <div className="text-lg font-bold">{fmtMs(avgLatency)}</div>
                <div className="text-muted-foreground">Avg latency</div>
              </div>
            </div>

            <Separator />

            <p className="text-[10px] text-muted-foreground">
              Last refreshed: {lastRefresh?.toLocaleTimeString() ?? "—"} · Click a trace to expand
            </p>

            <div className="space-y-2">
              {traces.map((t) => <TraceCard key={t.id} trace={t} />)}
            </div>
          </>
        )}

        {/* Empty state */}
        {!loading && traces.length === 0 && configured && !error && (
          <div className="flex flex-col items-center gap-3 py-12 text-center text-muted-foreground">
            <span className="text-4xl">📡</span>
            <p className="text-sm">No traces yet for this session.</p>
            <p className="text-xs opacity-60">
              Send a message and traces will appear here. Langfuse captures every LLM call — latency, tokens, inputs, and outputs.
            </p>
          </div>
        )}
      </ScrollArea>
    </div>
  );
}
