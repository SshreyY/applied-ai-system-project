"use client";

export interface StreamEvent {
  type: "node";
  node: string;
  icon: string;
  label: string;
  detail: string | null;
}

interface Props {
  events: StreamEvent[];
  isLoading: boolean;
}

export function ThinkingPanel({ events, isLoading }: Props) {
  if (events.length === 0 && !isLoading) return null;

  return (
    <div className="my-2 rounded-lg border border-border/40 bg-muted/30 px-3 py-2 text-xs">
      <p className="mb-1.5 font-medium text-muted-foreground tracking-wide uppercase text-[10px]">
        Agent reasoning
      </p>
      <div className="space-y-1.5">
        {events.map((ev, i) => {
          const isLast = i === events.length - 1;
          const pulsing = isLoading && isLast;
          return (
            <div key={i} className="flex items-start gap-2">
              {/* icon */}
              <span className={`mt-px shrink-0 ${pulsing ? "animate-pulse" : ""}`}>
                {ev.icon}
              </span>
              {/* text */}
              <div className="min-w-0">
                <span className="font-medium text-foreground/80">{ev.label}</span>
                {ev.detail && (
                  <span className="ml-1 text-muted-foreground">— {ev.detail}</span>
                )}
              </div>
              {/* done checkmark */}
              {!pulsing && (
                <span className="ml-auto shrink-0 text-green-500">✓</span>
              )}
              {/* spinner for last in-progress step */}
              {pulsing && (
                <span className="ml-auto shrink-0 text-muted-foreground animate-spin">⟳</span>
              )}
            </div>
          );
        })}

        {/* trailing "working..." line while loading with no events yet */}
        {isLoading && events.length === 0 && (
          <div className="flex items-center gap-2 text-muted-foreground animate-pulse">
            <span>⏳</span>
            <span>Starting up…</span>
          </div>
        )}
      </div>
    </div>
  );
}
