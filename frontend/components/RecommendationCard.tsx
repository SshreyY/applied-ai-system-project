"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { SongRecommendation } from "@/lib/graphql/types";

interface Props {
  rec: SongRecommendation;
  rank: number;
  onFeedback: (songId: number, rating: string) => void;
}

const RATING_BUTTONS = [
  { label: "👍", rating: "liked", title: "Liked" },
  { label: "👎", rating: "disliked", title: "Disliked" },
  { label: "➕", rating: "more_like_this", title: "More like this" },
  { label: "➖", rating: "less_like_this", title: "Less like this" },
];

function EnergyBar({ value, label }: { value: number; label: string }) {
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground">
      <span className="w-20 shrink-0">{label}</span>
      <div className="h-1.5 flex-1 rounded-full bg-muted">
        <div
          className="h-1.5 rounded-full bg-primary"
          style={{ width: `${Math.round(value * 100)}%` }}
        />
      </div>
      <span className="w-8 text-right">{Math.round(value * 100)}%</span>
    </div>
  );
}

export function RecommendationCard({ rec, rank, onFeedback }: Props) {
  const pct = Math.round(rec.confidence * 100);

  return (
    <Card className="border-border/50 bg-card/80 backdrop-blur">
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <CardTitle className="truncate text-base">
              #{rank} {rec.title}
            </CardTitle>
            <p className="truncate text-sm text-muted-foreground">{rec.artist}</p>
          </div>
          <Badge
            variant={pct >= 80 ? "default" : "secondary"}
            className="shrink-0"
          >
            {pct}% match
          </Badge>
        </div>
        <div className="flex flex-wrap gap-1 pt-1">
          <Badge variant="outline" className="text-xs">
            {rec.genre}
          </Badge>
          <Badge variant="outline" className="text-xs">
            {rec.mood}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="space-y-1.5">
          <EnergyBar value={rec.energy} label="Energy" />
          <EnergyBar value={rec.valence} label="Valence" />
          <EnergyBar value={rec.danceability} label="Dance" />
        </div>

        <p className="text-xs text-muted-foreground italic">{rec.explanation}</p>

        {rec.v1Score !== null && rec.v1Score !== undefined && (
          <p className="text-xs text-muted-foreground">
            V1 score: {rec.v1Score.toFixed(2)} / 7.5
          </p>
        )}

        <div className="flex gap-1">
          {RATING_BUTTONS.map(({ label, rating, title }) => (
            <button
              key={rating}
              title={title}
              onClick={() => onFeedback(rec.id, rating)}
              className="rounded px-2 py-1 text-sm transition-colors hover:bg-muted"
            >
              {label}
            </button>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
