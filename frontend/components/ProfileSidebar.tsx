"use client";

import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { UserProfile, FeedbackEntry } from "@/lib/graphql/types";

interface Props {
  profile: UserProfile;
  feedbackEntries: FeedbackEntry[];
  toolsCalled: string[];
  messageCount: number;
  onNewSession: () => void;
}

function StatBar({ label, value }: { label: string; value: number }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span>{Math.round(value * 100)}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-muted">
        <div
          className="h-1.5 rounded-full bg-primary transition-all"
          style={{ width: `${Math.round(value * 100)}%` }}
        />
      </div>
    </div>
  );
}

export function ProfileSidebar({
  profile,
  feedbackEntries,
  toolsCalled,
  messageCount,
  onNewSession,
}: Props) {
  const hasProfile = Object.values(profile).some(
    (v) => v !== null && v !== undefined && !(Array.isArray(v) && v.length === 0)
  );

  return (
    <aside className="flex h-full flex-col gap-4 overflow-y-auto p-4">
      <div className="flex items-center justify-between">
        <h2 className="font-semibold text-sm">Your Vibe Profile</h2>
        <button
          onClick={onNewSession}
          className="rounded-md border border-border px-2 py-1 text-xs transition-colors hover:bg-muted"
        >
          New Session
        </button>
      </div>

      {!hasProfile ? (
        <p className="text-xs text-muted-foreground">
          Start chatting to build your profile.
        </p>
      ) : (
        <div className="space-y-3">
          {profile.genre && (
            <div>
              <p className="mb-1 text-xs text-muted-foreground">Genre</p>
              <Badge>{profile.genre}</Badge>
            </div>
          )}
          {profile.mood && (
            <div>
              <p className="mb-1 text-xs text-muted-foreground">Mood</p>
              <Badge variant="secondary">{profile.mood}</Badge>
            </div>
          )}
          {profile.activity && (
            <div>
              <p className="mb-1 text-xs text-muted-foreground">Activity</p>
              <Badge variant="outline">{profile.activity}</Badge>
            </div>
          )}
          <div className="space-y-2 pt-1">
            {profile.energy !== null && profile.energy !== undefined && (
              <StatBar label="Energy" value={profile.energy} />
            )}
            {profile.valence !== null && profile.valence !== undefined && (
              <StatBar label="Valence" value={profile.valence} />
            )}
            {profile.danceability !== null && profile.danceability !== undefined && (
              <StatBar label="Danceability" value={profile.danceability} />
            )}
            {profile.acousticness !== null && profile.acousticness !== undefined && (
              <StatBar label="Acousticness" value={profile.acousticness} />
            )}
          </div>
        </div>
      )}

      <Separator />

      <div className="space-y-2">
        <h3 className="text-xs font-medium">Session Stats</h3>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="rounded-md bg-muted/50 p-2 text-center">
            <div className="text-lg font-bold">{messageCount}</div>
            <div className="text-muted-foreground">Messages</div>
          </div>
          <div className="rounded-md bg-muted/50 p-2 text-center">
            <div className="text-lg font-bold">{feedbackEntries.length}</div>
            <div className="text-muted-foreground">Ratings</div>
          </div>
        </div>
      </div>

      {toolsCalled.length > 0 && (
        <>
          <Separator />
          <div className="space-y-1">
            <h3 className="text-xs font-medium text-muted-foreground">Last tools used</h3>
            <div className="flex flex-wrap gap-1">
              {[...new Set(toolsCalled)].map((t) => (
                <Badge key={t} variant="outline" className="text-xs font-mono">
                  {t}
                </Badge>
              ))}
            </div>
          </div>
        </>
      )}

      {(profile.likedSongIds?.length > 0 || profile.excludedSongIds?.length > 0) && (
        <>
          <Separator />
          <div className="space-y-1 text-xs text-muted-foreground">
            {profile.likedSongIds?.length > 0 && (
              <p>👍 Liked IDs: {profile.likedSongIds.join(", ")}</p>
            )}
            {profile.excludedSongIds?.length > 0 && (
              <p>🚫 Excluded IDs: {profile.excludedSongIds.join(", ")}</p>
            )}
          </div>
        </>
      )}
    </aside>
  );
}
