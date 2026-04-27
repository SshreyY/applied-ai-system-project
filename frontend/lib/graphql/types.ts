// Matches the Strawberry schema (snake_case → camelCase in GraphQL)

export interface UserProfile {
  genre: string | null;
  mood: string | null;
  energy: number | null;
  valence: number | null;
  danceability: number | null;
  acousticness: number | null;
  activity: string | null;
  likedSongIds: number[];
  excludedSongIds: number[];
}

export interface SongRecommendation {
  id: number;
  title: string;
  artist: string;
  genre: string;
  mood: string;
  energy: number;
  valence: number;
  danceability: number;
  acousticness: number;
  score: number;
  confidence: number;
  explanation: string;
  v1Score: number | null;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
}

/** Returned by sendMessage / sendFeedback mutations */
export interface AgentResponse {
  sessionId: string;
  recommendations: SongRecommendation[];
  assistantMessage: string | null;
  conflictDetected: boolean;
  conflictDescription: string | null;
  biasIssues: string[];
  toolsCalled: string[];
  error: string | null;
}

export interface FeedbackEntry {
  songId: number;
  rating: string;
}

/** Returned by the session query */
export interface SessionState {
  sessionId: string;
  messages: Message[];
  userProfile: UserProfile;
  finalRecommendations: SongRecommendation[];
  conflictDetected: boolean;
  conflictDescription: string | null;
}
