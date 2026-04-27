// Strawberry auto-converts snake_case → camelCase in GraphQL

export const CREATE_SESSION = `
  mutation {
    createSession
  }
`;

// AgentResponseType fields returned by sendMessage / sendFeedback
const AGENT_RESPONSE_FIELDS = `
  sessionId
  recommendations {
    id title artist genre mood
    energy valence danceability acousticness
    score confidence explanation v1Score
  }
  assistantMessage
  conflictDetected
  conflictDescription
  biasIssues
  toolsCalled
  error
`;

export const SEND_MESSAGE = `
  mutation SendMessage($sessionId: String!, $message: String!) {
    sendMessage(sessionId: $sessionId, message: $message) {
      ${AGENT_RESPONSE_FIELDS}
    }
  }
`;

export const SEND_FEEDBACK = `
  mutation SendFeedback($sessionId: String!, $songId: Int!, $rating: String!) {
    sendFeedback(sessionId: $sessionId, songId: $songId, rating: $rating) {
      ${AGENT_RESPONSE_FIELDS}
    }
  }
`;

// SessionType fields returned by the session query
export const GET_SESSION = `
  query GetSession($sessionId: String!) {
    session(sessionId: $sessionId) {
      sessionId
      messages { role content }
      userProfile {
        genre mood energy valence danceability acousticness activity
        likedSongIds excludedSongIds
      }
      finalRecommendations {
        id title artist genre mood
        energy valence danceability acousticness
        score confidence explanation v1Score
      }
      conflictDetected
      conflictDescription
    }
  }
`;
