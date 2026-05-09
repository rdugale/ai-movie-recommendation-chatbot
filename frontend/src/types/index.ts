export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  thread_id: string;
  message: string;
  is_first_message: boolean;
}

export interface ChatResponse {
  reply: string;
  liked_genres: string[];
  seen_count: number;
  intent: string;
}

export interface NewSessionResponse {
  thread_id: string;
}

export interface SessionStateResponse {
  thread_id: string;
  liked_genres: string[];
  seen_count: number;
  message_count: number;
}

export interface StatsResponse {
  total_movies: number;
  top_genres: { genre: string; count: number }[];
}

export interface SetupStatus {
  csv_exists: boolean;
  index_exists: boolean;
  stats_db_exists: boolean;
}

export interface IndexProgress {
  done: number;
  total: number;
  pct: number;
  eta_min: number;
}

export type StepState = "idle" | "running" | "done" | "error";
