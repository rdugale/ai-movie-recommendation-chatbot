import type {
  ChatRequest,
  ChatResponse,
  NewSessionResponse,
  SessionStateResponse,
  StatsResponse,
} from "../types";

export async function createSession(): Promise<NewSessionResponse> {
  const res = await fetch("/api/session/new", { method: "POST" });
  if (!res.ok) throw new Error("Failed to create session");
  return res.json();
}

export async function sendMessage(req: ChatRequest): Promise<ChatResponse> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail ?? "Chat request failed");
  }
  return res.json();
}

export async function getSessionState(
  threadId: string
): Promise<SessionStateResponse> {
  const res = await fetch(`/api/session/${threadId}/state`);
  if (!res.ok) throw new Error("Failed to fetch session state");
  return res.json();
}

export async function getStats(): Promise<StatsResponse> {
  const res = await fetch("/api/stats");
  if (!res.ok) throw new Error("Failed to fetch stats");
  return res.json();
}
