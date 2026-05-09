import { useState, useCallback } from "react";
import { sendMessage } from "../api/chatApi";
import type { ChatMessage } from "../types";

export function useChat(threadId: string | null) {
  const [messages, setMessages]     = useState<ChatMessage[]>([]);
  const [likedGenres, setLikedGenres] = useState<string[]>([]);
  const [seenCount, setSeenCount]   = useState(0);
  const [isLoading, setIsLoading]   = useState(false);
  const [error, setError]           = useState<string | null>(null);

  const send = useCallback(
    async (text: string) => {
      if (!threadId || !text.trim()) return;
      const isFirst = messages.length === 0;
      setError(null);
      setMessages((prev) => [...prev, { role: "user", content: text }]);
      setIsLoading(true);
      try {
        const res = await sendMessage({
          thread_id: threadId,
          message: text,
          is_first_message: isFirst,
        });
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: res.reply },
        ]);
        setLikedGenres(res.liked_genres);
        setSeenCount(res.seen_count);
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Unknown error";
        setError(msg);
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: `Error: ${msg}` },
        ]);
      } finally {
        setIsLoading(false);
      }
    },
    [threadId, messages.length]
  );

  return { messages, likedGenres, seenCount, isLoading, error, send };
}
