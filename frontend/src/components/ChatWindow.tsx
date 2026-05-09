import { useEffect, useRef } from "react";
import type { ChatMessage } from "../types";
import { MessageBubble } from "./MessageBubble";

interface Props {
  messages: ChatMessage[];
  isLoading: boolean;
}

export function ChatWindow({ messages, isLoading }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4">
      {messages.length === 0 && (
        <div className="flex flex-col items-center justify-center h-full text-center text-cinema-dim">
          <div className="text-5xl mb-4">🎬</div>
          <p className="text-lg font-display text-white mb-2">What would you like to watch?</p>
          <p className="text-sm max-w-sm">
            Try: <span className="text-cinema-gold">"Recommend sci-fi movies"</span>,{" "}
            <span className="text-cinema-gold">"Show action films after 2010"</span>, or{" "}
            <span className="text-cinema-gold">"How many horror movies are there?"</span>
          </p>
        </div>
      )}

      {messages.map((msg, i) => (
        <MessageBubble key={i} message={msg} />
      ))}

      {isLoading && (
        <div className="flex justify-start mb-3">
          <div className="w-8 h-8 rounded-full bg-cinema-accent flex items-center justify-center text-white text-sm mr-2 mt-1 shrink-0">
            🎬
          </div>
          <div className="bg-cinema-card border border-cinema-border px-4 py-3 rounded-2xl rounded-tl-sm">
            <div className="flex gap-1 items-center">
              <span className="w-2 h-2 rounded-full bg-cinema-accent animate-bounce" style={{ animationDelay: "0ms" }} />
              <span className="w-2 h-2 rounded-full bg-cinema-accent animate-bounce" style={{ animationDelay: "150ms" }} />
              <span className="w-2 h-2 rounded-full bg-cinema-accent animate-bounce" style={{ animationDelay: "300ms" }} />
            </div>
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
