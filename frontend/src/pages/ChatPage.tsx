import { Film } from "lucide-react";
import { ChatWindow } from "../components/ChatWindow";
import { InputBar } from "../components/InputBar";
import { useSession } from "../hooks/useSession";
import { useChat } from "../hooks/useChat";

export function ChatPage() {
  const threadId = useSession();
  const { messages, likedGenres, seenCount, isLoading, send } = useChat(threadId);

  return (
    <div className="flex flex-col h-screen bg-cinema-dark">
      {/* Header */}
      <header className="shrink-0 flex items-center gap-3 px-5 py-4 border-b border-cinema-border bg-cinema-card">
        <Film size={24} className="text-cinema-accent" />
        <span className="text-white font-display text-xl">Movie Recommender</span>
        <span className="text-cinema-dim text-xs ml-auto">
          Powered by LangGraph + RAG
        </span>
      </header>

      {/* Messages */}
      <ChatWindow messages={messages} isLoading={isLoading} />

      {/* Input */}
      <InputBar
        onSend={send}
        isLoading={isLoading}
        likedGenres={likedGenres}
        seenCount={seenCount}
      />
    </div>
  );
}
