import type { ChatMessage } from "../types";

interface Props {
  message: ChatMessage;
}

function renderContent(text: string) {
  // Render **bold** inline and preserve line breaks
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={i} className="font-semibold text-white">{part.slice(2, -2)}</strong>;
    }
    return (
      <span key={i}>
        {part.split("\n").map((line, j, arr) => (
          <span key={j}>
            {line}
            {j < arr.length - 1 && <br />}
          </span>
        ))}
      </span>
    );
  });
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3`}>
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-cinema-accent flex items-center justify-center text-white text-sm mr-2 mt-1 shrink-0">
          🎬
        </div>
      )}
      <div
        className={`max-w-[75%] px-4 py-3 rounded-2xl text-sm leading-relaxed ${
          isUser
            ? "bg-cinema-accent text-white rounded-tr-sm"
            : "bg-cinema-card text-gray-200 rounded-tl-sm border border-cinema-border"
        }`}
      >
        {renderContent(message.content)}
      </div>
      {isUser && (
        <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center text-white text-sm ml-2 mt-1 shrink-0">
          👤
        </div>
      )}
    </div>
  );
}
