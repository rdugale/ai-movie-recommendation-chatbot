import { useState, useRef, type KeyboardEvent } from "react";
import { Send } from "lucide-react";

interface Props {
  onSend: (text: string) => void;
  isLoading: boolean;
  likedGenres: string[];
  seenCount: number;
}

export function InputBar({ onSend, isLoading, likedGenres, seenCount }: Props) {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    const trimmed = text.trim();
    if (!trimmed || isLoading) return;
    onSend(trimmed);
    setText("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = () => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
    }
  };

  return (
    <div className="border-t border-cinema-border bg-cinema-card px-4 py-3">
      <div className="flex gap-2 items-end">
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          placeholder="Ask for movie recommendations... (Enter to send, Shift+Enter for newline)"
          disabled={isLoading}
          rows={1}
          className="flex-1 resize-none bg-[#0d0d0d] border border-cinema-border rounded-xl px-4 py-3 text-sm text-gray-200 placeholder-cinema-dim focus:outline-none focus:border-cinema-accent transition-colors disabled:opacity-50"
          style={{ minHeight: "44px", maxHeight: "160px" }}
        />
        <button
          onClick={handleSend}
          disabled={isLoading || !text.trim()}
          className="w-11 h-11 rounded-xl bg-cinema-accent hover:bg-red-500 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center transition-colors shrink-0"
        >
          <Send size={18} className="text-white" />
        </button>
      </div>

      {(likedGenres.length > 0 || seenCount > 0) && (
        <div className="flex flex-wrap items-center gap-2 mt-2">
          {likedGenres.length > 0 && (
            <>
              <span className="text-xs text-cinema-dim">Likes:</span>
              {likedGenres.map((g) => (
                <span
                  key={g}
                  className="px-2 py-0.5 rounded-full text-xs bg-cinema-gold/20 text-cinema-gold border border-cinema-gold/30"
                >
                  {g}
                </span>
              ))}
            </>
          )}
          {seenCount > 0 && (
            <span className="text-xs text-cinema-dim ml-auto">
              • {seenCount} movies recommended
            </span>
          )}
        </div>
      )}
    </div>
  );
}
