import { useEffect, useRef } from "react";

interface Props {
  logs: string[];
  progress?: { done: number; total: number; pct: number; etaMin: number };
  showProgress?: boolean;
}

export function ProgressLog({ logs, progress, showProgress = false }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  return (
    <div className="mt-3 rounded-xl bg-black border border-cinema-border overflow-hidden">
      {showProgress && progress && progress.total > 0 && (
        <div className="px-3 pt-3">
          <div className="flex justify-between text-xs text-cinema-dim mb-1">
            <span>{progress.done.toLocaleString()} / {progress.total.toLocaleString()} movies</span>
            <span>ETA: {progress.etaMin} min</span>
          </div>
          <div className="w-full bg-cinema-border rounded-full h-2 mb-3">
            <div
              className="bg-cinema-accent h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress.pct}%` }}
            />
          </div>
        </div>
      )}
      <div className="h-40 overflow-y-auto px-3 py-2 font-mono text-xs text-green-400 space-y-0.5">
        {logs.map((line, i) => (
          <div key={i} className="leading-5">
            <span className="text-cinema-dim mr-2">›</span>
            {line}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
