import type { SetupStatus } from "../types";

export async function getSetupStatus(): Promise<SetupStatus> {
  const res = await fetch("/api/setup/status");
  if (!res.ok) throw new Error("Failed to fetch setup status");
  return res.json();
}

export function streamDownload(
  onLine: (line: string) => void,
  onDone: () => void,
  onError: (err: string) => void
): () => void {
  let cancelled = false;

  fetch("/api/setup/download", { method: "POST" })
    .then(async (res) => {
      if (!res.body) throw new Error("No response body");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (!cancelled) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const msg = line.slice(6).trim();
            if (msg === "DONE") {
              onDone();
              return;
            }
            onLine(msg);
          }
        }
      }
    })
    .catch((err) => onError(String(err)));

  return () => {
    cancelled = true;
  };
}

export function streamBuildIndex(
  device: string,
  onLine: (line: string) => void,
  onProgress: (done: number, total: number, pct: number, etaMin: number) => void,
  onDone: () => void,
  onError: (err: string) => void
): () => void {
  let cancelled = false;

  fetch("/api/setup/build-index", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ device }),
  })
    .then(async (res) => {
      if (!res.body) throw new Error("No response body");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (!cancelled) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const msg = line.slice(6).trim();
            if (msg === "DONE") {
              onDone();
              return;
            }
            // Try to parse as progress JSON
            try {
              const p = JSON.parse(msg);
              if (p.done !== undefined) {
                onProgress(p.done, p.total, p.pct, p.eta_min);
                continue;
              }
            } catch {
              // Not JSON — treat as log line
            }
            onLine(msg);
          }
        }
      }
    })
    .catch((err) => onError(String(err)));

  return () => {
    cancelled = true;
  };
}
