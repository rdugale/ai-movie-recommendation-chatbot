import { useEffect, useState } from "react";
import { createSession } from "../api/chatApi";

export function useSession(): string | null {
  const [threadId, setThreadId] = useState<string | null>(null);

  useEffect(() => {
    const stored = sessionStorage.getItem("thread_id");
    if (stored) {
      setThreadId(stored);
    } else {
      createSession()
        .then((data) => {
          sessionStorage.setItem("thread_id", data.thread_id);
          setThreadId(data.thread_id);
        })
        .catch(console.error);
    }
  }, []);

  return threadId;
}
