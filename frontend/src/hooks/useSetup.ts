import { useState, useCallback, useEffect } from "react";
import { getSetupStatus, streamDownload, streamBuildIndex } from "../api/setupApi";
import type { SetupStatus, StepState } from "../types";

export function useSetup() {
  const [status, setStatus]           = useState<SetupStatus | null>(null);
  const [downloadState, setDownloadState] = useState<StepState>("idle");
  const [buildState, setBuildState]   = useState<StepState>("idle");
  const [downloadLogs, setDownloadLogs] = useState<string[]>([]);
  const [buildLogs, setBuildLogs]     = useState<string[]>([]);
  const [buildProgress, setBuildProgress] = useState({ done: 0, total: 0, pct: 0, etaMin: 0 });
  const [device, setDevice]           = useState<"cuda" | "cpu">("cuda");

  const checkStatus = useCallback(async () => {
    try {
      const s = await getSetupStatus();
      setStatus(s);
      // Pre-mark steps that are already done
      if (s.csv_exists)   setDownloadState("done");
      if (s.index_exists) setBuildState("done");
    } catch {
      setStatus({ csv_exists: false, index_exists: false, stats_db_exists: false });
    }
  }, []);

  useEffect(() => {
    checkStatus();
  }, [checkStatus]);

  const runDownload = useCallback(() => {
    setDownloadState("running");
    setDownloadLogs([]);
    streamDownload(
      (line) => setDownloadLogs((prev) => [...prev, line]),
      () => {
        setDownloadState("done");
        setStatus((s) => s ? { ...s, csv_exists: true } : s);
      },
      (err) => {
        setDownloadState("error");
        setDownloadLogs((prev) => [...prev, `Error: ${err}`]);
      }
    );
  }, []);

  const runBuildIndex = useCallback(() => {
    setBuildState("running");
    setBuildLogs([]);
    setBuildProgress({ done: 0, total: 0, pct: 0, etaMin: 0 });
    streamBuildIndex(
      device,
      (line) => setBuildLogs((prev) => [...prev, line]),
      (done, total, pct, etaMin) => setBuildProgress({ done, total, pct, etaMin }),
      () => {
        setBuildState("done");
        setStatus((s) => s ? { ...s, index_exists: true, stats_db_exists: true } : s);
      },
      (err) => {
        setBuildState("error");
        setBuildLogs((prev) => [...prev, `Error: ${err}`]);
      }
    );
  }, [device]);

  const isReady = status?.index_exists ?? false;

  return {
    status,
    isReady,
    device,
    setDevice,
    downloadState,
    buildState,
    downloadLogs,
    buildLogs,
    buildProgress,
    runDownload,
    runBuildIndex,
  };
}
