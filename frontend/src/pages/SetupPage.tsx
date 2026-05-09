import { CheckCircle, Circle, Loader, AlertCircle, Cpu, Zap } from "lucide-react";
import { ProgressLog } from "../components/ProgressLog";
import type { StepState } from "../types";
import type { useSetup } from "../hooks/useSetup";

type SetupHook = ReturnType<typeof useSetup>;

interface Props {
  setup: SetupHook;
  onReady: () => void;
}

function StepIcon({ state }: { state: StepState }) {
  if (state === "done")    return <CheckCircle size={20} className="text-green-400 shrink-0" />;
  if (state === "running") return <Loader size={20} className="text-cinema-accent animate-spin shrink-0" />;
  if (state === "error")   return <AlertCircle size={20} className="text-red-400 shrink-0" />;
  return <Circle size={20} className="text-cinema-dim shrink-0" />;
}

export function SetupPage({ setup, onReady }: Props) {
  const {
    status,
    device, setDevice,
    downloadState, buildState,
    downloadLogs, buildLogs,
    buildProgress,
    runDownload, runBuildIndex,
  } = setup;

  const downloadDone = downloadState === "done";
  const buildDone    = buildState === "done";
  const canDownload  = downloadState === "idle" || downloadState === "error";
  const canBuild     = downloadDone && (buildState === "idle" || buildState === "error");

  return (
    <div className="flex-1 flex items-center justify-center bg-cinema-dark px-4">
      <div className="w-full max-w-lg">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="text-5xl mb-3">🎬</div>
          <h1 className="text-3xl font-display text-white mb-2">Movie Recommender</h1>
          <p className="text-cinema-dim text-sm">Set up the database before chatting</p>
        </div>

        <div className="bg-cinema-card border border-cinema-border rounded-2xl p-6 space-y-6">

          {/* Step 1: Download */}
          <div>
            <div className="flex items-center gap-3 mb-3">
              <StepIcon state={downloadState} />
              <div className="flex-1">
                <p className="text-white font-medium text-sm">Step 1: Download Dataset</p>
                <p className="text-cinema-dim text-xs">~238k movies from HuggingFace (jquigl/imdb-genres)</p>
              </div>
              {canDownload && (
                <button
                  onClick={runDownload}
                  className="px-3 py-1.5 text-xs rounded-lg bg-cinema-accent hover:bg-red-500 text-white transition-colors"
                >
                  Download
                </button>
              )}
              {downloadState === "done" && !status?.csv_exists && (
                <span className="text-xs text-green-400">Downloaded</span>
              )}
              {downloadState === "done" && status?.csv_exists && (
                <span className="text-xs text-green-400">Ready</span>
              )}
            </div>
            {downloadLogs.length > 0 && <ProgressLog logs={downloadLogs} />}
          </div>

          <div className="border-t border-cinema-border" />

          {/* Step 2: Build Index */}
          <div>
            <div className="flex items-center gap-3 mb-3">
              <StepIcon state={buildState} />
              <div className="flex-1">
                <p className={`font-medium text-sm ${downloadDone ? "text-white" : "text-cinema-dim"}`}>
                  Step 2: Build Vector Index
                </p>
                <p className="text-cinema-dim text-xs">Embed & index all movies into ChromaDB</p>
              </div>
            </div>

            {/* Device toggle */}
            {(buildState === "idle" || buildState === "error") && downloadDone && (
              <div className="flex items-center gap-2 mb-3 ml-8">
                <span className="text-xs text-cinema-dim">Embedding device:</span>
                <button
                  onClick={() => setDevice("cuda")}
                  className={`flex items-center gap-1 px-3 py-1 rounded-lg text-xs border transition-colors ${
                    device === "cuda"
                      ? "bg-cinema-accent border-cinema-accent text-white"
                      : "border-cinema-border text-cinema-dim hover:border-gray-500"
                  }`}
                >
                  <Zap size={12} /> GPU (cuda) ~11 min
                </button>
                <button
                  onClick={() => setDevice("cpu")}
                  className={`flex items-center gap-1 px-3 py-1 rounded-lg text-xs border transition-colors ${
                    device === "cpu"
                      ? "bg-cinema-accent border-cinema-accent text-white"
                      : "border-cinema-border text-cinema-dim hover:border-gray-500"
                  }`}
                >
                  <Cpu size={12} /> CPU ~31 min
                </button>
              </div>
            )}

            {canBuild && (
              <div className="ml-8">
                <button
                  onClick={runBuildIndex}
                  className="px-3 py-1.5 text-xs rounded-lg bg-cinema-accent hover:bg-red-500 text-white transition-colors"
                >
                  Build Index
                </button>
              </div>
            )}

            {buildLogs.length > 0 && (
              <div className="ml-8">
                <ProgressLog
                  logs={buildLogs}
                  progress={buildProgress}
                  showProgress={buildState === "running"}
                />
              </div>
            )}
          </div>

          {/* Go to chat */}
          {buildDone && (
            <>
              <div className="border-t border-cinema-border" />
              <div className="text-center">
                <p className="text-green-400 text-sm mb-3">
                  ✓ Setup complete! Everything is ready.
                </p>
                <button
                  onClick={onReady}
                  className="px-6 py-2.5 rounded-xl bg-cinema-accent hover:bg-red-500 text-white font-medium transition-colors"
                >
                  Go to Chat →
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
