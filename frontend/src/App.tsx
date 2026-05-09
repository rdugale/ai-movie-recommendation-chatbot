import { useState } from "react";
import { useSetup } from "./hooks/useSetup";
import { SetupPage } from "./pages/SetupPage";
import { ChatPage } from "./pages/ChatPage";

function App() {
  const setup = useSetup();
  const [forcedToChat, setForcedToChat] = useState(false);

  // Show setup page if index doesn't exist yet (and user hasn't manually proceeded)
  const showSetup = !forcedToChat && setup.status !== null && !setup.isReady;
  // Show loading state while status is being fetched
  const loading = setup.status === null;

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center bg-cinema-dark">
        <div className="text-cinema-dim text-sm animate-pulse">Connecting...</div>
      </div>
    );
  }

  if (showSetup) {
    return <SetupPage setup={setup} onReady={() => setForcedToChat(true)} />;
  }

  return <ChatPage />;
}

export default App;
