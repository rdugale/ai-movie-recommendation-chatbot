import sys
import os
from pathlib import Path
from functools import partial

# graph_manager.py lives at backend/core/graph_manager.py → .parent.parent.parent = movie-recommender/
PARENT_DIR = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, PARENT_DIR)

INDEX_DIR = os.path.join(PARENT_DIR, "chroma_imdb")

from langchain_core.messages import HumanMessage


def index_exists() -> bool:
    return os.path.isdir(INDEX_DIR) and bool(os.listdir(INDEX_DIR))


class GraphManager:
    _instance: "GraphManager | None" = None

    @classmethod
    def get_instance(cls) -> "GraphManager | None":
        """Returns None if the index doesn't exist yet."""
        if cls._instance is None and index_exists():
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Force re-initialization on next get_instance() call (used after index rebuild)."""
        # Remove cached chatbot module so it re-runs init_stats_if_needed with fresh index
        if "chatbot" in sys.modules:
            del sys.modules["chatbot"]
        cls._instance = None

    def __init__(self):
        # Importing chatbot triggers init_stats_if_needed() at module load time.
        # chdir so chatbot's relative "./chroma_imdb" resolves to the correct parent dir.
        orig = os.getcwd()
        os.chdir(PARENT_DIR)
        try:
            import chatbot
            # Patch relative path constants to absolute so SQLite resolves correctly
            # at call time (not just at import time when cwd is temporarily correct).
            chatbot.STATS_DB = os.path.join(PARENT_DIR, "movie_stats.db")
            from chatbot import build_graph
            self.graph = build_graph()
        finally:
            os.chdir(orig)

    def invoke(self, thread_id: str, user_message: str, is_first: bool) -> dict:
        config = {"configurable": {"thread_id": thread_id}}
        if is_first:
            input_state = {
                "intent": "",
                "context": "",
                "liked_genres": [],
                "seen_titles": [],
                "last_query": "",
                "min_year": None,
                "max_year": None,
                "min_rating": None,
                "max_rating": None,
                "messages": [HumanMessage(content=user_message)],
            }
        else:
            input_state = {"messages": [HumanMessage(content=user_message)]}

        self.graph.invoke(input_state, config=config)
        return self.graph.get_state(config).values

    def get_state(self, thread_id: str) -> dict:
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = self.graph.get_state(config)
            return state.values if state else {}
        except Exception:
            return {}
