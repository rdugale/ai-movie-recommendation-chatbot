"""
Development launcher — watches both backend/ and parent chatbot.py for changes.
Usage: python run.py
"""
import uvicorn
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PARENT_DIR  = BACKEND_DIR.parent

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(BACKEND_DIR), str(PARENT_DIR)],
        reload_excludes=[
            "chroma_imdb/*",
            "chroma_imdb_old/*",
            "venv/*",
            "__pycache__/*",
            "*.db",
            "*.csv",
        ],
    )
