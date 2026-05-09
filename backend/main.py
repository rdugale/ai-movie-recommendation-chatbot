import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure parent dir (containing chatbot.py) is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api import chat, session, stats, setup
from core.graph_manager import GraphManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Only initialize if the index already exists; otherwise setup page handles it
    GraphManager.get_instance()
    yield


app = FastAPI(title="Movie Recommender API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router,    prefix="/api")
app.include_router(session.router, prefix="/api")
app.include_router(stats.router,   prefix="/api")
app.include_router(setup.router,   prefix="/api")
