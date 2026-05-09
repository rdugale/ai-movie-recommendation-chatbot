import sqlite3
from pathlib import Path
from fastapi import APIRouter, HTTPException
from models.schemas import StatsResponse

router = APIRouter()

# stats.py lives at backend/api/stats.py → .parent.parent.parent = movie-recommender/
STATS_DB = str(Path(__file__).resolve().parent.parent.parent / "movie_stats.db")


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    if not os.path.exists(STATS_DB):
        raise HTTPException(status_code=503, detail="Stats database not ready. Run setup first.")

    conn = sqlite3.connect(STATS_DB)
    try:
        total = conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
        rows = conn.execute("SELECT genre, cnt FROM genre_counts LIMIT 15").fetchall()
        top_genres = [{"genre": g, "count": c} for g, c in rows]
    finally:
        conn.close()

    return StatsResponse(total_movies=total, top_genres=top_genres)
