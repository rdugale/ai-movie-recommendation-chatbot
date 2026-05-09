import os
import sys
import json
import time
import asyncio
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from models.schemas import SetupStatusResponse, BuildIndexRequest

router = APIRouter()

# setup.py lives at backend/api/setup.py → .parent.parent.parent = movie-recommender/
BASE_DIR   = Path(__file__).resolve().parent.parent.parent
CSV_PATH   = str(BASE_DIR / "imdb_movies.csv")
INDEX_DIR  = str(BASE_DIR / "chroma_imdb")
STATS_DB   = str(BASE_DIR / "movie_stats.db")


@router.get("/setup/status", response_model=SetupStatusResponse)
async def setup_status():
    csv_exists   = os.path.isfile(CSV_PATH)
    index_exists = os.path.isdir(INDEX_DIR) and bool(os.listdir(INDEX_DIR))
    stats_exists = os.path.isfile(STATS_DB)
    return SetupStatusResponse(
        csv_exists=csv_exists,
        index_exists=index_exists,
        stats_db_exists=stats_exists,
    )


def _run_download():
    """Generator: yields SSE lines while downloading the dataset."""
    sys.path.insert(0, BASE_DIR)
    from datasets import load_dataset
    import pandas as pd

    yield "data: Starting download from HuggingFace (jquigl/imdb-genres)...\n\n"
    dataset = load_dataset("jquigl/imdb-genres", split="train")
    df = dataset.to_pandas()
    yield f"data: Downloaded {len(df):,} rows with {len(df.columns)} columns\n\n"
    df.to_csv(CSV_PATH, index=False)
    yield f"data: Saved to imdb_movies.csv\n\n"
    yield "data: DONE\n\n"


def _run_build_index(device: str):
    """Generator: yields SSE lines while building the ChromaDB index."""
    sys.path.insert(0, BASE_DIR)
    import shutil
    import pandas as pd
    from langchain_core.documents import Document
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    yield f"data: Loading dataset from {CSV_PATH}...\n\n"
    df = pd.read_csv(CSV_PATH).dropna(subset=["description"])
    yield f"data: Loaded {len(df):,} movies\n\n"

    def row_to_doc(row):
        raw_title = str(row["movie title - year"])
        if " - " in raw_title and raw_title[-4:].isdigit():
            title = raw_title[:raw_title.rfind(" - ")].strip()
            year  = raw_title[-4:]
        else:
            title = raw_title
            year  = ""
        genres      = str(row.get("expanded-genres", row.get("genre", "")))
        rating      = str(row.get("rating", ""))
        description = str(row["description"]).strip()
        try:
            rating_num = float(row.get("rating", 0.0))
        except (ValueError, TypeError):
            rating_num = 0.0
        try:
            year_num = int(year)
        except (ValueError, TypeError):
            year_num = 0
        genre_list = [g.strip().lower() for g in genres.split(",") if g.strip()]
        content = (
            f"{title} is a {genres} film released in {year}. "
            f"Rating: {rating}/10. "
            f"Plot: {description}"
        )
        return Document(
            page_content=content,
            metadata={
                "title":  title,
                "genres": genre_list,
                "year":   year_num,
                "rating": rating_num,
            }
        )

    yield "data: Building documents...\n\n"
    docs = [row_to_doc(row) for _, row in df.iterrows()]
    yield f"data: Built {len(docs):,} documents\n\n"

    yield f"data: Loading embedding model (device={device})...\n\n"
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 256 if device == "cuda" else 64},
    )

    yield "data: Removing old index...\n\n"
    shutil.rmtree(INDEX_DIR, ignore_errors=True)

    vectorstore = Chroma(
        persist_directory=INDEX_DIR,
        embedding_function=embeddings,
        collection_name="imdb_movies",
    )

    BATCH_SIZE = 1000
    total = len(docs)
    start = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        vectorstore.add_documents(batch)
        elapsed = time.time() - start
        done    = min(i + BATCH_SIZE, total)
        rate    = done / elapsed if elapsed > 0 else 1
        eta_min = (total - done) / rate / 60
        progress = {
            "done": done,
            "total": total,
            "pct": round(done / total * 100, 1),
            "eta_min": round(eta_min, 1),
        }
        yield f"data: {json.dumps(progress)}\n\n"

    final_count = vectorstore._collection.count()
    yield f"data: Indexed {final_count:,} movies into ChromaDB\n\n"

    # Rebuild stats DB
    yield "data: Building stats database...\n\n"
    # Re-import to pick up fresh index
    from chatbot import build_stats_database
    build_stats_database()
    yield "data: Stats database ready\n\n"

    # Reset GraphManager so next chat request loads the fresh index
    from core.graph_manager import GraphManager
    GraphManager.reset()
    yield "data: DONE\n\n"


@router.post("/setup/download")
async def download_dataset():
    async def event_stream():
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def producer():
            try:
                for line in _run_download():
                    asyncio.run_coroutine_threadsafe(queue.put(line), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        loop.run_in_executor(None, producer)

        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/setup/build-index")
async def build_index(req: BuildIndexRequest):
    async def event_stream():
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def producer():
            try:
                for line in _run_build_index(req.device):
                    asyncio.run_coroutine_threadsafe(queue.put(line), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        loop.run_in_executor(None, producer)

        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
