import os
import json
import sqlite3
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from collections import Counter
# ── Setup ─────────────────────────────────────────────────────
# Add this at module level — loads once, stays in memory
_metadata_cache: list[dict] | None = None
STATS_DB = "./movie_stats.db"
USE_API_CLASSIFIER = False
USE_API_GENERATOR  = False
llm = OllamaLLM(model="phi3:mini", temperature=0.2, num_predict=4096)

if USE_API_CLASSIFIER or USE_API_GENERATOR:
    from llm_api import classify_with_api, generate_recommendation_with_api

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma(
    persist_directory="./chroma_imdb",
    embedding_function=embeddings,
    collection_name="imdb_movies",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

def build_stats_database():
    """
    Runs ONCE at startup. Reads all ChromaDB metadata
    and writes it into a fast local SQLite database.
    Takes ~10-15 seconds for 230K, then every query is instant.
    """
    print("Building stats database (one-time)...", flush=True)

    conn = sqlite3.connect(STATS_DB)
    c = conn.cursor()

    # Drop and recreate
    c.execute("DROP TABLE IF EXISTS movies")
    c.execute("""
        CREATE TABLE movies (
            id INTEGER PRIMARY KEY,
            title TEXT,
            year INTEGER,
            rating REAL,
            genres TEXT
        )
    """)
    c.execute("CREATE INDEX idx_genres ON movies(genres)")
    c.execute("CREATE INDEX idx_year ON movies(year)")
    c.execute("CREATE INDEX idx_rating ON movies(rating)")

    # Fetch all metadata from ChromaDB in batches
    all_meta = []
    offset = 0
    total = vectorstore._collection.count()

    while offset < total:
        result = vectorstore._collection.get(
            limit=5000,
            offset=offset,
            include=["metadatas"]
        )
        all_meta.extend(result["metadatas"])
        offset += 5000
        print(f"  Fetched {min(offset, total):,} / {total:,}", flush=True)

    # Insert into SQLite
    rows = []
    for m in all_meta:
        try:
            rating = float(m.get("rating", 0))
        except (ValueError, TypeError):
            rating = 0.0
        try:
            year = int(m.get("year", 0))
        except (ValueError, TypeError):
            year = 0

        raw_genres = m.get("genres", "")

        if isinstance(raw_genres, list):
            genres_str = ", ".join(raw_genres).lower()   # ["Action","Sci-Fi"] → "action, sci-fi"
        else:
            genres_str = str(raw_genres).lower()

        rows.append((
            m.get("title", ""),
            year,
            rating,
            genres_str
        ))

    c.executemany("INSERT INTO movies (title, year, rating, genres) VALUES (?,?,?,?)", rows)
    conn.commit()

    # Pre-build genre counts table for instant lookups
    c.execute("DROP TABLE IF EXISTS genre_counts")
    c.execute("""
        CREATE TABLE genre_counts AS
        SELECT genre, COUNT(*) as cnt
        FROM (
            SELECT TRIM(value) as genre
            FROM movies, json_each('["' || REPLACE(genres, ',', '","') || '"]')
            WHERE TRIM(value) != ''
        )
        GROUP BY genre
        ORDER BY cnt DESC
    """)
    conn.commit()
    conn.close()

    print(f"Stats database built with {len(rows):,} movies")

# ── DB helper: batched metadata fetch ────────────────────────
def get_all_metadata(batch_size: int = 5000) -> list[dict]:
    global _metadata_cache
    if _metadata_cache is not None:
        return _metadata_cache

    all_meta = []
    offset = 0
    total = vectorstore._collection.count()

    while offset < total:
        result = vectorstore._collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"]
        )
        all_meta.extend(result["metadatas"])
        offset += batch_size

    _metadata_cache = all_meta
    return _metadata_cache

def query_db_stats(user_msg: str) -> str | None:
    msg = user_msg.lower()

    stat_triggers = ["count", "how many", "number of", "total",
                     "least rated", "lowest rated", "most rated",
                     "highest rated", "best rated", "worst rated",
                     "top rated"]
    if not any(k in msg for k in stat_triggers):
        return None

    conn = sqlite3.connect(STATS_DB)
    c = conn.cursor()

    genre_map = {
        "sci-fi": "sci-fi", "scifi": "sci-fi", "science fiction": "sci-fi",
        "action": "action", "comedy": "comedy", "drama": "drama",
        "horror": "horror", "thriller": "thriller", "romance": "romance",
        "animation": "animation", "crime": "crime",
        "adventure": "adventure", "fantasy": "fantasy",
    }

    matched_genre = next(
        (genre for kw, genre in genre_map.items() if kw in msg), None
    )

    results = []

    # ── COUNT ─────────────────────────────────────────────
    if any(k in msg for k in ["count", "how many", "number of", "total","list count"]):
        if matched_genre:
            c.execute(
                "SELECT COUNT(*) FROM movies WHERE genres LIKE ?",
                (f"%{matched_genre}%",)
            )
            count = c.fetchone()[0]
            total = c.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
            results.append(
                f"Total **'{matched_genre}'** movies: **{count:,}** "
                f"(out of {total:,} total)"
            )
        else:
            c.execute("SELECT genre, cnt FROM genre_counts LIMIT 15")
            top = c.fetchall()
            total = c.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
            breakdown = "\n".join(f"  {g}: {c:,}" for g, c in top)
            results.append(
                f"Total movies: **{total:,}**\n\nTop genres:\n{breakdown}"
            )

    # ── HIGHEST / LOWEST RATED ────────────────────────────
    rating_triggers = ["highest rated", "best rated", "top rated",
                       "lowest rated", "worst rated", "least rated"]
    per_genre_triggers = ["each genre", "all genre", "every genre", "per genre",
                          "each category", "all categories"]
    if any(k in msg for k in rating_triggers):
        want_highest = any(k in msg for k in ["highest", "best", "top"])
        want_lowest  = any(k in msg for k in ["lowest", "worst", "least"])
        want_per_genre = any(k in msg for k in per_genre_triggers)

        queries = []
        if want_highest:
            queries.append(("DESC", "Highest"))
        if want_lowest:
            queries.append(("ASC", "Lowest"))
        if not queries:
            queries = [("DESC", "Highest")]

        if want_per_genre and not matched_genre:
            # Fetch all genres and return highest/lowest for each
            genres = [row[0] for row in c.execute(
                "SELECT genre FROM genre_counts ORDER BY cnt DESC"
            ).fetchall()]
            for g in genres:
                genre_results = []
                for order, label in queries:
                    row = c.execute(
                        f"SELECT title, year, rating, genres FROM movies "
                        f"WHERE genres LIKE ? AND rating > 0 "
                        f"ORDER BY rating {order} LIMIT 1",
                        (f"%{g}%",)
                    ).fetchone()
                    if row:
                        genre_results.append(
                            f"  {label}: {row[0]} ({row[1]}) — {row[2]}/10"
                        )
                if genre_results:
                    results.append(f"**{g}**:\n" + "\n".join(genre_results))
        else:
            for order, label in queries:
                if matched_genre:
                    row = c.execute(
                        f"SELECT title, year, rating, genres FROM movies "
                        f"WHERE genres LIKE ? AND rating > 0 "
                        f"ORDER BY rating {order} LIMIT 1",
                        (f"%{matched_genre}%",)
                    ).fetchone()
                else:
                    row = c.execute(
                        f"SELECT title, year, rating, genres FROM movies "
                        f"WHERE rating > 0 ORDER BY rating {order} LIMIT 1"
                    ).fetchone()

                if row:
                    results.append(
                        f"⭐ **{label} rated** {'(' + matched_genre + ')' if matched_genre else ''}:\n"
                        f"   {row[0]} ({row[1]}) — {row[2]}/10\n"
                        f"   Genres: {row[3]}"
                    )

    conn.close()
    return "\n".join(results) if results else None


# ── State ─────────────────────────────────────────────────────
class ChatbotState(TypedDict):
    messages:     Annotated[list[BaseMessage], add_messages]
    intent:       str
    context:      str
    liked_genres: list[str]
    seen_titles:  list[str]
    last_query:   str
    min_year:     float | None 
    max_year:     float | None
    min_rating:   float | None
    max_rating:   float | None

# ── Node 1: classify ──────────────────────────────────────────
def classify_node(state: ChatbotState) -> dict:
    if not state.get("messages"):
        return {"intent": "chitchat", "liked_genres": [], "last_query": ""}

    last_msg = state["messages"][-1].content

    # First try direct DB stats — no LLM needed
    db_answer = query_db_stats(last_msg)
    if db_answer:
        return {"intent": "db_stats", "last_query": last_msg,
                "liked_genres": state.get("liked_genres", []),"context": db_answer}

    # Updated Prompt to extract numbers
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        'Reply with JSON only. No markdown, no explanation.\n'
        'Schema: {{"intent": "...", "genres": [...], "min_year": int|null, "max_year": int|null, "min_rating": float|null, "max_rating": float|null}}\n\n'
        'intent values:\n'
        '  "recommend" — user wants movie suggestions\n'
        '  "refine"    — user wants different results\n'
        '  "chitchat"  — anything else\n\n'
        'genres: list any film genres explicitly mentioned, else []\n\n'
        'Examples:\n'
        '  "action movies after 2010 above 7" -> {{"intent":"recommend", "genres":["action"], "min_year": 2010, "max_year": null, "min_rating": 7.0}}\n'
        '  "sci-fi between 1990 and 2000" -> {{"intent":"recommend", "genres":["sci-fi"], "min_year": 1990, "max_year": 2000, "min_rating": null}}\n'
        '  "suggest comedy films" -> {{"intent":"recommend", "genres":["comedy"], "min_year": null, "max_year": null, "min_rating": null}}\n'),
        ("human", "{message}")
    ])

    raw = (prompt | llm | StrOutputParser()).invoke({"message": last_msg})

    # Safely declare default variables
    intent = "chitchat"
    genres = []
    min_year = None
    max_year = None
    min_rating = None
    max_rating = None
    
    if USE_API_CLASSIFIER:
        # API: Groq llama-3.3-70b (fast, accurate JSON)
        parsed = classify_with_api(last_msg)
        intent     = parsed["intent"]
        merged_genres = list(set(
            state.get("liked_genres", []) + parsed["genres"]
        ))
        
        intent = parsed.get("intent", "chitchat")
        genres = parsed.get("genres", [])
        min_year = parsed.get("min_year")
        max_year = parsed.get("max_year")
        min_rating = parsed.get("min_rating")
        max_rating = parsed.get("max_rating")
    else:

        try:
            clean  = raw.strip().strip("```json").strip("```").strip()
            start  = clean.find("{")
            end    = clean.rfind("}") + 1
            parsed = json.loads(clean[start:end])
            
            intent = parsed.get("intent", "chitchat")
            genres = parsed.get("genres", [])
            min_year = parsed.get("min_year")
            max_year = parsed.get("max_year")
            min_rating = parsed.get("min_rating")
            max_rating = parsed.get("max_rating")
            
        except Exception:
            # Fallback if the small LLM messes up the JSON formatting
            lower = last_msg.lower()
            if any(w in lower for w in ["recommend", "suggest", "find me"]):
                intent = "recommend"

    existing = state.get("liked_genres", [])
    merged   = list(set(existing + [g.lower() for g in genres]))

        # After the try/except block that parses the LLM output, add this:
    print(f"  [classify] intent={intent}, genres={merged}")
    print(f"  [classify] min_year={min_year}, max_year={max_year}, "
          f"min_rating={min_rating}, max_rating={max_rating}")

    # Return the new state variables!
    return {
        "intent": intent, 
        "liked_genres": merged, 
        "last_query": last_msg,
        "min_year": min_year,
        "max_year": max_year,
        "min_rating": min_rating,
        "max_rating": max_rating
    }

# ── Node 2: DB stats answer ───────────────────────────────────
def db_stats_node(state: ChatbotState) -> dict:
    # answer = query_db_stats(state["last_query"])
    # return {"messages": [AIMessage(content=answer or "Could not compute that stat.")]}
    # Use the pre-computed answer instead of re-querying
    answer = state.get("context", "Could not compute that stat.")
    return {"messages": [AIMessage(content=answer)]}

def retrieve_node(state: ChatbotState) -> dict:
    query = state["last_query"]
    seen  = state.get("seen_titles", [])
    liked = state.get("liked_genres", [])
    
    # 1. Build the metadata filter list dynamically
    filter_conditions = []

    if liked:
        # For a single genre, use direct filter
        if len(liked) == 1:
            filter_conditions.append(
                {"genres": {"$contains": liked[0]}}
            )
        else:
            # Multiple genres: match ANY of them
            genre_ors = [
                {"genres": {"$contains": g}} for g in liked
            ]
            filter_conditions.append({"$or": genre_ors})
    
    if state.get("min_year"):
        filter_conditions.append({"year": {"$gte": state["min_year"]}})
    if state.get("max_year"):
        filter_conditions.append({"year": {"$lte": state["max_year"]}})
    if state.get("min_rating"):
        filter_conditions.append({"rating": {"$gte": state["min_rating"]}})
    if state.get("max_rating"):
        filter_conditions.append({"rating": {"$lte": state["max_rating"]}})
    # 2. Format it into a Chroma-compatible dictionary
    where_clause = None
    if len(filter_conditions) == 1:
        where_clause = filter_conditions[0]
    elif len(filter_conditions) > 1:
        where_clause = {"$and": filter_conditions} # Chroma requires $and for multiple rules

    # 3. Search the vector database WITH the mathematical filters applied
    docs = vectorstore.similarity_search(
        query, 
        k=50, # Fetch more upfront so we have enough after filtering out seen movies
        filter=where_clause
    )

    seen_titles_set = set(seen)
    unique_docs = []
    for d in docs:
        title = d.metadata.get("title", "")
        if title not in seen_titles_set:
            unique_docs.append(d)
            seen_titles_set.add(title)

    if not unique_docs:
        unique_docs = [
            d for d in docs
            if d.metadata.get("title", "") not in set(seen)
        ]

    results_to_show = unique_docs[:10]

    # Clean formatting
    context_lines = []
    for i, d in enumerate(results_to_show):
        genres_raw = d.metadata.get('genres', '')
        if isinstance(genres_raw, list):
            genres_display = ", ".join(genres_raw)
        else:
            genres_display = str(genres_raw)

        context_lines.append(
            f"[{i+1}] {d.metadata['title']} ({d.metadata.get('year', 'N/A')})\n"
            f"    Genre: {genres_display}\n"
            f"    Rating: {d.metadata.get('rating', 'N/A')}/10\n"
            f"    Plot: {d.page_content.split('Plot:')[-1].strip()[:200]}"
        )

    context = "\n\n".join(context_lines)

    
    new_titles = [d.metadata.get("title", "") for d in results_to_show]
    all_seen   = list(set(seen + new_titles))

    print(f"  [retrieve] docs found: {len(docs)}, showing: {len(results_to_show)}")

    print(f"  [retrieve] query={query}")
    print(f"  [retrieve] where_clause={where_clause}")
    print(f"  [retrieve] docs found={len(docs)}")
    print(f"  [retrieve] fresh docs={len(unique_docs)}")

    
    return {"context": context, "seen_titles": all_seen}

# ── Node 4: refine ────────────────────────────────────────────
def refine_node(state: ChatbotState) -> dict:
    seen  = state.get("seen_titles", [])
    liked = state.get("liked_genres", [])
    query = f"different {' '.join(liked)} movies" if liked else "different movies"

    docs  = retriever.invoke(query)
    fresh = [d for d in docs if d.metadata.get("title", "") not in seen]
    if not fresh:
        fresh = docs

    context = "\n\n".join(
        f"[{i+1}] Title: {d.metadata['title']} ({d.metadata['year']})\n"
        f"     Genres: {d.metadata['genres']}\n"
        f"     Rating: {d.metadata['rating']}\n"
        f"     Plot: {d.page_content.split('Plot:')[-1].strip()[:250]}..."
        for i, d in enumerate(fresh[:5])
    )
    new_titles = [d.metadata.get("title", "") for d in fresh[:5]]
    all_seen   = list(set(seen + new_titles))
    return {"context": context, "seen_titles": all_seen}


# ── Node 5: generate RAG answer ───────────────────────────────
# following code was commented as local LLM was outputing 2-3 list of movies despite having passed more than it as context , if you can run large model then use following block 
# def generate_rag_node(state: ChatbotState) -> dict:
#     liked     = state.get("liked_genres", [])
#     pref_note = f"The user likes: {', '.join(liked)}. " if liked else ""
#     context   = state.get("context", "")

#     # If no context, tell the user directly — don't let LLM hallucinate
#     if not context.strip():
#         return {
#             "messages": [AIMessage(
#                 content="I couldn't find any movies matching your filters. "
#                         "Try widening your year range or lowering the rating threshold."
#             )]
#         }

#     prompt = ChatPromptTemplate.from_messages([
#         ("system",
#          "You are a movie recommendation assistant. "
#          f"{pref_note}"
#          "Use ONLY the movies in the Context below — do not mention any other movies. "
#          "For each movie state: title, year, genre, and one sentence about the plot.\n\n"
#          "Context:\n{context}"),
#         MessagesPlaceholder(variable_name="messages"),
#     ])
#     response = (prompt | llm | StrOutputParser()).invoke({
#         "context":  state["context"],
#         "messages": state["messages"],
#     })
#     return {"messages": [AIMessage(content=response)]}

# following code is added as alternative to LLM processing and to call extrenal API call for LLM
def generate_rag_node(state: ChatbotState) -> dict:
    context = state.get("context", "")

    # No context = no results
    if not context.strip():
        return {
            "messages": [AIMessage(
                content="No movies found matching your filters. "
                        "Try widening your year range or lowering the rating."
            )]
        }

    # Format directly — no LLM needed for structured data
    liked = state.get("liked_genres", [])
    if USE_API_GENERATOR:
        # API: let LLM format conversationally
        # Build conversation history as dicts for the API
        history = []
        for msg in state.get("messages", []):
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})

        reply = generate_recommendation_with_api(
            user_query=state.get("last_query", ""),
            context=context,
            liked_genres=liked,
            conversation_history=history
        )
    else:
        pref_note = f"Based on your interest in {', '.join(liked)}:\n\n" if liked else ""

        reply = pref_note + context

    return {"messages": [AIMessage(content=reply)]}


# ── Node 6: chitchat ──────────────────────────────────────────
def chitchat_node(state: ChatbotState) -> dict:
    liked = state.get("liked_genres", [])
    pref  = f"The user likes {', '.join(liked)} movies. " if liked else ""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         f"You are a friendly movie assistant. {pref}"
         "Answer naturally and concisely."),
        MessagesPlaceholder(variable_name="messages"),
    ])
    response = (prompt | llm | StrOutputParser()).invoke(
        {"messages": state["messages"]}
    )
    return {"messages": [AIMessage(content=response)]}


# ── Router ────────────────────────────────────────────────────
def route_intent(state: ChatbotState) -> Literal["retrieve", "refine",
                                                   "chitchat", "db_stats"]:
    intent = state.get("intent", "chitchat")
    return {
        "recommend": "retrieve",
        "refine":    "refine",
        "db_stats":  "db_stats",
    }.get(intent, "chitchat")

def init_stats_if_needed():
    if os.path.exists(STATS_DB):
        conn = sqlite3.connect(STATS_DB)
        count = conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
        conn.close()
        chroma_count = vectorstore._collection.count()

        if count == chroma_count:
            print(f"Stats DB ready ({count:,} movies)")
            return
        else:
            print(f"Stats DB stale ({count:,} vs {chroma_count:,} in ChromaDB), rebuilding...")

    build_stats_database()

# Call once at startup
init_stats_if_needed()

# ── Build graph ───────────────────────────────────────────────
def build_graph():
    builder = StateGraph(ChatbotState)

    builder.add_node("classify",     classify_node)
    builder.add_node("retrieve",     retrieve_node)
    builder.add_node("refine",       refine_node)
    builder.add_node("generate_rag", generate_rag_node)
    builder.add_node("chitchat",     chitchat_node)
    builder.add_node("db_stats",     db_stats_node)

    builder.set_entry_point("classify")

    builder.add_conditional_edges(
        "classify",
        route_intent,
        {
            "retrieve": "retrieve",
            "refine":   "refine",
            "chitchat": "chitchat",
            "db_stats": "db_stats",
        }
    )

    builder.add_edge("retrieve",     "generate_rag")
    builder.add_edge("refine",       "generate_rag")
    builder.add_edge("generate_rag", END)
    builder.add_edge("chitchat",     END)
    builder.add_edge("db_stats",     END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# ── CLI ───────────────────────────────────────────────────────
def run_cli():
    print("Loading model and index...", flush=True)
    graph  = build_graph()
    config = {"configurable": {"thread_id": "session-1"}}
    first_run = True

    print("\n🎬 Movie Recommendation Chatbot")
    print("Powered by LangGraph + RAG + phi3:mini (fully local)")
    print("Try: 'recommend sci-fi movies', 'show me different ones', 'quit'\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        if first_run:
            input_state = {
                "intent":       "",
                "context":      "",
                "liked_genres": [],
                "seen_titles":  [],
                "last_query":   "",
                "messages":     [HumanMessage(content=user_input)],
            }
            first_run = False
        else:
            input_state = {"messages": [HumanMessage(content=user_input)]}

        graph.invoke(input_state, config=config)
        result = graph.get_state(config).values
        reply  = result["messages"][-1].content
        print(f"\nAgent: {reply}")

        liked = result.get("liked_genres", [])
        seen  = result.get("seen_titles",  [])
        if liked:
            print(f"\n  [Preferences: {', '.join(liked)}]")
        if seen:
            print(f"  [Recommended so far: {len(seen)} movies]")
        print()


if __name__ == "__main__":
    run_cli()