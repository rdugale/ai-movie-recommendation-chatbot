import os
import json
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
llm = OllamaLLM(model="phi3:mini", temperature=0.2, num_predict=500)

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
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# ── DB helper: batched metadata fetch ────────────────────────
def get_all_metadata(batch_size: int = 5000) -> list[dict]:
    """Fetch all ChromaDB metadata in batches to avoid SQLite variable limit."""
    all_meta = []
    offset   = 0
    total    = vectorstore._collection.count()

    while offset < total:
        result = vectorstore._collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"]
        )
        all_meta.extend(result["metadatas"])
        offset += batch_size

    return all_meta

def query_db_stats(user_msg: str) -> str | None:
    msg = user_msg.lower()

    stat_triggers = ["count", "how many", "number of", "total",
                     "least rated", "lowest rated", "most rated",
                     "highest rated", "best rated", "worst rated",
                     "top rated", "display", "show"]
    if not any(k in msg for k in stat_triggers):
        return None

    all_meta = get_all_metadata()
    total    = len(all_meta)

    genre_map = {
        "sci-fi": "sci-fi", "scifi": "sci-fi", "science fiction": "sci-fi",
        "action": "action", "comedy": "comedy", "drama": "drama",
        "horror": "horror", "thriller": "thriller", "romance": "romance",
        "animation": "animation", "animated": "animation",
        "crime": "crime", "adventure": "adventure", "fantasy": "fantasy",
        "documentary": "documentary", "biography": "biography",
    }

    matched_genre = next(
        (genre for kw, genre in genre_map.items() if kw in msg), None
    )

    results = []

    # ── Count section ─────────────────────────────────────────
    if any(k in msg for k in ["count", "how many", "number of", "total",
                               "list count"]):
        if matched_genre:
            count = sum(
                1 for m in all_meta
                if matched_genre.lower() in m.get("genres", "").lower()
            )
            results.append(
                f"Total **'{matched_genre}'** movies in ChromaDB: **{count:,}** "
                f"(out of {total:,} total)"
            )
        else:
            from collections import Counter
            genre_counter = Counter()
            for m in all_meta:
                for g in m.get("genres", "").split(","):
                    g = g.strip()
                    if g:
                        genre_counter[g] += 1
            top       = genre_counter.most_common(10)
            breakdown = "\n".join(f"  {g}: {c:,}" for g, c in top)
            results.append(
                f"Total movies in ChromaDB: **{total:,}**\n\n"
                f"Top 10 genres:\n{breakdown}"
            )

    # ── Rating section ────────────────────────────────────────
    rating_triggers = ["least rated", "lowest rated", "worst rated",
                       "most rated", "highest rated", "best rated",
                       "top rated", "least", "highest", "lowest"]

    if any(k in msg for k in rating_triggers):

        # ── Per-genre breakdown ───────────────────────────────
        per_genre_keywords = ["each genre", "every genre", "per genre",
                               "all genre", "each", "every", "per"]
        do_per_genre = any(k in msg for k in per_genre_keywords)

        if do_per_genre:
            # Collect all unique genres
            from collections import defaultdict
            genre_buckets = defaultdict(list)
            for m in all_meta:
                if m.get("rating", "") in ("", "nan", "N/A", None):
                    continue
                try:
                    rating = float(m["rating"])
                except (ValueError, KeyError):
                    continue
                for g in m.get("genres", "").split(","):
                    g = g.strip()
                    if g:
                        genre_buckets[g].append((rating, m))

            # Sort each genre bucket and pick best/worst
            genre_lines = []
            for genre in sorted(genre_buckets.keys()):
                bucket = sorted(genre_buckets[genre], key=lambda x: x[0])
                worst_r, worst = bucket[0]
                best_r,  best  = bucket[-1]
                genre_lines.append(
                    f"\n🎬 **{genre}** ({len(bucket):,} rated movies)\n"
                    f"   ⭐ Highest: {best.get('title','N/A')} "
                    f"({best.get('year','N/A')}) — {best_r}/10\n"
                    f"   💤 Lowest:  {worst.get('title','N/A')} "
                    f"({worst.get('year','N/A')}) — {worst_r}/10"
                )

            results.append("\n**Highest & Lowest rated movie per genre:**\n"
                           + "\n".join(genre_lines))

        else:
            # Single genre or overall — existing logic
            candidates = [
                m for m in all_meta
                if (matched_genre is None or
                    matched_genre.lower() in m.get("genres", "").lower())
                and m.get("rating", "") not in ("", "nan", "N/A", None)
            ]
            try:
                candidates.sort(key=lambda m: float(m["rating"]))
            except (ValueError, KeyError):
                pass

            if candidates:
                worst = candidates[0]
                best  = candidates[-1]
                label = f"'{matched_genre}'" if matched_genre else "all genres"
                results.append(
                    f"\n⭐ **Highest rated** {label} movie:\n"
                    f"   Title:   {best.get('title', 'N/A')} ({best.get('year', 'N/A')})\n"
                    f"   Genres:  {best.get('genres', 'N/A')}\n"
                    f"   Rating:  {best.get('rating', 'N/A')}/10\n"
                    f"\n💤 **Lowest rated** {label} movie:\n"
                    f"   Title:   {worst.get('title', 'N/A')} ({worst.get('year', 'N/A')})\n"
                    f"   Genres:  {worst.get('genres', 'N/A')}\n"
                    f"   Rating:  {worst.get('rating', 'N/A')}/10"
                )
            else:
                results.append(f"\nNo rated movies found for '{matched_genre}'.")
    return "\n".join(results) if results else None

# ── State ─────────────────────────────────────────────────────
class ChatbotState(TypedDict):
    messages:     Annotated[list[BaseMessage], add_messages]
    intent:       str
    context:      str
    liked_genres: list[str]
    seen_titles:  list[str]
    last_query:   str

# ── Node 1: classify ──────────────────────────────────────────

def classify_node(state: ChatbotState) -> dict:
    # Guard: if messages is empty something went wrong upstream
    if not state.get("messages"):
        return {"intent": "chitchat", "liked_genres": [], "last_query": ""}

    last_msg = state["messages"][-1].content

    # First try direct DB stats — no LLM needed
    db_answer = query_db_stats(last_msg)
    if db_answer:
        return {"intent": "db_stats", "last_query": last_msg,
                "liked_genres": state.get("liked_genres", [])}

    # Otherwise classify with LLM — use a tighter prompt
    # REPLACE the classify prompt with this — all { } in examples are escaped
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        'Reply with JSON only. No markdown, no explanation.\n'
        'Schema: {{"intent": "...", "genres": [...]}}\n\n'
        'intent values:\n'
        '  "recommend" — user wants movie suggestions or recommendations\n'
        '  "refine"    — user wants different/more/other results than before\n'
        '  "chitchat"  — anything else: greetings, explanations, opinions\n\n'
        'genres: list any film genres explicitly mentioned, else []\n\n'
        'Examples:\n'
        '  "recommend sci-fi movies"  -> {{"intent":"recommend","genres":["sci-fi"]}}\n'
        '  "suggest action films"     -> {{"intent":"recommend","genres":["action"]}}\n'
        '  "show me different ones"   -> {{"intent":"refine","genres":[]}}\n'
        '  "what is a black hole"     -> {{"intent":"chitchat","genres":[]}}\n'),
        ("human", "{message}")
    ])

    raw = (prompt | llm | StrOutputParser()).invoke({"message": last_msg})

    # Debug: uncomment to see raw LLM output
    # print(f"\n  [DEBUG classify raw]: {raw!r}")

    try:
        clean  = raw.strip().strip("```json").strip("```").strip()
        # Sometimes phi3 wraps in extra text — find the JSON object
        start  = clean.find("{")
        end    = clean.rfind("}") + 1
        parsed = json.loads(clean[start:end])
        intent = parsed.get("intent", "chitchat")
        genres = parsed.get("genres", [])
    except Exception:
        # Fallback: keyword match
        lower = last_msg.lower()
        if any(w in lower for w in ["recommend", "suggest", "suggest me",
                                     "show me", "find me", "what should i watch",
                                     "good movies", "movies to watch"]):
            intent = "recommend"
        elif any(w in lower for w in ["different", "more", "other", "else",
                                       "another", "show more"]):
            intent = "refine"
        else:
            intent = "chitchat"
        genres = []

    existing = state.get("liked_genres", [])
    merged   = list(set(existing + [g.lower() for g in genres]))

    print(f"  [intent: {intent} | genres: {merged}]")  # visible debug line
    return {"intent": intent, "liked_genres": merged, "last_query": last_msg}


# ── Node 2: DB stats answer ───────────────────────────────────
def db_stats_node(state: ChatbotState) -> dict:
    answer = query_db_stats(state["last_query"])
    return {"messages": [AIMessage(content=answer or "Could not compute that stat.")]}


# ── Node 3: retrieve ──────────────────────────────────────────
def retrieve_node(state: ChatbotState) -> dict:
    query = state["last_query"]
    seen  = state.get("seen_titles", [])
    liked = state.get("liked_genres", [])
    if liked:
        query = f"{query} genres: {', '.join(liked)}"

    docs       = retriever.invoke(query)
    fresh_docs = [d for d in docs if d.metadata.get("title", "") not in seen]
    if not fresh_docs:
        fresh_docs = docs

    context = "\n\n".join(
        f"[{i+1}] Title: {d.metadata['title']} ({d.metadata['year']})\n"
        f"     Genres: {d.metadata['genres']}\n"
        f"     Rating: {d.metadata['rating']}\n"
        f"     Plot: {d.page_content.split('Plot:')[-1].strip()[:250]}..."
        for i, d in enumerate(fresh_docs[:5])
    )
    new_titles = [d.metadata.get("title", "") for d in fresh_docs[:5]]
    all_seen   = list(set(seen + new_titles))
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
def generate_rag_node(state: ChatbotState) -> dict:
    liked     = state.get("liked_genres", [])
    pref_note = f"The user likes: {', '.join(liked)}. " if liked else ""

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a movie recommendation assistant. "
         f"{pref_note}"
         "Use ONLY the movies in the Context below — do not mention any other movies. "
         "For each movie state: title, year, genre, and one sentence about the plot.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder(variable_name="messages"),
    ])
    response = (prompt | llm | StrOutputParser()).invoke({
        "context":  state["context"],
        "messages": state["messages"],
    })
    return {"messages": [AIMessage(content=response)]}


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