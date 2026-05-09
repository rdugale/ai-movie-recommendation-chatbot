# 🎬 Movie Recommendation Chatbot

A **fully local, privacy-first AI-powered movie recommendation chatbot** built with LangChain, LangGraph, RAG (Retrieval-Augmented Generation), ChromaDB, and Ollama — no paid API required.

---

## What kind of application is this?

This is an **Agentic RAG Chatbot** — a conversational AI system that combines three modern AI engineering techniques:

| Technique | What it does in this project |
|---|---|
| **RAG** (Retrieval-Augmented Generation) | Fetches relevant movies from a local vector database before answering, so the LLM is grounded in real data instead of hallucinating |
| **LangGraph Agent** | Decides dynamically whether to retrieve movies, refine previous results, answer a stats query, or just chat — routing each message to the right handler |
| **Local LLM via Ollama** | Runs `phi3:mini` entirely on your machine — no OpenAI, no API costs, no data leaving your computer |

### Tech stack

```
User message
     │
     ▼
LangGraph Agent  ──→  Intent classifier (phi3:mini)
     │
     ├── recommend  ──→  ChromaDB retriever  ──→  RAG answer (phi3:mini)
     ├── refine     ──→  ChromaDB retriever  ──→  RAG answer (phi3:mini)
     ├── db_stats   ──→  Direct metadata query (no LLM)
     └── chitchat   ──→  General answer (phi3:mini)
```

- **LangChain** — prompt templates, chains, output parsers
- **LangGraph** — stateful agent with conditional routing and memory
- **ChromaDB** — local vector store with HNSW index (238k+ movies)
- **sentence-transformers** (`all-MiniLM-L6-v2`) — local embeddings, runs on CPU
- **Ollama + phi3:mini** — local LLM, runs on GPU (GTX 1650 / 4GB VRAM)
- **FastAPI + React/TypeScript + Tailwind v3** — modern web UI (new)
- **Gradio** — lightweight browser-based chat UI (original)
- **Dataset** — [jquigl/imdb-genres](https://huggingface.co/datasets/jquigl/imdb-genres) from HuggingFace (~238k movies with title, genre, rating, plot)

---

## Features

- 🎯 **Smart intent routing** — detects whether you want recommendations, refinements, stats, or general chat
- 🔍 **RAG retrieval** — answers are grounded in the actual movie database, not LLM memory
- 🔄 **"Show me different ones"** — refine loop avoids re-recommending already-seen movies
- 🧠 **Preference memory** — learns your liked genres across the conversation
- 📊 **Database analytics** — query counts, highest/lowest rated movies per genre directly from ChromaDB
- 💻 **100% local** — no API keys, no internet required after setup, no data sent anywhere
- ⚛️ **React/TypeScript UI** — dark cinema-themed web app with interactive setup wizard
- 🖥️ **Gradio UI** — lightweight browser interface at `http://localhost:7860` (original)

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/rdugale/ai-movie-recommendation-chatbot.git
cd ai-movie-recommendation-chatbot
```

### 2. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install langchain langgraph langchain-ollama langchain-chroma \
            langchain-huggingface chromadb sentence-transformers \
            datasets gradio pandas scikit-learn python-dotenv groq \
            torch torchvision torchaudio
```
append --index-url https://download.pytorch.org/whl/cu124  after torchaudio when want to use GPU

### 4. Install Ollama and pull the model

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama list

# Pull phi3:mini (approx 2.3 GB)
ollama pull phi3:mini

# Test the model (optional)
ollama run phi3:mini
# Type /bye to exit
```

### 5. Start Ollama server

```bash
# Run in a separate terminal — keep it running
ollama serve
```

---

## Project Structure

```
movie-recommender/
├── download_data.py        # Step 1 — fetch dataset from HuggingFace
├── build_index.py          # Step 2 — embed movies and build ChromaDB index
├── chatbot.py              # Core — LangGraph agent (nodes, edges, routing)
├── llm_api.py              # Optional Groq API integration
├── gradio_app.py           # UI — Gradio web interface (original)
├── imdb_movies.csv         # Downloaded dataset (created by download_data.py)
├── movie_stats.db          # SQLite stats cache (created at first run)
├── chroma_imdb/            # ChromaDB vector index (created by build_index.py)
│   ├── chroma.sqlite3
│   └── <uuid>/
├── backend/                # FastAPI backend (new)
│   ├── main.py             # FastAPI app, CORS, lifespan startup
│   ├── run.py              # Dev launcher (watches parent dir for reload)
│   ├── requirements.txt
│   ├── core/
│   │   └── graph_manager.py   # Singleton wrapping the LangGraph graph
│   ├── api/
│   │   ├── chat.py            # POST /api/chat
│   │   ├── session.py         # POST /api/session/new
│   │   ├── stats.py           # GET  /api/stats
│   │   └── setup.py           # GET/POST /api/setup/* (SSE streaming)
│   └── models/
│       └── schemas.py         # Pydantic request/response models
└── frontend/               # React + TypeScript + Tailwind v3 (new)
    ├── src/
    │   ├── pages/
    │   │   ├── SetupPage.tsx   # Interactive setup wizard
    │   │   └── ChatPage.tsx    # Full-screen cinema-themed chat
    │   ├── components/         # MessageBubble, InputBar, ProgressLog, ...
    │   ├── hooks/              # useChat, useSession, useSetup
    │   └── api/                # Typed fetch wrappers
    ├── tailwind.config.js
    └── vite.config.ts
```

---

## Running the project

There are **two ways** to run this project:
- **Option A — React + FastAPI** (new, recommended): full-featured web app with an interactive setup wizard
- **Option B — Gradio / CLI** (original): quick single-file launch

---

### Option A — React + FastAPI web app

#### Prerequisites (one-time)

Make sure Node.js ≥ 18 is installed:
```bash
node --version    # should print v18 or higher
npm --version
```

Install frontend dependencies (one-time):
```bash
cd frontend
npm install
cd ..
```

#### Step 1 — Start the FastAPI backend

```bash
# From the movie-recommender/ root
source venv/bin/activate
cd backend
python run.py or uvicorn main:app --reload
```

The backend starts at **http://localhost:8000**.  
`run.py` uses `--reload` and watches both `backend/` and the parent directory, so any change to `chatbot.py` or backend files restarts automatically.

#### Step 2 — Start the React frontend

Open a **second terminal**:

```bash
# From the movie-recommender/ root
cd frontend
npm run dev
```

The frontend starts at **http://localhost:5173**.

#### Step 3 — Open the app

Go to **http://localhost:5173** in your browser.

**First time only:** The app detects that the dataset/index is missing and shows the **Setup Wizard**:

1. **Download Dataset** — click the button; streams progress as it downloads ~238k movies from HuggingFace and saves `imdb_movies.csv`
2. **Build Vector Index** — choose GPU (cuda, ~11 min) or CPU (~31 min); click the button; a live progress bar and log stream the indexing
3. **Go to Chat** — once setup is complete, click to enter the chat

**On subsequent runs:** The app skips setup and goes straight to the chat.

---

### Option B — Original CLI / Gradio

> **Steps 1 and 2 only need to be run once.** After the index is built, skip to step 3 or 4.

#### Step 1 — Download the dataset (run once)

```bash
python3 download_data.py
```

Downloads ~238k movies and saves to `imdb_movies.csv`.

#### Step 2 — Build the vector index (run once)

```bash
python3 build_index.py
```

Embeds all movies using `all-MiniLM-L6-v2` and stores them in ChromaDB. Takes ~11–31 minutes depending on hardware.

#### Step 3a — Run the CLI chatbot

```bash
python3 chatbot.py
```

#### Step 3b — Run the Gradio web UI

```bash
python3 gradio_app.py
```

Open your browser at **http://localhost:7860**

---

## Example queries

```
Recommend some thriller movies
I love sci-fi and action films, what should I watch?
Show me something different
What makes Christopher Nolan films special?
list count of all sci-fi movies and display least rated and highest rated sci-fi movie
list count of all genre movies and display least rated and highest rated movie from each genre
how many comedy movies are in the database?
```

---

## How it works — under the hood

### 1. Data pipeline (offline, runs once)
```
imdb_movies.csv
      │
      ▼
Convert each row → LangChain Document
(natural language: "Inception is a Sci-Fi, Thriller film released in 2010. Rating: 8.8. Plot: ...")
      │
      ▼
Embed with all-MiniLM-L6-v2  (384-dimensional vectors, runs on CPU)
      │
      ▼
Store in ChromaDB  (HNSW index on disk at ./chroma_imdb)
```

### 2. Query pipeline (runs on every message)
```
User message
      │
      ▼
classify_node  →  phi3:mini decides: recommend / refine / db_stats / chitchat
      │
      ├─ recommend/refine → ChromaDB similarity search → top-k movies retrieved
      │                          │
      │                          ▼
      │                   generate_rag_node → phi3:mini answers using ONLY retrieved movies
      │
      ├─ db_stats → direct ChromaDB metadata query (no LLM, instant, accurate)
      │
      └─ chitchat → phi3:mini answers from general knowledge
```

### 3. Memory & state
LangGraph's `MemorySaver` checkpoints the full state after every turn:
- `liked_genres` — genres mentioned by the user, accumulated across turns
- `seen_titles` — movies already recommended, excluded from future results
- `messages` — full conversation history passed to the LLM for context

---

## Hardware notes

| Component | Runs on | Memory used |
|---|---|---|
| phi3:mini (LLM) | GPU (CUDA) | ~2.5 GB VRAM |
| all-MiniLM-L6-v2 (embeddings) | CPU | ~300 MB RAM |
| ChromaDB (vector search) | CPU | ~200 MB RAM |

The embeddings deliberately run on CPU to leave all VRAM free for Ollama. This is the recommended setup for 4 GB VRAM cards like the GTX 1650.

---

## Added API LLM call for classify and generate node

using Grok for API LLM call for classify and generate node logic, which can be turned on/off using boolean flag , create .env file add GROQ_API_KEY key, to create and use free API key visit - [console.groq.com](https://console.groq.com/) and create account

---

## Acknowledgements

- [LangChain](https://python.langchain.com) — LLM application framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) — stateful agent framework
- [Ollama](https://ollama.com) — local LLM serving
- [ChromaDB](https://www.trychroma.com) — local vector database
- [HuggingFace Datasets](https://huggingface.co/datasets/jquigl/imdb-genres) — IMDb genres dataset
- [FastAPI](https://fastapi.tiangolo.com) — Python web framework for the backend API
- [React](https://react.dev) + [TypeScript](https://www.typescriptlang.org) — frontend UI
- [Tailwind CSS v3](https://tailwindcss.com) — utility-first CSS framework
- [Vite](https://vitejs.dev) — frontend build tool
- [Gradio](https://gradio.app) — original lightweight web UI framework