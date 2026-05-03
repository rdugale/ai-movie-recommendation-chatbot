import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import shutil
import time

df = pd.read_csv("imdb_movies.csv").dropna(subset=["description"])
print(f"Loaded {len(df)} movies with descriptions")

def row_to_doc(row):
    raw_title = str(row["movie title - year"])
    # Parse "Die Hard - 1988" format
    if " - " in raw_title and raw_title[-4:].isdigit():
        title = raw_title[:raw_title.rfind(" - ")].strip()
        year  = raw_title[-4:]
    else:
        title = raw_title
        year  = ""

    genres      = str(row.get("expanded-genres", row.get("genre", "")))
    rating      = str(row.get("rating", ""))
    description = str(row["description"]).strip()

    # 1. Safely convert rating to a float (or 0.0 if missing)
    try:
        rating_num = float(row.get("rating", 0.0))
    except (ValueError, TypeError):
        rating_num = 0.0
        
    # 2. Safely convert year to an integer (or 0 if missing)
    try:
        year_num = int(year)
    except (ValueError, TypeError):
        year_num = 0

    # Split the comma-separated string into a list
    genre_list = [g.strip().lower()  for g in genres.split(",") if g.strip()] 
    # Natural language format — best for embedding quality
    content = (
        f"{title} is a {genres} film released in {year}. "
        f"Rating: {rating}/10. "
        f"Plot: {description}"
    )
    
    return Document(
        page_content=content,
        metadata={
            "title":  title,
            "genres": genre_list, #genres,
            "year":   year_num,    # <--- Now saved as an Integer!
            "rating": rating_num,  # <--- Now saved as a Float!
        }
    )

docs = [row_to_doc(row) for _, row in df.iterrows()]
print(f"Built {len(docs)} documents")
print(f"\nSample:\n{docs[0].page_content[:200]}")

# use following code when system does not have GPU
# embeddings = HuggingFaceEmbeddings(
#     model_name="all-MiniLM-L12-v2",
#     model_kwargs={"device": "cpu"}, # when used CPU takes around 31 minutes
#     # encode_kwargs={"normalize_embeddings": True},
#     encode_kwargs={
#         "normalize_embeddings": True,
#         "batch_size": 64,                    # ← smaller batch for CPU
#     },
# )

# use following code when system have GPU (but make sure driver are installed and related python package which utilise GPU are also installed)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"},  # when used GPU takes around 11 minutes
    encode_kwargs={"normalize_embeddings": True, "batch_size": 256},
)

# Remove old index
shutil.rmtree("./chroma_imdb", ignore_errors=True)

BATCH_SIZE = 1000

# Create empty collection first
vectorstore = Chroma(
    persist_directory="./chroma_imdb",
    embedding_function=embeddings,
    collection_name="imdb_movies",
)

total = len(docs)
start = time.time()

for i in range(0, total, BATCH_SIZE):
    batch = docs[i : i + BATCH_SIZE]
    vectorstore.add_documents(batch)

    elapsed = time.time() - start
    done = min(i + BATCH_SIZE, total)
    rate = done / elapsed
    remaining = (total - done) / rate if rate > 0 else 0

    print(
        f"  Indexed {done:,}/{total:,} "
        f"({done/total*100:.1f}%) "
        f"ETA: {remaining/60:.1f} min",
        flush=True
    )

print(f"\nDone! {vectorstore._collection.count():,} movies indexed")