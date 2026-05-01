import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

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
            "year":   year,
            "genres": genres,
            "rating": rating,
        }
    )

docs = [row_to_doc(row) for _, row in df.iterrows()]
print(f"Built {len(docs)} documents")
print(f"\nSample:\n{docs[0].page_content[:200]}")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("\nBuilding ChromaDB index (takes ~5-10 min for 28k movies)...")
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_imdb",
    collection_name="imdb_movies",
)
print(f"Done! Indexed {vectorstore._collection.count()} movies")