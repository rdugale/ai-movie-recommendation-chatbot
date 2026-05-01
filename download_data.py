from datasets import load_dataset
import pandas as pd

# Downloads ~30MB, no account needed
dataset = load_dataset("jquigl/imdb-genres", split="train")
df = dataset.to_pandas()

print(df.shape)          # (~28000, 5)
print(df.columns.tolist())
# ['movie title - year', 'genre', 'expanded-genres', 'rating', 'description']
print(df.head(2))

# Save locally so we don't re-download
df.to_csv("imdb_movies.csv", index=False)
print("Saved to imdb_movies.csv")