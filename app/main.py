import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Movie Recommender API")

# ── Load model from disk ─────────────────────────────────────────
print("Loading model...")
with open("models/recommender.pkl", "rb") as f:
    model = pickle.load(f)

# ── Load user-item matrix ────────────────────────────────────────
print("Loading user-item matrix...")
matrix = pd.read_csv("data/processed/user_item_matrix.csv", index_col="user_id")
matrix.index = matrix.index.astype(int)
matrix.columns = matrix.columns.astype(int)

# ── Load movie metadata ──────────────────────────────────────────
print("Loading movie metadata...")
genre_cols = [
    "unknown", "Action", "Adventure", "Animation", "Childrens",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "FilmNoir", "Horror", "Musical", "Mystery", "Romance",
    "SciFi", "Thriller", "War", "Western"
]
movie_cols = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"] + genre_cols
movies_df = pd.read_csv(
    "data/raw/ml-100k/u.item",
    sep="|", names=movie_cols,
    encoding="latin-1"
)
movies_df["genres"] = movies_df[genre_cols].apply(
    lambda row: " | ".join([g for g, v in zip(genre_cols, row) if v == 1]), axis=1
)
movies_df = movies_df[["movie_id", "title", "genres"]].set_index("movie_id")

# ── Reconstruct ratings matrix ───────────────────────────────────
print("Reconstructing ratings matrix...")
reconstructed = model.inverse_transform(model.transform(matrix))
reconstructed_df = pd.DataFrame(
    reconstructed,
    index=matrix.index,
    columns=matrix.columns
)
print("Model ready!")

# ── Routes ───────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Movie Recommender API is running"}

@app.get("/recommend/{user_id}")
def recommend(user_id: int, top_n: int = 10):
    if user_id not in reconstructed_df.index:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    already_rated = matrix.loc[user_id]
    already_rated = already_rated[already_rated > 0].index.tolist()

    user_preds = reconstructed_df.loc[user_id]
    user_preds = user_preds.drop(index=already_rated, errors="ignore")
    top_movies = user_preds.nlargest(top_n)

    recommendations = []
    for mid, score in top_movies.items():
        movie_info = movies_df.loc[mid] if mid in movies_df.index else None
        recommendations.append({
            "movie_id": int(mid),
            "title": movie_info["title"] if movie_info is not None else "Unknown",
            "genres": movie_info["genres"] if movie_info is not None else "Unknown",
            "predicted_rating": round(float(score), 2)
        })

    return {
        "user_id": user_id,
        "recommendations": recommendations
    }

@app.get("/health")
def health():
    return {"status": "healthy"}