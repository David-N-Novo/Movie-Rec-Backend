import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

MODEL_DIR = Path("src/models")

class MovieRecommender:
    def __init__(self):
        self.item_factors = np.load(MODEL_DIR / "als_item_factors.npy")

        with open(MODEL_DIR / "movie_mappings.pkl", "rb") as f:
            mappings = pickle.load(f)

        # movieId -> index (0..N-1)
        self.movieId_to_index = mappings["movieId_to_index"]

        # index -> movieId 
        self.index_to_movieId = np.array(mappings["index_to_movieId"], dtype=np.int64)

        # rows of item_factors must match number of movies
        num_items = self.item_factors.shape[0]
        assert num_items == self.index_to_movieId.shape[0], (
            f"item_factors has {num_items} rows but index_to_movieId has "
            f"{self.index_to_movieId.shape[0]} entries"
        )

        self.movies_df = pd.read_csv(MODEL_DIR / "movies_metadata.csv")

    def _movie_ids_to_indices(self, movie_ids):
        indices = []
        for mid in movie_ids:
            idx = self.movieId_to_index.get(mid)
            if idx is not None:
                indices.append(idx)
        return indices

    def recommend_from_likes(self, liked_movie_ids, top_k=10, exclude_liked=True):
        liked_indices = self._movie_ids_to_indices(liked_movie_ids)

        if not liked_indices:
            # fallback: sample popular/random
            return self.movies_df.sample(top_k)

        liked_vecs = self.item_factors[liked_indices]
        user_vec = liked_vecs.mean(axis=0, keepdims=True)

        sims = cosine_similarity(user_vec, self.item_factors)[0]
        ranked_indices = np.argsort(-sims)

        if exclude_liked:
            liked_set = set(liked_indices)
            ranked_indices = [i for i in ranked_indices if i not in liked_set]

        top_indices = np.array(ranked_indices[:top_k])

        top_movie_ids = self.index_to_movieId[top_indices]

        # Join with metadata, preserving order
        recs = (
            self.movies_df[self.movies_df["movieId"].isin(top_movie_ids)]
            .set_index("movieId")
            .loc[top_movie_ids]
            .reset_index()
        )
        return recs