import pandas as pd
import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
import pickle
from pathlib import Path

DATA_DIR = Path("data")
MODEL_DIR = Path("backend/models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

def load_data():
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    return ratings, movies

def build_mapping_from_ratings(ratings):
    """
    Build movieId <-> index mappings **only** from movies that appear in ratings.
    """
    # Only movies that actually have ratings
    unique_movies = np.sort(ratings["movieId"].unique())
    print("num distinct rated movies:", len(unique_movies))
    print("max movieId in ratings:", unique_movies.max())

    # movieId -> 0..N-1
    movieId_to_index = {mid: i for i, mid in enumerate(unique_movies)}

    # index -> movieId as a dense array
    index_to_movieId = unique_movies.copy()

    return movieId_to_index, index_to_movieId

def build_user_mapping_from_ratings(ratings):
    unique_users = np.sort(ratings["userId"].unique())
    print("num distinct users:", len(unique_users))
    print("max userId in ratings:", unique_users.max())
    userId_to_index = {uid: i for i, uid in enumerate(unique_users)}
    return userId_to_index

def build_user_item_matrix(
    ratings: pd.DataFrame,
    userId_to_index: dict,
    movieId_to_index: dict,
):
    """
    Build a sparse User x Item matrix using the mappings above.
    We'll treat ratings >= 4.0 as implicit "likes".
    """
    interactions = ratings[ratings["rating"] >= 4.0].copy()

    # Map to compressed indices
    user_idx_s = interactions["userId"].map(userId_to_index)
    item_idx_s = interactions["movieId"].map(movieId_to_index)

    # Sanity: no NaNs after mapping
    assert user_idx_s.isna().sum() == 0, "Some userIds did not map!"
    assert item_idx_s.isna().sum() == 0, "Some movieIds did not map!"

    user_indices = user_idx_s.to_numpy(dtype=np.int64)
    item_indices = item_idx_s.to_numpy(dtype=np.int64)

    data = np.ones(len(interactions), dtype=np.float32)

    num_users = len(userId_to_index)
    num_items = len(movieId_to_index)

    # EXTRA sanity checks
    print("num_users (mapping):", num_users)
    print("num_items (mapping):", num_items)
    print("max user index in interactions:", user_indices.max())
    print("max item index in interactions:", item_indices.max())

    assert user_indices.max() < num_users, "User index out of range!"
    assert item_indices.max() < num_items, "Item index out of range!"

    user_item = sp.csr_matrix(
        (data, (user_indices, item_indices)),
        shape=(num_users, num_items),
    )

    print("user_item shape:", user_item.shape)
    return user_item

def train_als_model(user_item, factors=64, regularization=0.1, iterations=20):
    """
    Train an implicit ALS model.
    """
    # implicit expects item-user by default; we can transpose
    item_user = user_item.T.tocsr()

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations
    )

    
    model.fit(item_user)

    return model

def save_artifacts(model, movieId_to_index, index_to_movieId, movies_df):
    # Factors from implicit
    item_factors = model.item_factors
    user_factors = model.user_factors

    print(f"item_factors shape: {item_factors.shape}")
    print(f"user_factors shape: {user_factors.shape}")
    print(f"len(movieId_to_index): {len(movieId_to_index)}")
    print(f"len(index_to_movieId): {len(index_to_movieId)}")

    # Decide which factor matrix actually corresponds to MOVIES.
    # We know movies == len(index_to_movieId).
    if item_factors.shape[0] == len(index_to_movieId):
        print("Using item_factors as movie embeddings")
        movie_factors = item_factors
    elif user_factors.shape[0] == len(index_to_movieId):
        print("Using user_factors as movie embeddings")
        movie_factors = user_factors
    else:
        raise ValueError(
            f"Neither item_factors ({item_factors.shape[0]}) nor user_factors "
            f"({user_factors.shape[0]}) match number of movies ({len(index_to_movieId)})"
        )

    # Final sanity check
    assert movie_factors.shape[0] == len(index_to_movieId), (
        "movie_factors rows != number of movies in mapping!"
    )

    # Save the MOVIE factor matrix
    np.save(MODEL_DIR / "als_item_factors.npy", movie_factors)

    # Save mappings
    with open(MODEL_DIR / "movie_mappings.pkl", "wb") as f:
        pickle.dump(
            {
                "movieId_to_index": movieId_to_index,
                "index_to_movieId": index_to_movieId,
            },
            f,
        )

    # Save movie metadata
    movies_df.to_csv(MODEL_DIR / "movies_metadata.csv", index=False)


def main():
    ratings, movies = load_data()
    userId_to_index = build_user_mapping_from_ratings(ratings)
    movieId_to_index, index_to_movieId = build_mapping_from_ratings(ratings)

    user_item = build_user_item_matrix(ratings, userId_to_index, movieId_to_index)
    model = train_als_model(user_item)

    save_artifacts(model, movieId_to_index, index_to_movieId, movies)

if __name__ == "__main__":
    main()
