# backend/src/services/train_lightweight_model.py

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import joblib
import os

# Load ratings
data_dir = "data"
ratings_path = os.path.join(data_dir, "ratings.csv")
ratings = pd.read_csv(ratings_path)

# Filter: Only top 1000 users
top_users = ratings['userId'].value_counts().head(1000).index
ratings = ratings[ratings['userId'].isin(top_users)]

# Filter: Only top 500 movies
top_movies = ratings['movieId'].value_counts().head(500).index
ratings = ratings[ratings['movieId'].isin(top_movies)]

# Prepare Surprise dataset
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Build and train model
trainset = data.build_full_trainset()
model = SVD(n_factors=10)
model.fit(trainset)

# Save small model
joblib.dump(model, "backend/src/services/recommender_model.pkl")
print("✅ Lightweight model saved successfully!")
