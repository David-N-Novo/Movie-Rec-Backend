import pandas as pd
import joblib

# Load model and movie data
#model = joblib.load("backend/src/services/recommender_model.pkl")
model = None
movies = pd.read_csv("data/movies.csv")

def recommend_movies(user_id, num_recs=5):
    movie_ids = movies['movieId'].tolist()
    predictions = [(movie_id, model.predict(user_id, movie_id).est) for movie_id in movie_ids]
    top_movies = sorted(predictions, key=lambda x: x[1], reverse=True)[:num_recs]

    recommended_titles = [
        movies[movies['movieId'] == movie_id].iloc[0]['title']
        for movie_id, _ in top_movies
    ]

    return recommended_titles