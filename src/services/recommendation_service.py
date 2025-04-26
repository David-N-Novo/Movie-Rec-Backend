import pandas as pd
import requests
import joblib

# Load model and movie data
MODEL_URL = "https://huggingface.co/DavidNNovo/Movie-Recomender/resolve/main/recommender_model.pkl"
MODEL_PATH = "recommender_model.pkl"
movies = pd.read_csv("data/movies.csv")

if not os.path.exists(MODEL_PATH):
    print("Downloading ML model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("Download complete!")

model = joblib.load(MODEL_PATH)

def recommend_movies(user_id, num_recs=5):
    movie_ids = movies['movieId'].tolist()
    predictions = [(movie_id, model.predict(user_id, movie_id).est) for movie_id in movie_ids]
    top_movies = sorted(predictions, key=lambda x: x[1], reverse=True)[:num_recs]

    recommended_titles = [
        movies[movies['movieId'] == movie_id].iloc[0]['title']
        for movie_id, _ in top_movies
    ]

    return recommended_titles