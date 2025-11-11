import os
import requests
from flask import Blueprint, request, jsonify
#from src.services.recommendation_service import recommend_movies REPLACE
from src.services.recommender import MovieRecommender

recommendation_bp = Blueprint('recommendation', __name__)

recommender = MovieRecommender()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

@recommendation_bp.route('/api/trending', methods=['GET'])
def get_trending_movies():
    if not TMDB_API_KEY:
        return jsonify({"error": "TMDB_API_KEY not configured"}), 500
    
    try:
        url = "https://api.themoviedb.org/3/trending/movie/day"
        params = {"api_key": TMDB_API_KEY, "language": "en-US"}
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        movies = []
        for m in results:
            poster_path = m.get("poster_path")
            movies.append({
                "id": m.get("id"),
                "title": m.get("title"),
                "overview": m.get("overview"),
                "poster_url": f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None,
                "release_date": m.get("release_date"),
            })

        return jsonify(movies), 200
    except Exception as e:
        print(f"Error fetching trending movies:", e)
        return jsonify([]), 200
        #return jsonify({"error": "Failed to fetch trending movies"}), 500

@recommendation_bp.route('/api/recommend', methods=['POST'])
def get_recommendations():
    data = request.get_json(force=True)
    liked_movie_ids = data.get("liked_movie_ids", [])
    top_k = data.get("top_k", 10)

    recs_df = recommender.recommend_from_likes(liked_movie_ids, top_k=top_k)
    recs = [
        {
            "movieId": int(row.movieId),
            "title": row.title,
            "genres": row.genres,
        }
        for _, row in recs_df.iterrows()
    ]
    return jsonify({"recommendations": recs})

@recommendation_bp.route('/movies', methods=['GET'])
def get_movies():
    df = recommender.movies_df[["movieId", "title", "genres"]]
    movies = df.to_dict(orient="records")
    return jsonify(movies)